import os
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt


# 1. 모델 구조
class ECG_CNN(nn.Module):
    def __init__(self, in_channels=2, num_classes=1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        feats = self.feature_extractor(x)
        pooled = self.global_pool(feats).squeeze(-1)
        logits = self.classifier(pooled)
        return logits.view(-1)


# 2. IG 계산 함수
def compute_ig(model, arr):
    if arr.shape == (3600, 2):
        arr = arr.T

    x = torch.tensor(arr).float().unsqueeze(0)
    baseline = torch.zeros_like(x)

    ig = IntegratedGradients(model)
    attr, _ = ig.attribute(x, baselines=baseline, return_convergence_delta=True)

    return attr.mean(dim=1).detach().numpy()[0]  # (3600,)


# 3. 정량 분석 함수들
def compute_iou_shift(clean_imp, noise_imp, percentile=90):
    thr_c = np.percentile(np.abs(clean_imp), percentile)
    thr_n = np.percentile(np.abs(noise_imp), percentile)
    mask_c = (np.abs(clean_imp) >= thr_c).astype(int)
    mask_n = (np.abs(noise_imp) >= thr_n).astype(int)
    inter = (mask_c & mask_n).sum()
    union = (mask_c | mask_n).sum()
    iou = inter / union if union != 0 else 0
    shift = 1 - iou
    return iou, shift

def compute_qrs_strength(clean_imp, noise_imp, start=1500, end=1700):
    clean_q = np.sum(np.abs(clean_imp[start:end]))
    noise_q = np.sum(np.abs(noise_imp[start:end]))
    return noise_q / clean_q if clean_q != 0 else 0

def compute_nar(noise_imp):
    thr = np.percentile(np.abs(noise_imp), 95)
    nar = (np.abs(noise_imp) >= thr).mean()
    return nar


# 4. Test 전체 정량 분석
def analyze_dataset(model, clean_dir, noise_dir):
    metrics = {"IoU": [], "Shift": [], "QRS": [], "NAR": []}

    for fname in os.listdir(clean_dir):
        if not fname.endswith(".npy"):
            continue

        clean_path = os.path.join(clean_dir, fname)
        noise_path = os.path.join(noise_dir, fname)

        if not os.path.exists(noise_path):
            continue

        clean_arr = np.load(clean_path)
        noise_arr = np.load(noise_path)

        clean_imp = compute_ig(model, clean_arr)
        noise_imp = compute_ig(model, noise_arr)

        iou, shift = compute_iou_shift(clean_imp, noise_imp)
        qrs = compute_qrs_strength(clean_imp, noise_imp)
        nar = compute_nar(noise_imp)

        metrics["IoU"].append(iou)
        metrics["Shift"].append(shift)
        metrics["QRS"].append(qrs)
        metrics["NAR"].append(nar)

    return {k: float(np.mean(v)) for k, v in metrics.items()}


# 5. Noise별 정량 비교 시각화
def plot_noise_comparison(results, model_name):
    metrics = ["IoU", "Shift", "QRS", "NAR"]
    noise_types = ["BW", "MA", "EM"]

    for metric in metrics:
        values = [
            results["BW"][metric],
            results["MA"][metric],
            results["EM"][metric],
        ]

        plt.figure(figsize=(7, 4))
        plt.bar(noise_types, values, color=["#4A90E2", "#50E3C2", "#F5A623"])
        plt.title(f"{model_name} — Noise Comparison ({metric})")
        plt.ylabel(metric)
        plt.ylim(0, max(values) * 1.2)
        plt.tight_layout()
        plt.show()


# 6. 실행 (경로 수정 !!!!!)
clean_test = r"C:\경로\clean_segments\test"
bw_test    = r"C:\경로\bw_noise\test"
ma_test    = r"C:\경로\ma_noise\test"
em_test    = r"C:\경로\em_noise\test"

for model_name, weight in [("M0", "M0_clean.pt"), ("M1", "M1_noisy.pt")]:
    print(f"\n===== Running {model_name} =====")
    
    model = ECG_CNN()
    model.load_state_dict(torch.load(weight, map_location="cpu"))
    model.eval()

    results = {
        "BW": analyze_dataset(model, clean_test, bw_test),
        "MA": analyze_dataset(model, clean_test, ma_test),
        "EM": analyze_dataset(model, clean_test, em_test),
    }

    # 정량분석 결과 출력
    print(f"\n==== {model_name} 정량분석 결과 ====")
    for noise in ["BW", "MA", "EM"]:
        print(f"{noise}: IoU={results[noise]['IoU']:.4f} | "
              f"Shift={results[noise]['Shift']:.4f} | "
              f"QRS={results[noise]['QRS']:.4f} | "
              f"NAR={results[noise]['NAR']:.4f}")

    # 노이즈별 비교 그래프 시각화
    plot_noise_comparison(results, model_name)