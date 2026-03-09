import os
import glob
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam


# =========================
# 1. 유틸 함수
# =========================

def fix_ecg_shape(arr: np.ndarray) -> np.ndarray:
    """
    ECG 세그먼트 shape 통일:
    - (2, 3600) 그대로 사용
    - (3600, 2) -> (2, 3600)로 transpose
    - (3600,) -> (1, 3600)
    """
    if arr.ndim == 2:
        if arr.shape == (2, 3600):
            return arr
        elif arr.shape == (3600, 2):
            return arr.T
        else:
            # 예외 케이스: 채널 수/길이가 다르면 여기서 맞춰줘야 함
            # 일단 (channels, length) 형태로 가정
            if arr.shape[0] < arr.shape[1]:
                return arr
            else:
                return arr.T
    elif arr.ndim == 1:
        return arr.reshape(1, -1)
    else:
        raise ValueError(f"Unexpected ECG shape: {arr.shape}")


def zscore_normalize(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    세그먼트별 z-score 정규화 (채널 단위)
    arr: (C, L)
    """
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    return (arr - mean) / (std + eps)


# =========================
# 2. Dataset 정의
# =========================

class ECGDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        noise_types: List[str],
        labels_csv: str,
        normalize: bool = True,
    ):
        """
        data_root: /home/ssu/mediupxai/data
        split: "train" / "val" / "test"
        noise_types: ["clean_segments"], ["bw_noise", "em_noise", "ma_noise"], ...
        labels_csv: all_labels_split.csv 경로
        """
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.noise_types = noise_types
        self.normalize = normalize

        df = pd.read_csv(labels_csv)

        # 현재 CSV 컬럼: id, label, source, split
        if not {"id", "split", "label"}.issubset(df.columns):
            raise ValueError(
                "CSV에 'id', 'split', 'label' 컬럼이 있어야 합니다. "
                "datacheck.py 출력과 다르면 이 부분을 수정하세요."
            )

        # split 필터링 (train / val / test)
        df = df[df["split"] == split].copy()

        # id -> label 매핑 (id가 npy 파일 이름과 같다고 가정)
        self.label_map = dict(zip(df["id"].astype(str), df["label"].astype(int)))


        # ----- 파일 리스트 만들기 -----
        self.samples: List[Tuple[str, int]] = []

        for noise in noise_types:
            split_dir = os.path.join(data_root, noise, split)
            if not os.path.isdir(split_dir):
                print(f"[WARN] {split_dir} not found, skip.")
                continue

            # .npy 파일 전부
            npy_files = glob.glob(os.path.join(split_dir, "*.npy"))
            for fpath in npy_files:
                # 파일 이름에서 segment_id 추출 (확장자 제거)
                seg_id = os.path.splitext(os.path.basename(fpath))[0]
                if seg_id in self.label_map:
                    label = self.label_map[seg_id]
                    self.samples.append((fpath, label))
                else:
                    # 라벨이 없으면 스킵
                    pass

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for split={split}, noise_types={noise_types}. "
                f"CSV 매핑/폴더 구조를 다시 확인해주세요."
            )

        print(
            f"[ECGDataset] split={split}, noise_types={noise_types}, "
            f"num_samples={len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        arr = np.load(fpath)  # numpy array

        arr = fix_ecg_shape(arr)        # (C, L)
        if self.normalize:
            arr = zscore_normalize(arr) # 정규화

        # torch tensor로 변환
        x = torch.from_numpy(arr).float()  # (C, L)
        y = torch.tensor(label, dtype=torch.float32)  # BCEWithLogitsLoss용

        return x, y


# =========================
# 3. 모델 정의 (간단 1D CNN)
# =========================

class ECGCNN(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 3600 -> 1800

            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 1800 -> 900

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 900 -> 450
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)  # (B, 64, 1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, C, L)
        feats = self.feature_extractor(x)
        pooled = self.global_pool(feats).squeeze(-1)  # (B, 64)
        logits = self.classifier(pooled)  # (B, 1)
        return logits.view(-1)  # (B,)


# =========================
# 4. 학습 / 평가 루프
# =========================

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return acc, f1, auc


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(y.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    acc, f1, auc = compute_metrics(np.array(all_targets), np.array(all_probs))
    return epoch_loss, acc, f1, auc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_targets.extend(y.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    acc, f1, auc = compute_metrics(np.array(all_targets), np.array(all_probs))
    return epoch_loss, acc, f1, auc


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=30,
    lr=1e-3,
    pos_weight=None,
    model_name="model",
    save_dir="./models",
):
    os.makedirs(save_dir, exist_ok=True)

    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = Adam(model.parameters(), lr=lr)

    best_val_auc = -1.0
    patience = 5
    no_improve = 0
    best_model_path = os.path.join(save_dir, f"{model_name}.pt")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_f1, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, val_auc = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"[{model_name}] Epoch {epoch:02d} | "
            f"TrainLoss={train_loss:.4f} ACC={train_acc:.3f} F1={train_f1:.3f} AUC={train_auc:.3f} || "
            f"ValLoss={val_loss:.4f} ACC={val_acc:.3f} F1={val_f1:.3f} AUC={val_auc:.3f}"
        )

        # Early stopping: val AUC 기준
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✅ Saved best {model_name} (val AUC={best_val_auc:.3f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  🛑 Early stopping {model_name} at epoch {epoch}")
                break

    print(f"[{model_name}] Best val AUC = {best_val_auc:.3f}, saved to {best_model_path}")
    return best_model_path


# =========================
# 5. M0 / M1 학습 스크립트
# =========================

def compute_pos_weight_from_loader(loader):
    """
    데이터 imbalance용 pos_weight 계산
    pos_weight = neg / pos
    """
    all_labels = []
    for _, y in loader:
        all_labels.extend(y.numpy().tolist())
    all_labels = np.array(all_labels)
    pos = (all_labels == 1).sum()
    neg = (all_labels == 0).sum()
    if pos == 0:
        return 1.0
    return neg / pos


def main():
    data_root = "/home/ssu/mediupxai/data"
    labels_csv = os.path.join(data_root, "all_labels_split.csv")
    batch_size = 128
    num_workers = 4
    num_epochs = 30
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # M0: clean으로만 학습
    # -------------------------
    print("\n====================")
    print("Training M0 (clean only)")
    print("====================")

    m0_train_dataset = ECGDataset(
        data_root=data_root,
        split="train",
        noise_types=["clean_segments"],
        labels_csv=labels_csv,
    )
    m0_val_dataset = ECGDataset(
        data_root=data_root,
        split="val",
        noise_types=["clean_segments"],
        labels_csv=labels_csv,
    )

    # 채널 수 추정 (첫 샘플)
    sample_x, _ = m0_train_dataset[0]
    in_channels = sample_x.shape[0]

    m0_train_loader = DataLoader(
        m0_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    m0_val_loader = DataLoader(
        m0_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    m0_pos_weight = compute_pos_weight_from_loader(m0_train_loader)
    print(f"[M0] pos_weight (neg/pos) = {m0_pos_weight:.3f}")

    m0_model = ECGCNN(in_channels=in_channels).to(device)
    m0_best_path = train_model(
        model=m0_model,
        train_loader=m0_train_loader,
        val_loader=m0_val_loader,
        device=device,
        num_epochs=num_epochs,
        lr=lr,
        pos_weight=m0_pos_weight,
        model_name="M0_clean",
        save_dir="./models",
    )

    # -------------------------
    # M1: noisy(bw + em + ma)로 학습
    # -------------------------
    print("\n====================")
    print("Training M1 (noisy: bw+em+ma)")
    print("====================")

    m1_train_dataset = ECGDataset(
        data_root=data_root,
        split="train",
        noise_types=["bw_noise", "em_noise", "ma_noise"],
        labels_csv=labels_csv,
    )
    m1_val_dataset = ECGDataset(
        data_root=data_root,
        split="val",
        noise_types=["bw_noise", "em_noise", "ma_noise"],
        labels_csv=labels_csv,
    )

    m1_train_loader = DataLoader(
        m1_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    m1_val_loader = DataLoader(
        m1_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    m1_pos_weight = compute_pos_weight_from_loader(m1_train_loader)
    print(f"[M1] pos_weight (neg/pos) = {m1_pos_weight:.3f}")

    m1_model = ECGCNN(in_channels=in_channels).to(device)
    m1_best_path = train_model(
        model=m1_model,
        train_loader=m1_train_loader,
        val_loader=m1_val_loader,
        device=device,
        num_epochs=num_epochs,
        lr=lr,
        pos_weight=m1_pos_weight,
        model_name="M1_noisy",
        save_dir="./models",
    )

    # -------------------------
    # 공통 noisy test로 성능 비교
    # -------------------------
    print("\n====================")
    print("Evaluate M0 / M1 on noisy test (bw+em+ma)")
    print("====================")

    noisy_test_dataset = ECGDataset(
        data_root=data_root,
        split="test",
        noise_types=["bw_noise", "em_noise", "ma_noise"],
        labels_csv=labels_csv,
    )
    noisy_test_loader = DataLoader(
        noisy_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # criterion (평가용, pos_weight 없이)
    criterion = nn.BCEWithLogitsLoss()

    # --- M0 ---
    m0_model.load_state_dict(torch.load(m0_best_path, map_location=device))
    test_loss, test_acc, test_f1, test_auc = evaluate(
        m0_model, noisy_test_loader, criterion, device
    )
    print(
        f"[M0 on noisy test] Loss={test_loss:.4f} ACC={test_acc:.3f} "
        f"F1={test_f1:.3f} AUC={test_auc:.3f}"
    )

    # --- M1 ---
    m1_model.load_state_dict(torch.load(m1_best_path, map_location=device))
    test_loss, test_acc, test_f1, test_auc = evaluate(
        m1_model, noisy_test_loader, criterion, device
    )
    print(
        f"[M1 on noisy test] Loss={test_loss:.4f} ACC={test_acc:.3f} "
        f"F1={test_f1:.3f} AUC={test_auc:.3f}"
    )


if __name__ == "__main__":
    main()
