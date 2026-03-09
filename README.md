# XAI 기반 웨어러블 ECG 노이즈 취약성 분석 및 강건 모델 제안

## 1. Project Topic
> 웨어러블 환경에서 발생하는 ECG 노이즈(BW, MA, EM)가 딥러닝 기반 부정맥 분류 모델에 미치는 영향을 분석하고, 
Explainable AI(XAI)를 활용하여 노이즈 취약성을 분석한 뒤 강건한 ECG 모델 설계 전략을 탐구

---

## 2. Project Workflow

```
1. ECG Dataset 구축
   - MIT-BIH Arrhythmia Database 기반 clean ECG 수집
   - ECG 신호를 10초 단위 segment로 분할

2. Noise 데이터 생성
   - MIT-BIH Noise Stress Test Database 활용
   - BW (Baseline Wander)
   - MA (Motion Artifact)
   - EM (Electrode Motion)
   - 다양한 SNR 수준으로 clean ECG에 노이즈 합성

3. Baseline 모델 구축 (M0)
   - clean ECG 데이터만 사용하여 학습
   - 기본 ECG 분류 성능 기준 모델 설정

4. Noise 학습 모델 구축 (M1)
   - noisy ECG 데이터를 포함하여 모델 학습
   - 웨어러블 환경에서의 robustness 향상

5. Explainable AI 분석
   - Grad-CAM
   - Integrated Gradients
   - SHAP
   - 노이즈 환경에서 모델의 attention 및 prediction 변화 분석

6. 노이즈 취약성 분석
   - 노이즈 유형별 모델 취약 메커니즘 분석
   - 모델의 feature attention과 prediction instability 평가

7. Robust 모델 설계 (M1')
   - XAI 분석 결과 기반 모델 개선
   - noise-aware augmentation 적용
   - 웨어러블 환경에 강건한 ECG 모델 설계
```

---

## 3. Dataset

- **MIT-BIH Arrhythmia Database**  
  부정맥 ECG 신호 데이터셋

- **MIT-BIH Noise Stress Test Database**  
  ECG 노이즈 데이터셋

-> 해당 데이터셋을 기반으로 clean ECG 신호에 다양한 노이즈를 합성하여 학습 데이터를 구성했습니다.

---

## 4. Key Techniques

본 프로젝트에서는 다음 기술들을 활용했습니다.

- Deep Learning 기반 ECG classification
- Explainable AI (Grad-CAM, Integrated Gradients, SHAP)
- Noise-aware data generation
- Robust model design for wearable ECG
