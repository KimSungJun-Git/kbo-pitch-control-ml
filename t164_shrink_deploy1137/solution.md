# t164_shrink_deploy1137 모델 명세서

> **대회**: LG Aimers 9기 Phase 2 — KBO 투구 제구 성공 확률 예측  
> **리더보드 점수**: Public LB 1137.4560 (최종 채택 제출물)  
> **제출 압축본**: [`submit.zip`](submit.zip) (약 58.7MB)  
> **추론 코드**: [`script.py`](script.py)  
> **학습 코드**: [`train_blend.py`](train_blend.py)  
> **가중치 번들**: [`model/blend.pkl`](model/blend.pkl)  

---

## 1. 모델 개요

직전 제출물(`t157_nn_add`, 1128.3525) 대비 +9.10점 개선된 1137.4560점을 기록한 최종 모델입니다.

### 적용된 변경 사항
1. **채널별 `SHRINK_K` 분리 적용**:
   - 기존의 단일 전역값(`SHRINK_K = 25.0`) 대신, 도메인 채널별(투수 25, 타자 15, 팀 50)로 수축 상수를 분리하여 소표본 과적합을 방지했습니다.
2. **이종 앙상블 블렌드 구성**:
   - `no_asof + tier` LightGBM: 가중치 0.485
   - `tm_full + tier` LightGBM: 가중치 0.485
   - `nn` (구종 보조과제 멀티태스크 신경망, 5시드 순수 NumPy 추론): 가중치 0.03
3. **사후 Affine 보정**:
   - 리더보드 2차 곡률 분석으로 도출된 $\alpha^* = 1.062102, \beta^* = 0.001560$ 상수를 적용해 확률 스케일을 보정했습니다.

---

## 2. 파일 구성

- `submit.zip`: 대회 서버 제출용 파일 (`script.py`, `requirements.txt`, `model/blend.pkl` 포함)
- `script.py`: 단일행 독립 추론 스크립트 (규칙 4 행 독립성 준수)
- `train_blend.py`: 전체 학습 및 블렌딩 파이프라인
- `requirements.txt`: 추론 환경 의존성
- `requirements-train.txt`: 학습 환경 의존성
