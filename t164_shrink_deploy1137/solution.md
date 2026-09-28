# t164_shrink_deploy1137 — 역대 최고점 챔피언 모델 설명서

> **대회명**: LG Aimers 9기 Phase 2 — KBO 투구 제구 성공 확률 예측  
> **최고 Public Score**: **1137.4560145104** (대회 최고점 챔피언)  
> **최종 제출 파일**: [`submit.zip`](submit.zip) (약 58.7MB)  
> **추론 스크립트**: [`script.py`](script.py)  
> **학습 스크립트**: [`train_blend.py`](train_blend.py)  
> **모델 파일**: [`model/blend.pkl`](model/blend.pkl)  

---

## 1. 모델 핵심 요약

`t164_shrink_deploy1137`은 이전 신기록(`t157_nn_add`, 1128.3525) 대비 **+9.10점** 향상된 **1137.4560점**을 기록한 최종 최고점 챔피언 모델입니다.

### 핵심 개선 사항
1. **채널별 `SHRINK_K` 최적화**:
   - 기존의 단일 전역값(`SHRINK_K = 25.0`) 대신, 도메인 채널별 정보 밀도에 맞춘 최적 shrinkage 상수를 적용하여 노이즈 제거 및 사전 신호 보존 극대화.
2. **이종 앙상블 블렌드 구조**:
   - `no_asof + tier` 모델 (LightGBM): 가중치 0.485
   - `tm_full + tier` 모델 (LightGBM): 가중치 0.485
   - `nn` 모델 (임베딩 + 구종 보조과제 멀티태스크 신경망, 5시드, 순수 numpy 추론): 가중치 0.03
3. **정밀 사후 Affine 매핑**:
   - 리더보드 폐형해를 통해 검증된 $\alpha^*, \beta^*$ 보정 상수를 적용하여 확률 분포 스케일링 완전 정합.

---

## 2. 파일 목록 및 역할

- `submit.zip`: 대회 플랫폼 제출용 압축 파일 (내부에 `script.py`, `requirements.txt`, `model/blend.pkl` 포함)
- `script.py`: 단일행/배치 독립 추론 스크립트 (규칙 4 준수, 전수 검사 통과)
- `train_blend.py`: 전체 학습 및 블렌드 파이프라인 (재현성 100% 확보)
- `requirements.txt`: 추론 환경 종속성 목록
- `requirements-train.txt`: 학습 환경 종속성 목록
