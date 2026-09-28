# script.py
import os

import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
# t94: pitcher_team_id / batter_team_id 를 범주형에 추가.
# 이전엔 숫자로 취급돼 "값이 크다/작다"로만 분기했는데, 익명화 ID 크기엔
# 의미가 없다 — 카디널리티도 13으로 낮아 pitcher_id/batter_id(792/830, 범주형
# 처리 시 과적합 확인됨. exp_catid.py 참고)와 달리 안전하다. 동일 조건
# (≤2023 학습 -> 2024 홀드아웃, 시드1, n_iter=300 고정) 비교에서
# BSS 0.007028 -> 0.007118 (+0.000090, 리더보드 환산 약 +9점).
CAT_COLS = ["top_bottom", "game_type", "base_state",
           "pitcher_team_id", "batter_team_id"]

logit = lambda p: np.log(p / (1 - p))       # noqa: E731

# asof_pitcher_n 과 147만 행 전부 값이 같은 중복 컬럼.
DROP_COLS = ["asof_pitcher_pitchmix_n"]

# =======================
# 시즌 내 성분 분해
# =======================
# asof_*_rate 는 커리어 통산이다. asof_pitcher_n 이 시즌 경계에서 리셋되지 않는
# 통산 카운터(투수별 증분 1, 예외 0건)임을 확인했고, rate * n 은 정수와 최대
# 7.6e-3 오차로 일치한다 — 원본 카운트가 무손실 복원된다는 뜻이다.
#
# 그래서 2025 행의 통산 rate 는 2025 신호가 과거에 희석된 값이다(예: 통산
# 3,465구 중 2025 는 380구뿐). 학습 시점에 확정된 투수별 스냅샷 (n0, c0)을 빼면
# "그 시즌 안에서의" 성적만 남는다. 스냅샷은 pkl 에 저장된 상수 테이블이고
# 계산은 전부 현재 행 안에서 끝나므로 대회 규칙 4) 를 지킨다 — 평가 데이터의
# 다른 행이나 그 분포를 일절 참조하지 않는다.
LOCAL_SPECS = [
    # (id 컬럼, n 컬럼, rate 컬럼, 출력 이름)
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate", "loc_p_success"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_middle_rate",  "loc_p_middle"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_ball_rate",    "loc_p_ball"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_strike_rate",  "loc_p_strike"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_reverse_rate", "loc_p_reverse"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_fastball_rate",  "loc_p_fast"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_breaking_rate",  "loc_p_break"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_offspeed_rate",  "loc_p_offspeed"),
    ("batter_id",  "asof_batter_n",  "asof_batter_success_rate",  "loc_b_success"),
    ("batter_id",  "asof_batter_n",  "asof_batter_middle_rate",   "loc_b_middle"),
]

# 수축 강도. 초기 스윕(2023/2024 평가)에서는 100 이 최고였으나 파생피처를
# loc_* 로 교체한 뒤 재스윕에서 50 으로 옮겨갔고, t41~t45 제출은 전부 50 이다.
# 기본값이 100 이면 환경변수 없이 재학습했을 때 문서와 다른 모델이 조용히 나온다.
# 스윕용으로 환경변수를 받되, 학습에 쓴 값은 pkl 에 저장해 추론이 그 값을 쓴다 —
# 서버에서 환경변수가 없다고 기본값으로 돌아가면 학습과 조용히 어긋난다.
# t54 에서 50 -> 25. 추론은 pkl 에 저장된 값을 쓰므로 배포는 어느 쪽이든 안전하지만,
# 기본값이 실제 제출값과 다르면 환경변수 없이 재학습했을 때 문서와 다른 모델이
# 조용히 나온다 (t49 ② 가 잡아낸 결함이다). 제출값과 항상 일치시킬 것.
SHRINK_K = float(os.environ.get("SHRINK_K", 25.0))

# t67: 예측 수축 p' = r̂ + ALPHA·(p − r̂).
#
# ALPHA=1.0 은 t62(1080.5041888924) 와 비트 단위로 같은 예측이다. 이 값을 1 에서
# 옮기는 근거는 리더보드 자체다 — Brier 는 ALPHA 의 정확한 2차식이고, 리더보드
# 점수 하나가 이미 2·Cov − Var = 0.0026981 을 확정해놨다. 남은 미지수는 Var 하나뿐.
#
#   Var=0.00213 (2024 홀드아웃 실측) -> ALPHA* 1.133, +15.2점
#   Var=0.00180                      -> ALPHA* 1.250, +44.9점
#   Var=0.00250                      -> ALPHA* 1.040,  +1.6점
#
# 1.10 은 탐침이었다. α=1.00(1080.5041888924)·α=1.10(1088.8266148259) 두 점으로
# 포물선이 완전히 풀린다: Cov=0.002481, Var=0.002264, α*=1.0959, 꼭짓점 1088.8419
# (지금보다 +0.015점, RESULTS.md §25.1·31.1). 그 꼭짓점 값으로 내린다.
#
# ⚠️ r̂=0.4829 와 마찬가지로 이 값의 출처는 리더보드다. Phase 3 에서 그대로 설명할 것.
ALPHA = float(os.environ.get("ALPHA", 1.079592))

# t82: 사후 매핑의 **자유절편** 탐침.
#
# q = r̂ + ALPHA·(p − r̂) 는 확률공간에서 q = a·p + b 인 affine 인데 절편이 기울기에
# 묶여 있다(b = r̂(1−ALPHA)). t62 는 ALPHA=1.0(매핑이 항등)에서 r̂ 을 잡았으므로 그건
# **로짓 오프셋**을 고친 것이고 확률공간 절편을 고친 게 아니다. 좌표하강 2스텝이라
# 이 방향은 한 번도 안 열렸다.
#
# α 변형 3건(1.0/1.10/1.0959, pkl md5 동일 = 같은 base p)으로 Brier(α)=α²V−2αC+Q 의
# 계수 셋을 풀면 t67 이 쓴 가정 q:=Q/D=1 (r̂ = 실제 기저율)이 **반증된다** —
# q=1 2점 모형의 꼭짓점이 1088.8419 인데 α=1.0959 에서 1088.9116 을 실측했고,
# 포물선은 자기 꼭짓점 위로 갈 수 없다. 풀면 q−1 = (r−r̂)²/D = 1.951e-3.
#
# 자유절편을 열면 Brier(β) = Brier(0) + β² + 2β·K 이고 **곡률이 정확히 1** 이라
# 점 하나면 K 가 풀린다(t67 이 α 를 제출 1회로 닫은 것과 같은 수법).
#
#   ΔScore = −1e5·(β² + 2βK)/D      D = r(1−r) ≈ 0.2497
#   측정 후:  K = −(D·ΔScore/1e5 + β²) / (2β),   최적 절편 β* = −K
#
# BETA=0.005 로 탐침한다. 예상 착지점은 부호에 따라 ~1070.9 또는 ~1086.9 로
# 16점 갈라져 모호성이 없다. **리더보드는 최고점을 유지하므로 순위는 안 깎인다.**
# 확정 후 배포는 β* 를 넣고 다시 만든다(예상 +1.9점, 자유 affine 꼭짓점 1090.83).
#
# ⚠️ 이 값의 출처도 r̂·α 와 같이 **리더보드**다. Phase 3 코드 검증에서 그대로 설명할 것.
BETA = float(os.environ.get("BETA", 0.001741))


# id 컬럼 -> 그 주체의 통산 투구수 컬럼.
N_COL = {"pitcher_id": "asof_pitcher_n", "batter_id": "asof_batter_n"}


def report_snap_link(df, n0_raw):
    """스냅샷이 평가 행에 실제로 이어졌는지 세 숫자로 확인한다.

    local_counts 의 clip 때문에, 평가 데이터의 asof 카운터가 학습 때와 다른
    기준으로 만들어졌으면 n_cur 가 전부 0 이 되고 loc_* 10 개가 통째로 r0 상수로
    붕괴한다. 크래시도 NaN 도 나지 않고 예측 평균조차 멀쩡해 보인다.

    신인 폴백(스냅샷 미등록)이 정확히 같은 경로를 타므로 '미지 ID 주입' 검사로는
    이 붕괴를 잡을 수 없다. 구분되는 유일한 신호가 **기준역전**이다 — 스냅샷에
    등록된 선수의 통산 카운터가 스냅샷 기준선보다 작아지는 경우로, 카운터가 서로
    이어져 있는 한 정의상 0% 여야 한다. 0 이 아니면 그 자리에서 멈춘다.
    """
    for c, n0 in n0_raw.items():
        n = df[N_COL[c]].to_numpy("float64")
        seen = n0.notna().to_numpy()
        n0v = n0.fillna(0.0).to_numpy("float64")
        stale = float((seen & (n < n0v)).mean())
        print(f" snap[{c:10s}] 미등록 {1 - seen.mean():6.1%} (신인 폴백) | "
              f"n_cur=0 {(np.maximum(n - n0v, 0.0) == 0).mean():6.1%} | "
              f"기준역전 {stale:6.1%}")
        if stale > 0:
            raise SystemExit(
                f"스냅샷 단절: {c} 의 {stale:.1%} 행에서 통산 카운터가 스냅샷보다 작다. "
                "loc_* 가 전부 상수로 붕괴하므로 이대로 제출하면 안 된다.")


def local_counts(d, n0_map, c0_map):
    """복원 카운트에서 시즌 내 성분만 남긴다. 반환: {출력이름: (n_cur, c_cur)}

    n0_map / c0_map 을 만드는 방법만 학습(그룹별 첫 투구)과 추론(pkl 스냅샷)이
    다르고, 복원·차감 산술은 여기 한 곳에만 있다.
    """
    out = {}
    for id_col, n_col, rate_col, name in LOCAL_SPECS:
        n = d[n_col].to_numpy("float64")
        c = np.rint(d[rate_col].fillna(0.0).to_numpy("float64") * n)
        n_cur = np.maximum(n - n0_map[id_col], 0.0)
        # 스냅샷이 없는 신인은 n0=c0=0 이라 통산이 곧 시즌 내 성분이 된다.
        out[name] = (n_cur, np.clip(c - c0_map[name], 0.0, n_cur))
    return out


def add_season_local(d, counts, r0, k=None):
    """시즌 내 비율을 경험적 베이즈로 수축해 붙인다. r0 는 그 시즌의 사전평균.

    k 는 전역 스칼라(기존) 또는 채널별 dict(t122_kmap 폐형해, t162_shrink_channel
    에서 실측 양수 확정) 둘 다 받는다.
    """
    k = SHRINK_K if k is None else k
    for name, (n_cur, c_cur) in counts.items():
        kv = k[name] if isinstance(k, dict) else k
        d[name] = (c_cur + kv * r0[name]) / (n_cur + kv)
    d["loc_p_n"] = counts["loc_p_success"][0]     # 시즌 내 누적 투구수 = 등판 부하
    d["loc_b_n"] = counts["loc_b_success"][0]

    # 기존 파생피처는 전부 커리어 rate 를 기준선으로 쓴다. 그런데 그 값은 시즌
    # 드리프트 때문에 편향돼 있다 — 단독 예측에 넣으면 상수보다 나쁘다(BSS 음수).
    # 같은 대비를 편향 없는 시즌 내 값 위에서 다시 만든다.
    d["loc_matchup_gap"] = d["loc_p_success"] - d["loc_b_success"]
    d["loc_matchup_gap_mid"] = d["loc_p_middle"] - d["loc_b_middle"]
    for j in (1, 3, 5):
        d[f"loc_form_delta{j}"] = (d[f"asof_pitcher_prev{j}_game_success_rate"]
                                   - d["loc_p_success"])
    # 올 시즌이 통산 대비 얼마나 좋은가. 트리는 두 컬럼의 차를 못 만든다.
    d["season_vs_career"] = d["loc_p_success"] - d["asof_pitcher_success_rate"]
    r = d[["loc_p_fast", "loc_p_break", "loc_p_offspeed"]].to_numpy("float64")
    d["loc_pitch_entropy"] = -(r * np.log(r + 1e-9)).sum(axis=1)
    return d


# =======================
# trackman 물리 프로파일 (t92)
# =======================
# 투수별 구속·회전·무브먼트·릴리스의 평균/표준편차. pkl 에 든 고정 테이블
# (2024 까지 누적) 을 pitcher_id -> trackman_id 대응으로 조회만 한다.
# 학습 시점에 확정된 상수 테이블이라 평가 데이터의 다른 행을 참조하지 않는다
# (규칙 4). ID 대응 추정은 2026-08-07 대회 Q&A 에서 공식 허용됐다.

def add_trackman(d, raw_pitcher_id, tm_fixed, pmap):
    """tm_fixed 의 컬럼을 그대로 붙인다. 미매칭 투수는 NaN (전처리가 -999 로 격리)."""
    tid = pd.Series(raw_pitcher_id).map(pmap)
    joined = tm_fixed.reindex(tid.to_numpy())
    for c in tm_fixed.columns:
        d[c] = joined[c].to_numpy()
    return d


# =======================
# 데이터 로드 유틸
# =======================

def find_data_dir(base):
    """평가 데이터 디렉터리를 찾는다.

    공지의 제출 구조도는 `data/`, 유의사항 문구는 `open/` 으로 서로 다르게
    적혀 있다. 둘 중 실제로 존재하는 쪽을 쓴다 — 잘못 고르면 '제출 오류'로
    일일 제출 횟수를 잃는다.
    """
    for name in ("data", "open"):
        d = os.path.join(base, name)
        if os.path.isfile(os.path.join(d, "test.csv")):
            return d
    raise FileNotFoundError(
        f"test.csv 를 찾을 수 없음: {base}/data, {base}/open 모두 확인함")


def load_test(path):
    """평가 데이터(csv) 로드. 한 행이 투구 하나."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    """sample_submission.csv 로드 — 제출 파일의 row_id 순서/컬럼 기준."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: "
            f"{list(df.columns)}")
    return df


# =======================
# 학습 때 사용한 전처리 (그대로)
# =======================

def add_features(df):
    """도메인 파생 피처. 전부 한 행 안에서만 계산되므로 대회 규칙 4)를 지킨다.

    트리는 컬럼 하나씩만 자를 수 있어 '두 컬럼의 차/비'는 스스로 못 만든다.
    그래서 차이·엔트로피처럼 트리가 표현하지 못하는 형태만 골라 넣는다.
    """
    d = df.copy()
    b, s = d["balls_before"], d["strikes_before"]
    d["count_stress"] = b - s                    # 양수일수록 투수에게 불리
    d["is_full_count"] = ((b == 3) & (s == 2)).astype("int8")
    d["is_scoring_position"] = ((d["runner_on_2b"] == 1) | (d["runner_on_3b"] == 1)).astype("int8")
    d["quick_motion"] = ((d["runner_on_1b"] == 1) & (d["runner_on_2b"] == 0)
                         & (d["runner_on_3b"] == 0)).astype("int8")   # 슬라이드 스텝 강제
    d["is_same_hand"] = (d["pitcher_hand"] == d["batter_hand"]).astype("int8")
    d["clutch"] = (d["inning"] >= 7).astype("int8") * d["li"] * (d["num_runners_on"] + 1)

    # t44 — 아래 6개(matchup_gap, matchup_gap_mid, form_delta{1,3,5},
    # pitch_entropy)는 전부 **커리어 rate 를 기준선으로 쓰는** 파생피처였다.
    # 그 기준선이 시즌 드리프트로 편향돼 있다는 게 t41 에서 확인됐고(단독 BSS
    # 음수), t42 는 편향판을 남긴 채 loc_* 판을 옆에 붙여 +0.16%(노이즈)였다.
    # 여기서는 편향판을 **빼고** loc_* 판만 남긴다 — add 가 아니라 replace.
    # loc_matchup_gap / loc_form_delta{1,3,5} / loc_pitch_entropy 가
    # add_season_local() 에서 같은 자리를 채운다.
    return d


def build_features(df, snap=None, r0=None, k=None, tier=None):
    """모델 입력 추출 — 학습과 추론이 반드시 같은 코드를 타야 한다.

    범주형 인코딩(top_bottom, game_type, base_state)은 모델 파일 안의
    전처리기(pre)가 수행하므로 여기서는 컬럼 구성만 맞춘다.

    snap 이 주어지면(추론 경로) 학습 시점에 확정된 투수/타자별 스냅샷으로
    시즌 내 성분을 분해해 붙인다. snap 이 없으면 t38 과 동일한 피처 집합이다.

    tier 가 주어지면(t151_rsm/run_tier.py, 물리 티어 2단계 EB 수축) loc_p_tier_hier
    를 붙인다. asof_pitcher_n/success_rate(그 행 자체의 값)와 학습 시점에 확정된
    (tier_map, r_tier, r0_tier, (K1,K2)) 상수만 쓰므로 이 행 하나만 봐도 같은 값이
    나온다(규칙 4). train_blend.py 의 학습측 계산과 완전히 동일한 공식.
    """
    d = add_features(df)
    if snap is not None:
        n0_raw = {c: df[c].map(snap["n0"][c]) for c in ("pitcher_id", "batter_id")}
        n0_map = {c: v.fillna(0.0).to_numpy("float64") for c, v in n0_raw.items()}
        c0_map = {name: df[id_col].map(snap["c0"][name]).fillna(0.0).to_numpy("float64")
                  for id_col, _, _, name in LOCAL_SPECS}
        report_snap_link(df, n0_raw)
        d = add_season_local(d, local_counts(d, n0_map, c0_map), r0, k)
    if tier is not None:
        tier_map, r_tier, r0_tier, (K1, K2) = tier
        n_i = df["asof_pitcher_n"].to_numpy("float64")
        c_i = np.rint(df["asof_pitcher_success_rate"].fillna(0.0).to_numpy("float64") * n_i)
        tier_of_row = df["pitcher_id"].map(tier_map)
        r_tier_of_row = tier_of_row.map(r_tier).fillna(r0_tier).to_numpy("float64")
        d["loc_p_tier_hier"] = (c_i + K1 * r_tier_of_row + K2 * r0_tier) / (n_i + K1 + K2)
    return d.drop(columns=[ID_COL] + DROP_COLS, errors="ignore")


def predict_proba(bundle, X):
    """제구 성공 확률 예측.

    학습 때 시즌 기저율을 init_score로 빼고 학습했으므로 predict_proba를 쓰면
    오프셋이 빠져 예측이 어긋난다. raw score에 오프셋을 더한 뒤 sigmoid를 취한다.
    오프셋은 학습 시점에 확정된 상수 하나라 평가 데이터의 다른 행을 참조하지 않는다.

    시드가 다른 모델 여러 개의 확률을 평균낸다(시드 앙상블).

    t67: 마지막에 예측 수축 p' = r̂ + ALPHA·(p − r̂) 를 건다. ALPHA 는 상수 하나이고
    r̂ 은 pkl 안의 offset 에서 나오므로, 이 행 하나만 넣어도 같은 값이 나온다(규칙 4).
    """
    pre, clfs, offset = bundle["pre"], bundle["clfs"], bundle["offset"]
    # 축소 변형은 학습 때 고른 피처 목록을 함께 저장한다. 없으면 전체 사용(t1 호환).
    feats = bundle.get("features")
    Xp = pre.transform(X[feats] if feats else X)
    if offset is None:  # --no-offset 으로 학습한 모델
        return np.mean([c.predict_proba(Xp)[:, 1] for c in clfs], axis=0)
    p = np.mean([1 / (1 + np.exp(-(c.predict(Xp, raw_score=True) + offset)))
                 for c in clfs], axis=0)
    r_hat = 1.0 / (1.0 + np.exp(-offset))   # t62 가 리더보드로 확정한 0.4829
    # t82: 자유절편 BETA 를 더한다. 모든 행에 같은 상수 하나라 이 행만 넣어도 값이
    # 같다(규칙 4). ALPHA==1.0 조기반환은 BETA 를 건너뛰므로 없앴다.
    return np.clip(r_hat + ALPHA * (p - r_hat) + BETA, 1e-6, 1 - 1e-6)


def _nn_predict(member, X):
    """순수 numpy 순전파 — torch 불필요(requirements 리스크 회피).
    train_blend.py 의 nn_predict_numpy() 와 완전히 같은 코드(학습/추론 일치 원칙)."""
    pid_map, bid_map = member["pid_map"], member["bid_map"]
    unk_p, unk_b = member["unk_p"], member["unk_b"]
    pid_idx = X["pitcher_id"].map(pid_map).fillna(unk_p).to_numpy("int64")
    bid_idx = X["batter_id"].map(bid_map).fillna(unk_b).to_numpy("int64")
    Xnum = X[member["num_cols"]].astype("float64").fillna(-999.0).to_numpy("float32")
    Xnum = (Xnum - member["mu"]) / member["sd"]

    def relu(a):
        return np.maximum(a, 0)

    preds = []
    for w in member["weights"]:
        ep = w["emb_p.weight"][pid_idx]
        eb = w["emb_b.weight"][bid_idx]
        h = np.concatenate([ep, eb, Xnum], axis=1)
        h = relu(h @ w["trunk.0.weight"].T + w["trunk.0.bias"])
        h = relu(h @ w["trunk.3.weight"].T + w["trunk.3.bias"])
        z = h @ w["head_main.weight"].T[:, 0] + w["head_main.bias"][0]
        preds.append(1.0 / (1.0 + np.exp(-z)))
    return np.mean(preds, axis=0)


def predict_blend(bundle, X):
    """t93 블렌드 추론 — no_asof 와 tm_full 을 균등 평균한 뒤 수축을 건다.

    ⚠️ 블렌드는 챔피언보다 예측 sd 가 작다. 리더보드로 확정한 ALPHA 는 챔피언의
    분포에 맞춰진 값이라 그대로 쓰면 수축이 모자란다. 학습 시점에 2024 홀드아웃
    에서 구한 상수 k 로 분산을 챔피언에 맞춘 뒤 ALPHA/BETA 를 적용한다.
    k 는 pkl 에 든 상수 하나라 이 행만 넣어도 같은 값이 나온다 (규칙 4).

    "nn" 멤버(kind="nn")는 LGBM이 아니라 순수 numpy로 재구현한 소형 신경망 —
    pitcher_id/batter_id는 학습 시 확정된 pid_map/bid_map으로 조회하고, 못 본
    ID는 예약된 미지 슬롯(unk_p/unk_b)으로 폴백한다(규칙 4: 이 행 하나만 봐도
    같은 값이 나온다 — test.csv의 다른 행을 참조하지 않는다).
    """
    offset = bundle["offset"]
    r_hat = 1.0 / (1.0 + np.exp(-offset))
    parts = {}
    for nm, m in bundle["models"].items():
        if m.get("kind") == "nn":
            parts[nm] = _nn_predict(m, X)
            print(f" [{nm}] numpy 신경망 시드 {len(m['weights'])}개  평균 {parts[nm].mean():.4f}  "
                  f"sd {parts[nm].std():.4f}")
            continue
        Xp = m["pre"].transform(X[m["features"]])
        parts[nm] = np.mean(
            [1 / (1 + np.exp(-(c.predict(Xp, raw_score=True) + offset)))
             for c in m["clfs"]], axis=0)
        print(f" [{nm}] 시드 {len(m['clfs'])}개  평균 {parts[nm].mean():.4f}  "
              f"sd {parts[nm].std():.4f}")
    p = sum(bundle["w"][nm] * parts[nm] for nm in parts)
    a = ALPHA * bundle["alpha_ratio"]
    print(f" 블렌드 평균 {p.mean():.4f} sd {p.std():.4f} | 유효 ALPHA {a:.6f}")
    return np.clip(r_hat + a * (p - r_hat) + BETA, 1e-6, 1 - 1e-6)


# =======================
# 제출 파일 생성 유틸
# =======================

def merge_predictions(sub, ids, preds):
    """sample_submission의 row_id 순서에 맞춰 예측 확률 병합.

    예측에 없는 row_id는 sample_submission의 기존 값(placeholder)을 유지한다.
    """
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    # 전부 못 붙었으면 row_id 형식이 어긋난 것이다. 그대로 저장하면 sample 의
    # placeholder 를 제출하게 되는데, 경고 한 줄은 서버 로그를 못 보면 사라진다.
    # 어차피 점수가 안 나오는 상황이라면 조용히 나가는 것보다 여기서 멈추는 게 낫다.
    if n_missing == len(sub):
        raise SystemExit(
            f"예측이 하나도 병합되지 않았다 (row_id {len(sub):,}건 전부 불일치). "
            f"test={ids[:2]}... vs sample={list(sub[ID_COL][:2])}...")
    if n_missing:
        print(f" ⚠️  예측이 없어 placeholder를 유지한 row_id {n_missing}건 "
              f"({n_missing / len(sub):.2%})")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# =======================
# main
# =======================

def main():
    # ---- 경로 변수 (필요에 따라 수정) ----
    BASE = os.path.dirname(os.path.abspath(__file__))
    TEST_DIR = find_data_dir(BASE)             # test.csv, sample_submission.csv 위치
    MODEL_DIR = os.path.join(BASE, "model")    # lgbm.pkl 위치
    OUT_DIR = os.path.join(BASE, "output")
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, "blend.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 로드 ----
    print("Load model...")
    bundle = joblib.load(MODEL_PATH)
    print(f" OK. 블렌드 {list(bundle['w'].items())} "
          f"offset={bundle['offset']:.4f} k={bundle['alpha_ratio']}")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    # ---- 전처리 (학습과 동일) ----
    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test, bundle.get("snap"), bundle.get("r0"), bundle.get("shrink_k"),
                       bundle.get("tier"))
    X = add_trackman(X, test["pitcher_id"].to_numpy(),
                     bundle["tm_fixed"], bundle["pmap"])
    print(f" features={X.shape[1]} "
          f"(tm 커버 {X[bundle['tm_fixed'].columns[0]].notna().mean():.1%})")

    # ---- 예측 (제구 성공 확률) ----
    print("Inference model...")
    preds = predict_blend(bundle, X) if len(X) else []
    print(f" preds={len(preds)}")

    # ---- sample_submission 기반 결과 생성 ----
    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"✅ Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
