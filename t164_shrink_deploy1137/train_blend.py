# train_blend.py — t152_t113_repro(재현) + loc_p_tier_hier(물리 티어 2단계 EB 수축) 추가.
#
# t151_rsm/run_tier.py 에서 CatBoost 챔피언에 이 피처 하나만 추가해 8시드 2회
# 재현: verdict.py 점수@꼭짓점 +0.59 / +1.17 (둘 다 양수, RF처럼 부호 안 갈림).
# 사전진단(암슬롯x구속 6티어)도 2019~2024 전 6년 부호일치(high_slow/mid_fast).
# 크기가 제출문턱(1.5%p)의 1/25라 홀드아웃만으론 결판 안 남 — 리더보드로 검증.
#
#   python train_blend.py            # <=2023 학습 -> 2024 홀드아웃 (검증만)
#   python train_blend.py --full     # <=2024 학습 -> model/blend.pkl (제출용)
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

CHAMP = "/home/kim/lg_ws/t87_seed100_1093"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CHAMP)
import train as T  # noqa: E402
import script  # noqa: E402  (SHRINK_K 를 모듈 속성으로 덮어써야 해서 모듈째로 임포트)
from script import CAT_COLS, build_features, logit  # noqa: E402

DATA = "./data"
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]
TM_TABLES = "/home/kim/lg_ws/test/report/t65_tm/tm_tables.joblib"
GROUPS = ["fastball", "breaking", "offspeed"]

SEEDS = tuple(range(42, 72))            # pkl 저장값: 30시드
# pkl 저장값 그대로 (blend.pkl 에서 직접 읽음, 2026-08-28)
N_ITER = {"no_asof": 335, "tm_full": 278}
BLEND_W = {"no_asof": 0.485, "tm_full": 0.485, "nn": 0.03}   # w_nn=0.03: t156_multitask 오라클 최고점
ALPHA_RATIO = 1.062102
# t162_shrink_channel 에서 실측 확정(corr +0.094%p, 점수@a* +1.71, 점수@고정a
# +1.03, 두 지표 부호 일치) — 전역 스칼라 25.0 대신 t122_kmap 폐형해 채널별 dict.
SHRINK_K_GLOBAL = {
    "loc_p_success": 37.0, "loc_p_middle": 114.0, "loc_p_ball": 149.0,
    "loc_p_strike": 264.0, "loc_p_reverse": 28.0, "loc_p_fast": 18.0,
    "loc_p_break": 12.0, "loc_p_offspeed": 9.0,
    "loc_b_success": 41.0, "loc_b_middle": 149.0,
}
K1, K2 = 15.0, 10.0                     # 티어 사전 가중치, 전역 사전 가중치 (t151_rsm/run_tier.py 와 동일)


def build_tier_prior(data_dir, upto):
    """암슬롯 x 구속 6티어 -> {pitcher_id: tier} 와 {tier: r_tier} (<=upto 데이터로만)."""
    pmap = joblib.load(TM_TABLES)["pmap"]
    inv_pmap = {v: k for k, v in pmap.items()}
    tm = pd.read_csv(os.path.join(data_dir, "trackman_history.csv"), encoding="utf-8-sig",
                     usecols=["pitcher_trackman_id", "season", "rel_side", "rel_height", "rel_speed"])
    tmU = tm[tm.season <= upto]
    prof = tmU.groupby("pitcher_trackman_id").agg(
        rel_side=("rel_side", "mean"), rel_height=("rel_height", "mean"), rel_speed=("rel_speed", "mean"))
    prof["arm_slot"] = prof["rel_height"] / prof["rel_side"].abs().clip(lower=0.05)
    prof["slot_bin"] = pd.qcut(prof["arm_slot"], 3, labels=["low", "mid", "high"])
    prof["velo_bin"] = pd.qcut(prof["rel_speed"], 2, labels=["slow", "fast"])
    prof["tier"] = prof["slot_bin"].astype(str) + "_" + prof["velo_bin"].astype(str)
    prof["pitcher_id"] = prof.index.map(inv_pmap)
    tier_map = prof.dropna(subset=["pitcher_id"]).copy()
    tier_map["pitcher_id"] = tier_map["pitcher_id"].astype(int)
    tier_map = tier_map.set_index("pitcher_id")["tier"]

    df = pd.read_csv(os.path.join(data_dir, "train.csv"), encoding="utf-8-sig",
                     usecols=["season", "pitcher_id", "control_success"])
    tr = df[df.season <= upto].copy()
    tr["tier"] = tr["pitcher_id"].map(tier_map)
    r_tier = tr.dropna(subset=["tier"]).groupby("tier")["control_success"].mean().to_dict()
    r0 = tr["control_success"].mean()
    return tier_map, r_tier, r0


def build_tm_fixed(data_dir, upto):
    """pkl 의 tm_full 피처(126개, std+share 포함)와 정확히 일치하도록 수정."""
    pmap = joblib.load(TM_TABLES)["pmap"]
    tm = pd.read_csv(os.path.join(data_dir, "trackman_history.csv"), encoding="utf-8-sig",
                     usecols=["pitcher_trackman_id", "season", "pitch_type_group"] + PHYS)
    lv = "pitcher_trackman_id"

    def build(src, cum):
        roll = (lambda x: x.groupby(level=lv).cumsum().groupby(level=lv).shift(1)
                ) if cum else (lambda x: x.groupby(level=lv).sum())
        tot = roll(src.groupby([lv, "season"]).size().rename("n"))
        out = [tot.rename("tm_n").to_frame()]
        for g in GROUPS:
            s = src[src.pitch_type_group == g]
            k = s.groupby([lv, "season"])
            cn = roll(k.size().rename("n"))
            c1 = roll(k[PHYS].sum())
            c2 = roll(k[PHYS].apply(lambda d: (d ** 2).sum()))
            mean = (c1.div(cn, axis=0)).add_prefix(f"tm_{g}_").add_suffix("_mean")
            var = (c2.div(cn, axis=0) - (c1.div(cn, axis=0)) ** 2).clip(lower=0)
            std = var.pow(0.5).add_prefix(f"tm_{g}_").add_suffix("_std")
            share = (cn / tot).rename(f"tm_{g}_share")
            out += [cn.rename(f"tm_{g}_n").to_frame(), mean, std, share.to_frame()]
        t = pd.concat(out, axis=1)
        return t[t["tm_n"].notna()]

    tr = build(tm, cum=True)
    fx = build(tm[tm.season <= upto], cum=False)
    fx.index.name = lv
    return tr, fx[tr.columns], pmap


def _cb(Xp, cats):
    return pd.DataFrame(Xp).astype({c: "int" for c in cats})


NN_SEEDS = tuple(range(1, 6))   # 5시드 앙상블 (t156_multitask 스크리닝은 1시드였음)
NN_LAMBDA = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def recover_pitch_labels(df):
    """t81_decomp/probe.py recover_labels() 와 동일 기법 — 구종군 보조라벨 복원."""
    order = np.lexsort((df["asof_pitcher_n"].to_numpy(), df["pitcher_id"].to_numpy()))
    pid = df["pitcher_id"].to_numpy()[order]
    n = df["asof_pitcher_n"].to_numpy("float64")[order]
    out = {}
    for name in ("fastball", "breaking", "offspeed"):
        c = np.rint(df[f"asof_pitcher_{name}_rate"].fillna(0.0).to_numpy("float64")[order] * n)
        lab = pd.Series(c).groupby(pid).shift(-1).to_numpy() - c
        back = np.empty_like(lab)
        back[order] = lab
        out[name] = back
    return out


class MultiTaskNet(nn.Module):
    def __init__(self, n_pitcher, n_batter, n_num, emb_dim=16, hidden=128):
        super().__init__()
        self.emb_p = nn.Embedding(n_pitcher + 1, emb_dim)   # +1: 미지 ID 폴백 슬롯
        self.emb_b = nn.Embedding(n_batter + 1, emb_dim)
        self.trunk = nn.Sequential(
            nn.Linear(2 * emb_dim + n_num, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.head_main = nn.Linear(hidden // 2, 1)
        self.head_aux = nn.Linear(hidden // 2, 3)

    def forward(self, pid, bid, num):
        h = torch.cat([self.emb_p(pid), self.emb_b(bid), num], dim=1)
        h = self.trunk(h)
        return self.head_main(h).squeeze(-1), self.head_aux(h)


def export_weights(model):
    """학습된 모델을 순수 numpy 배열로 뽑는다 — 추론에서 torch 불필요(요구사항 리스크 회피)."""
    sd = model.state_dict()
    return {k: v.cpu().numpy() for k, v in sd.items()
            if not k.startswith("head_aux")}   # 보조 헤드는 추론에 안 씀


def train_nn_ensemble(df, base, season, y_all, mask_tr, mask_es, mask_fit, seeds):
    aux = recover_pitch_labels(df)
    aux_stack = np.stack([aux["fastball"], aux["breaking"], aux["offspeed"]], axis=1)
    aux_valid = np.isfinite(aux_stack).all(axis=1) & (aux_stack.sum(axis=1) == 1)
    aux_class = np.where(aux_valid, aux_stack.argmax(axis=1), -1)

    pid_raw, bid_raw = df["pitcher_id"].to_numpy(), df["batter_id"].to_numpy()
    pid_map = {v: i for i, v in enumerate(np.unique(pid_raw[mask_fit]))}
    bid_map = {v: i for i, v in enumerate(np.unique(bid_raw[mask_fit]))}
    UNK_P, UNK_B = len(pid_map), len(bid_map)
    pid_idx = np.array([pid_map.get(v, UNK_P) for v in pid_raw], dtype=np.int64)
    bid_idx = np.array([bid_map.get(v, UNK_B) for v in bid_raw], dtype=np.int64)

    nn_num_cols = [c for c in base if c not in ("pitcher_id", "batter_id", "loc_p_tier_hier")
                   and df[c].dtype != object]
    Xnum = df[nn_num_cols].astype("float64").fillna(-999.0).to_numpy("float32")
    mu, sd_ = Xnum[mask_fit].mean(0), Xnum[mask_fit].std(0) + 1e-6
    Xnum = (Xnum - mu) / sd_

    def to_t(a, dtype=torch.float32):
        return torch.as_tensor(a, dtype=dtype, device=DEVICE)

    def batches(mask, bs=8192, shuffle=True, rng_seed=0):
        idx = np.where(mask)[0]
        if shuffle:
            np.random.default_rng(rng_seed).shuffle(idx)
        for i in range(0, len(idx), bs):
            yield idx[i:i + bs]

    bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()

    def run_epoch(model, opt, mask, train_mode, rng_seed):
        model.train(train_mode)
        tot, n = 0.0, 0
        for idx in batches(mask, shuffle=train_mode, rng_seed=rng_seed):
            pb, bb = to_t(pid_idx[idx], torch.long), to_t(bid_idx[idx], torch.long)
            nb, yb = to_t(Xnum[idx]), to_t(y_all[idx].astype("float32"))
            ab = to_t(aux_class[idx], torch.long)
            am = ab >= 0
            if train_mode:
                opt.zero_grad()
            mo, ao = model(pb, bb, nb)
            loss = bce(mo, yb)
            if am.any():
                loss = loss + NN_LAMBDA * ce(ao[am], ab[am])
            if train_mode:
                loss.backward()
                opt.step()
            tot += bce(mo, yb).item() * len(idx)
            n += len(idx)
        return tot / n

    # 조기종료 epoch 수는 시드 1개(42)로만 정하고, 나머지 시드는 그 수만큼 재사용
    # (시드마다 ES를 따로 돌리면 5배 느려짐 — 스크리닝에서 이미 7 epoch로 수렴 확인)
    probe = MultiTaskNet(len(pid_map), len(bid_map), len(nn_num_cols)).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-5)
    best_es, best_ep, patience, bad = np.inf, 0, 5, 0
    for epoch in range(60):
        run_epoch(probe, opt, mask_tr, True, epoch)
        es = run_epoch(probe, opt, mask_es, False, epoch)
        if es < best_es - 1e-5:
            best_es, best_ep, bad = es, epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    n_epochs = max(1, best_ep + 1)
    print(f"  [nn] ES epoch={n_epochs} (probe es_bce={best_es:.5f})", flush=True)

    exported = []
    for sd_seed in seeds:
        torch.manual_seed(sd_seed)
        m = MultiTaskNet(len(pid_map), len(bid_map), len(nn_num_cols)).to(DEVICE)
        o = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
        for epoch in range(n_epochs):
            run_epoch(m, o, mask_fit, True, sd_seed * 100 + epoch)
        exported.append(export_weights(m))
    print(f"  [nn] 시드 {len(seeds)}개 학습 완료", flush=True)

    return {
        "kind": "nn", "weights": exported, "pid_map": pid_map, "bid_map": bid_map,
        "unk_p": UNK_P, "unk_b": UNK_B, "num_cols": nn_num_cols,
        "mu": mu, "sd": sd_,
    }, (pid_idx, bid_idx, Xnum)


def nn_predict_numpy(member, pid_idx, bid_idx, Xnum):
    """torch 없이 순수 numpy 순전파 — 추론 경로와 완전히 같은 코드(script.py 에 복붙)."""
    def relu(x):
        return np.maximum(x, 0)

    preds = []
    for w in member["weights"]:
        ep = w["emb_p.weight"][pid_idx]
        eb = w["emb_b.weight"][bid_idx]
        h = np.concatenate([ep, eb, Xnum], axis=1)
        h = relu(h @ w["trunk.0.weight"].T + w["trunk.0.bias"])
        h = relu(h @ w["trunk.3.weight"].T + w["trunk.3.bias"])
        logit_main = h @ w["head_main.weight"].T[:, 0] + w["head_main.bias"][0]
        preds.append(1.0 / (1.0 + np.exp(-logit_main)))
    return np.mean(preds, axis=0)


def make_pre(num_cols):
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="constant", fill_value=-999.0), num_cols),
    ])


def main():
    full = "--full" in sys.argv
    t0 = time.time()
    test_cols = pd.read_csv(os.path.join(DATA, "test.csv"), encoding="utf-8-sig", nrows=0).columns
    raw = [c for c in test_cols if c != T.ID]
    df = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig", usecols=raw + [T.TARGET])
    season, y_all = df["season"].values, df[T.TARGET].values
    pid = df["pitcher_id"].to_numpy()
    df = build_features(df)
    # t113 은 채널별 SHRINK_K dict(script.py 기본값)가 아니라 전역 스칼라 25.0 을
    # 쓴다(pkl 실측). add_season_local(k=None) 이 script.SHRINK_K 를 그대로 읽으므로
    # 모듈 속성을 직접 덮어써서 add_local_train 내부까지 반영시킨다.
    script.SHRINK_K = SHRINK_K_GLOBAL
    df, r0_hist = T.add_local_train(df, season)

    # loc_p_tier_hier: 2단계 계층 EB = (c_i + K1*r_tier + K2*r0) / (n_i + K1 + K2)
    tier_upto = 2024 if full else 2023
    tier_map, r_tier, r0_train = build_tier_prior(DATA, tier_upto)
    n_i = df["asof_pitcher_n"].to_numpy("float64")
    c_i = np.rint(df["asof_pitcher_success_rate"].fillna(0.0).to_numpy("float64") * n_i)
    tier_of_row = pd.Series(pid).map(tier_map)
    r_tier_of_row = tier_of_row.map(r_tier).fillna(r0_train).to_numpy("float64")
    df["loc_p_tier_hier"] = (c_i + K1 * r_tier_of_row + K2 * r0_train) / (n_i + K1 + K2)
    print(f"[tier] 매핑 커버율 {tier_of_row.notna().mean():.1%}  r0={r0_train:.4f}  "
          f"티어={sorted(r_tier.items())}", flush=True)

    base = [c for c in df.columns if c != T.TARGET]
    rate = df.groupby("season")[T.TARGET].mean()

    tm_train, tm_fixed, pmap = build_tm_fixed(DATA, upto=2024 if full else 2023)
    tm_cols = list(tm_fixed.columns)
    joined = tm_train.reindex(pd.MultiIndex.from_arrays(
        [pd.Series(pid).map(pmap).to_numpy(), season]))
    for c in tm_cols:
        df[c] = joined[c].to_numpy()

    FEATS = {
        "no_asof": [c for c in base if not c.startswith("asof_")],
        "tm_full": base + tm_cols,
    }
    print(f"[load] {df.shape} | no_asof {len(FEATS['no_asof'])} | tm_full {len(FEATS['tm_full'])} "
          f"| {time.time()-t0:.1f}s", flush=True)

    fit_yr = 2024 if full else 2023
    mask_fit = season <= fit_yr
    cats_idx = list(range(len(CAT_COLS)))

    models, ps = {}, {}
    for nm, feats in FEATS.items():
        t1 = time.time()
        num_cols = [c for c in feats if c not in CAT_COLS]
        pre = make_pre(num_cols).fit(df.loc[mask_fit, feats])
        X = pre.transform(df.loc[mask_fit, feats])
        y = y_all[mask_fit]
        init = logit(rate.loc[season[mask_fit]].values)
        fkw = {"categorical_feature": cats_idx}
        clfs = [T.make_clf("lgbm", sd).set_params(n_estimators=N_ITER[nm])
                .fit(X, y, init_score=init, **fkw) for sd in SEEDS]
        models[nm] = {"pre": pre, "clfs": clfs, "features": feats}
        print(f"  [{nm:8s}] 피처 {len(feats):3d} | 시드 {len(SEEDS)} | {time.time()-t1:.0f}s", flush=True)

        if not full:
            Xho = pre.transform(df.loc[season == 2024, feats])
            z = logit(T.extrapolate_rate(rate, 2023, 2024))
            ps[nm] = np.mean([T.sigmoid(c.predict(Xho, raw_score=True) + z) for c in clfs], axis=0)

    t1 = time.time()
    nn_member, (pid_idx, bid_idx, Xnum) = train_nn_ensemble(
        df, base, season, y_all, season <= 2021, season == 2022, mask_fit, NN_SEEDS)
    models["nn"] = nn_member
    print(f"  [nn      ] 피처 {len(nn_member['num_cols']):3d} | 시드 {len(NN_SEEDS)} | "
          f"{time.time()-t1:.0f}s", flush=True)
    if not full:
        mask_ho2 = season == 2024
        ps["nn"] = nn_predict_numpy(nn_member, pid_idx[mask_ho2], bid_idx[mask_ho2], Xnum[mask_ho2])

    if not full:
        y_ho = y_all[season == 2024]
        r_hat = T.extrapolate_rate(rate, 2023, 2024)
        blend = sum(BLEND_W[k] * ps[k] for k in BLEND_W)
        for nm, p in ps.items():
            print(f"  {nm:8s} corr {np.corrcoef(p, y_ho)[0,1]:.6f}  점수 {1e5*T.bss(y_ho, p):8.2f}")
        astar = float(((blend - r_hat) * (y_ho - r_hat)).mean() / ((blend - r_hat) ** 2).mean())
        print(f"  블렌드 corr {np.corrcoef(blend, y_ho)[0,1]:.6f}  a*={astar:.6f}  "
              f"점수@a* {1e5*T.bss(y_ho, np.clip(r_hat + astar*(blend-r_hat), 1e-6, 1-1e-6)):8.2f}")
        from script import ALPHA, BETA
        a_eff = ALPHA * ALPHA_RATIO
        p_fin = np.clip(r_hat + a_eff * (blend - r_hat) + BETA, 1e-6, 1 - 1e-6)
        print(f"  [최종형, 배포 ALPHA*ALPHA_RATIO={a_eff:.6f}, BETA={BETA}] "
              f"점수 {1e5*T.bss(y_ho, p_fin):8.2f}  (참고: t113 실제 리더보드 1121.8486588715)")
        print(f"총 소요 {time.time()-t0:.0f}s")
        return

    # --full: 배포용 번들 저장.
    # snap/r0/r0_patch/offset 은 학습으로 재도출되는 게 아니라 리더보드 실측으로
    # 확정된 상수다(r0_patch 자체가 "leaderboard closed form" 이라고 pkl 안에
    # 명시돼 있다) — 재학습 대상이 아니므로 원본 pkl에서 그대로 가져온다.
    # 재현 대상은 "models"(학습되는 부분)뿐이고, 이 결정 자체가 Phase3 소명 포인트다.
    orig = joblib.load("/home/kim/lg_ws/test/report/t113_vertex/model/blend.pkl")
    bundle = dict(orig)
    bundle["models"] = models
    bundle["w"] = BLEND_W
    bundle["alpha_ratio"] = ALPHA_RATIO
    bundle["shrink_k"] = SHRINK_K_GLOBAL
    bundle["tm_fixed"] = tm_fixed
    bundle["pmap"] = pmap
    # loc_p_tier_hier 를 추론(script.py build_features(tier=...))에서 재현하려면
    # 티어 상수도 저장해야 한다. script.py 가 기대하는 형태: (tier_map, r_tier, r0, (K1,K2))
    bundle["tier"] = (tier_map, r_tier, r0_train, (K1, K2))
    os.makedirs("model", exist_ok=True)
    joblib.dump(bundle, "model/blend.pkl")
    print(f"저장: model/blend.pkl (snap/r0/r0_patch/offset 은 원본 pkl 상수 재사용) | 총 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
