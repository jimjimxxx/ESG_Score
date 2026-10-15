"""模擬研究：在「已知真實 ESG 水準」的假資料上，比較三種加權方法能否還原真相。

資料生成（每次重複）：
  - 1000 家公司、20 個產業、18 個指標（E 6 / S 5 / G 7，與真實資料相同）
  - 每家公司有三個潛在面向品質 q_E, q_S, q_G（彼此相關 ρ = 0.3），
    真實整體品質 Q = 三個面向的平均 → 這是要還原的目標
  - 指標 x_j = λ_j · g_j(q_面向) + 產業效果 + 雜訊
      λ_j       ：資訊量，0.3–1.0；「雜訊指標」的 λ = 0
      g_j       ：線性，或非線性（門檻二元化、飽和）
      產業效果   ：每個產業每個指標一個平移，模擬產業間的系統性差異
      雜訊       ：常態或 t(3) 厚尾
  - 隨機缺值（MCAR），另有一個指標缺 50%（對應真實資料的再生能源使用率）
  - 用與真實資料「完全相同」的產業內穩健標準化

評估指標：
  spearman   ：分數與真實 Q 的等級相關
  tier_acc   ：產業內分成三級（優 / 中 / 差）與真實三級一致的比例
  noise_wt   ：分數放在雜訊指標上的權重比例（越低越好）
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

from build_dataset import INDICATORS, standardize_within
from model import PILLARS, equal_weight_score, fit_pwae, pca_score

ROOT = Path(__file__).resolve().parent.parent
PILLAR_OF = [p for _, p, _ in INDICATORS]
P = len(PILLAR_OF)

SCENARIOS = {
    "1 Linear, Gaussian": dict(noise_frac=0.0, nonlinear=False, heavy=False, miss=0.10),
    "2 Noise indicators": dict(noise_frac=0.4, nonlinear=False, heavy=False, miss=0.10),
    "3 Nonlinear, heavy-tailed": dict(noise_frac=0.0, nonlinear=True, heavy=True, miss=0.10),
    "4 Realistic mix": dict(noise_frac=0.3, nonlinear=True, heavy=True, miss=0.30),
}


def generate(rng, n=1000, n_ind=20, noise_frac=0.0, nonlinear=False, heavy=False, miss=0.1, rho=0.3):
    cov = np.full((3, 3), rho) + (1 - rho) * np.eye(3)
    q = rng.multivariate_normal(np.zeros(3), cov, size=n)
    Q = q.mean(1)
    industry = rng.integers(0, n_ind, size=n)

    lam = rng.uniform(0.3, 1.0, size=P)
    is_noise = np.zeros(P, bool)
    for p in PILLARS:  # 每個面向內按比例指定雜訊指標，確保每面向至少留 1 個有效指標
        ix = [j for j in range(P) if PILLAR_OF[j] == p]
        k = min(int(round(noise_frac * len(ix))), len(ix) - 1)
        is_noise[rng.choice(ix, size=k, replace=False)] = True
    lam[is_noise] = 0.0

    X = np.empty((n, P))
    for j in range(P):
        qp = q[:, PILLARS.index(PILLAR_OF[j])]
        signal = qp
        if nonlinear:
            kind = j % 3
            if kind == 1:  # 門檻：只有「有 / 沒有」（如是否取得驗證）
                signal = (qp > rng.normal(0, 0.5)).astype(float) * 2 - 1
            elif kind == 2:  # 飽和：好到一定程度後指標不再上升（如出席率 100%）
                signal = np.tanh(1.5 * qp) * 1.5
        eps = rng.standard_t(3, size=n) / np.sqrt(3) if heavy else rng.normal(size=n)
        shift = rng.normal(0, 1.0, size=n_ind)[industry]
        X[:, j] = lam[j] * signal + 0.7 * eps + shift

    X[rng.random((n, P)) < miss] = np.nan
    X[rng.random(n) < 0.5, 2] = np.nan  # 第 3 個指標（再生能源）缺 50%

    Z = standardize_within(pd.DataFrame(X), pd.Series(industry)).to_numpy()
    return Z, Q, industry, is_noise, lam


def oracle_score(Z, lam):
    """天花板參考：已知真實 λ，面向內以 λ 加權平均，再三個面向等權平均。
    （實務上不可能知道 λ，只用來看各方法離「最好能做到」還有多遠）"""
    pillar = []
    for p in PILLARS:
        ix = [j for j in range(P) if PILLAR_OF[j] == p]
        w = np.where(np.isnan(Z[:, ix]), 0, lam[ix])
        pillar.append(np.nansum(Z[:, ix] * w, 1) / np.clip(w.sum(1), 1e-8, None))
    pillar = np.array(pillar).T
    return ((pillar - pillar.mean(0)) / pillar.std(0)).mean(1)


def tiers_within(score, industry, k=3):
    """產業內用 k-means 把分數分成三群，再依群中心排序成 0 = 差、1 = 中、2 = 優。"""
    out = np.empty(len(score), int)
    for g in np.unique(industry):
        ix = np.where(industry == g)[0]
        s = score[ix].reshape(-1, 1)
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(s)
        order = np.argsort(np.argsort(km.cluster_centers_.ravel()))
        out[ix] = order[km.labels_]
    return out


def true_tiers(Q, industry):
    out = np.empty(len(Q), int)
    for g in np.unique(industry):
        ix = np.where(industry == g)[0]
        out[ix] = np.digitize(Q[ix], np.quantile(Q[ix], [1 / 3, 2 / 3]))
    return out


def evaluate(score, Q, industry, weights, is_noise):
    w = np.abs(weights)
    return {
        "spearman": spearmanr(score, Q)[0],
        "tier_acc": float(np.mean(tiers_within(score, industry) == true_tiers(Q, industry))),
        "noise_wt": float(w[is_noise].sum() / w.sum()) if is_noise.any() else 0.0,
    }


METHODS = ["Equal weights", "PCA", "Pillar-weighted AE", "Oracle (true λ)"]


def one_rep(rng, r, cfg):
    Z, Q, ind, is_noise, lam = generate(rng, **cfg)
    eq = equal_weight_score(Z)
    pc, v = pca_score(Z)
    res = fit_pwae(Z, PILLAR_OF, seed=r, epochs=1000)
    # PWAE 的「整體指標權重」= 面向權重 × 面向內權重
    pw = np.array([res["pillar_weights"][PILLARS.index(p)] for p in PILLAR_OF])
    ae_w = pw * res["indicator_weights"]
    scored = [(METHODS[0], eq, np.ones(P)), (METHODS[1], pc, v),
              (METHODS[2], res["composite"], ae_w), (METHODS[3], oracle_score(Z, lam), lam)]
    return [{"method": m, **evaluate(s, Q, ind, w, is_noise)} for m, s, w in scored]


def run_scenarios(reps=20, seed=2026):
    rng = np.random.default_rng(seed)
    rows = []
    for name, cfg in SCENARIOS.items():
        for r in range(reps):
            rows += [{"scenario": name, "rep": r, **x} for x in one_rep(rng, r, cfg)]
            print(f"{name} rep {r + 1}/{reps}", flush=True)
    return pd.DataFrame(rows)


def run_noise_sweep(fracs=(0.0, 0.2, 0.4, 0.6), reps=10, seed=7):
    """掃描雜訊指標比例：看等權重從哪裡開始崩壞（其餘設定同情境 1）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for f in fracs:
        cfg = dict(noise_frac=f, nonlinear=False, heavy=False, miss=0.10)
        for r in range(reps):
            rows += [{"noise_frac": f, "rep": r, **x} for x in one_rep(rng, r, cfg)]
        print(f"noise sweep {f:.1f} done", flush=True)
    return pd.DataFrame(rows)


def main():
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df = run_scenarios()
    df.to_csv(out / "simulation_raw.csv", index=False)
    summary = (df.groupby(["scenario", "method"])[["spearman", "tier_acc", "noise_wt"]]
                 .agg(["mean", "std"]).round(3))
    summary.to_csv(out / "simulation_summary.csv")
    print(summary.to_string())

    sw = run_noise_sweep()
    sw.to_csv(out / "simulation_noise_sweep.csv", index=False)
    print(sw.groupby(["noise_frac", "method"])["spearman"].agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
