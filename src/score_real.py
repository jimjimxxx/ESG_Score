"""用真實資料（證交所 114 年度 ESG 揭露）計算每家上市公司的 ESG 綜合分數、同業比較與分級。

輸出：
  results/firm_scores.csv        每家公司的分數、同業百分位、分級、與領先群的相似度、落後最多的指標
  results/weights.csv            學到的指標權重（10 個種子的平均 ± 標準差）
  results/real_data_summary.txt  主要數字
  figures/weights.png、figures/peer_semiconductor.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score

from build_dataset import INDICATORS
from model import PILLARS, equal_weight_score, fit_pwae_ensemble, pca_score
from simulate import tiers_within

ROOT = Path(__file__).resolve().parent.parent
COLS = [c for c, _, _ in INDICATORS]
PILLAR_OF = [p for _, p, _ in INDICATORS]
LABELS = {
    "ghg_intensity": "GHG intensity (scope 1+2 / revenue)", "ghg_verified": "GHG inventory verified",
    "renewable_ratio": "Renewable energy share", "energy_certified": "Energy management certified",
    "water_intensity": "Water intensity", "waste_intensity": "Waste intensity",
    "avg_benefit": "Average employee benefits", "median_salary": "Median non-manager salary",
    "female_manager_ratio": "Female managers (%)", "injury_rate": "Occupational injury rate",
    "fire_casualty": "Fire casualties", "indep_director_ratio": "Independent directors (%)",
    "female_director_ratio": "Female directors (%)", "board_attendance": "Board attendance",
    "director_training": "Director training compliance", "investor_conferences": "Investor conferences",
    "audit_attendance": "Audit committee attendance", "comp_indep_ratio": "Independent comp. committee (%)",
}
TIER_NAME = {2: "Excellent", 1: "Moderate", 0: "Poor"}
# 圖表配色：面向用類別色前三格；分級是有序的 → 同一色相由淺到深
PILLAR_COLOR = {"E": "#2a78d6", "S": "#eb6834", "G": "#1baf7a"}
TIER_COLOR = {0: "#86b6ef", 1: "#2a78d6", 2: "#104281"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": ["Calibri", "Microsoft JhengHei", "DejaVu Sans"],
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def peer_analysis(f, composite):
    """同業內：與領先群（前 20%）輪廓的相似度、最相近的 3 家同業、落後領先群最多的 3 個指標。"""
    Z = f[COLS].to_numpy(float)
    Z0 = np.nan_to_num(Z, nan=0.0)  # 標準化後 0 = 同業中位數
    sim = np.full(len(f), np.nan)
    gaps, nearest = [""] * len(f), [""] * len(f)
    for g in f["peer_group"].unique():
        ix = np.where(f["peer_group"].to_numpy() == g)[0]
        cut = np.quantile(composite[ix], 0.8)
        leaders = ix[composite[ix] >= cut]
        centroid = np.nanmean(Z[leaders], axis=0)
        centroid = np.nan_to_num(centroid, nan=0.0)
        for i in ix:
            sim[i] = cosine(Z0[i], centroid)
            gap = centroid - Z0[i]
            top = np.argsort(-gap)[:3]
            gaps[i] = "; ".join(LABELS[COLS[j]] for j in top if gap[j] > 0.5)
            others = [k for k in ix if k != i]
            sims = np.array([cosine(Z0[i], Z0[k]) for k in others])
            nearest[i] = "、".join(f.iloc[others[k]]["name"] for k in np.argsort(-sims)[:3])
    return sim, gaps, nearest


def plot_weights(res, path):
    pw = np.array([res["pillar_weights"][PILLARS.index(p)] for p in PILLAR_OF])
    overall = pw * res["indicator_weights"]
    sd = pw * res["indicator_weights_sd"]
    order = np.argsort(overall)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    y = np.arange(len(COLS))
    ax.barh(y, overall[order], xerr=sd[order], height=0.62,
            color=[PILLAR_COLOR[PILLAR_OF[j]] for j in order],
            error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
    ax.set_yticks(y, [LABELS[COLS[j]] for j in order])
    ax.set_xlabel("Learned weight in composite score (pillar weight × within-pillar weight)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PILLAR_COLOR[p]) for p in PILLARS]
    pw_txt = [f"{p}  (pillar weight {res['pillar_weights'][k]:.2f})" for k, p in enumerate(PILLARS)]
    ax.legend(handles, pw_txt, frameon=False, loc="lower right")
    ax.set_title("What the autoencoder learned: indicator weights (mean ± sd over 10 seeds)",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_peer_group(out, group, path, label_codes):
    d = out[out["peer_group"] == group]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for t in (0, 1, 2):
        s = d[d["tier_code"] == t]
        ax.scatter(s["composite"], s["leader_similarity"], s=36, color=TIER_COLOR[t],
                   edgecolor="white", linewidth=1, label=f"{TIER_NAME[t]} (n={len(s)})", zorder=3)
    for code in label_codes:
        if code in d.index:
            r = d.loc[code]
            ax.annotate(f"{r['name_en']} ({code})", (r["composite"], r["leader_similarity"]),
                        xytext=(5, 4), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("Composite ESG score (within-industry standardized)")
    ax.set_ylabel("Profile similarity to industry leaders (cosine)")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"Semiconductor industry: score vs. similarity to top-20% leaders (n={len(d)})",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    f = pd.read_csv(ROOT / "data" / "processed" / "features.csv", index_col=0, dtype={0: str})
    f.index = f.index.astype(str)
    X = f[COLS].to_numpy(float)

    res = fit_pwae_ensemble(X, PILLAR_OF, seeds=range(10), epochs=1500)
    composite = res["composite"]
    eq = equal_weight_score(X)
    pc, _ = pca_score(X)

    group = f["peer_group"].to_numpy()
    codes = pd.Series(group).astype("category").cat.codes.to_numpy()
    tier = tiers_within(composite, codes)
    sim, gaps, nearest = peer_analysis(f, composite)

    out = f[["name", "industry", "peer_group"]].copy()
    info = pd.read_json(ROOT / "data" / "raw" / "company_info.json", dtype={"公司代號": str})
    out.insert(1, "name_en", info.set_index("公司代號")["英文簡稱"].reindex(out.index).str.strip())
    for k, p in enumerate(PILLARS):
        out[f"pillar_{p}"] = res["pillar_scores"][:, k].round(3)
    out["composite"] = composite.round(3)
    out["peer_percentile"] = (pd.Series(composite, index=f.index)
                              .groupby(group).rank(pct=True).round(3))
    out["tier_code"] = tier
    out["tier"] = [TIER_NAME[t] for t in tier]
    out["leader_similarity"] = np.round(sim, 3)
    out["largest_gaps_to_leaders"] = gaps
    out["most_similar_peers"] = nearest
    out["equal_weight_score"] = np.round(eq, 3)
    out.index.name = "code"
    out.sort_values(["peer_group", "composite"], ascending=[True, False]).to_csv(
        ROOT / "results" / "firm_scores.csv", encoding="utf-8-sig")

    pd.DataFrame({
        "indicator": COLS, "pillar": PILLAR_OF,
        "within_pillar_weight": res["indicator_weights"].round(4),
        "within_pillar_weight_sd": res["indicator_weights_sd"].round(4),
        "overall_weight": (np.array([res["pillar_weights"][PILLARS.index(p)] for p in PILLAR_OF])
                           * res["indicator_weights"]).round(4),
    }).to_csv(ROOT / "results" / "weights.csv", index=False)

    # 跨種子穩定度：兩兩 Spearman 的平均
    runs = res["composite_runs"]
    stab = np.mean([spearmanr(runs[i], runs[j])[0]
                    for i in range(len(runs)) for j in range(i + 1, len(runs))])
    sil = []
    for g in np.unique(codes):
        ix = codes == g
        sil.append(silhouette_score(composite[ix].reshape(-1, 1), tier[ix]))
    lines = [
        f"firms: {len(f)}, peer groups: {len(np.unique(codes))}",
        f"pillar weights E/S/G: {np.round(res['pillar_weights'], 3)} (sd {np.round(res['pillar_weights_sd'], 3)})",
        f"seed stability (mean pairwise Spearman over 10 seeds): {stab:.3f}",
        f"Spearman composite vs equal weights: {spearmanr(composite, eq)[0]:.3f}",
        f"Spearman composite vs PCA: {spearmanr(composite, pc)[0]:.3f}",
        f"Spearman equal weights vs PCA: {spearmanr(eq, pc)[0]:.3f}",
        f"Spearman composite vs leader similarity: {spearmanr(composite, sim)[0]:.3f}",
        f"tier counts: {pd.Series(out['tier']).value_counts().to_dict()}",
        f"mean silhouette of within-group tiers: {np.mean(sil):.3f}",
    ]
    (ROOT / "results" / "real_data_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    plot_weights(res, ROOT / "figures" / "weights.png")
    plot_peer_group(out, "半導體業", ROOT / "figures" / "peer_semiconductor.png",
                    ["2330", "2303", "2454", "3711", "2379", "3034", "5347"])


if __name__ == "__main__":
    main()
