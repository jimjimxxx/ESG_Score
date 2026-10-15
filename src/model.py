"""ESG 綜合分數模型。

主模型：階層式加權 autoencoder（Pillar-Weighted Autoencoder）
  1. 每個面向 p ∈ {E, S, G} 內，用 softmax 權重把指標加權平均成面向分數 s_p
     （權重非負且加總為 1 → 可直接解讀成「這個指標佔該面向多少比重」）
  2. 解碼器（小型 MLP）只看 (s_E, s_S, s_G) 去重建全部指標
  3. 只在「有觀測到的格子」計算重建誤差（masked MSE），缺值不需要先補
  訓練完成後，面向權重 = 把該面向分數換成平均值時，重建誤差上升的比例（資訊量）。

對照組：
  equal_weight  — 所有可觀測指標的簡單平均
  pca           — 補值後第一主成分（符號調整成與多數指標同向）
"""
import numpy as np
import torch
import torch.nn as nn

PILLARS = ("E", "S", "G")


class PillarWeightedAE(nn.Module):
    def __init__(self, pillar_of, hidden=16):
        super().__init__()
        self.pillar_of = list(pillar_of)
        self.idx = {p: [j for j, q in enumerate(self.pillar_of) if q == p] for p in PILLARS}
        # 每個面向一組可學習的 logits，softmax 後就是指標權重
        self.logits = nn.ParameterDict({p: nn.Parameter(torch.zeros(len(ix))) for p, ix in self.idx.items()})
        self.decoder = nn.Sequential(
            nn.Linear(len(PILLARS), hidden), nn.Tanh(), nn.Linear(hidden, len(self.pillar_of))
        )

    def weights(self):
        return {p: torch.softmax(self.logits[p], dim=0) for p in PILLARS}

    def pillar_scores(self, x, m):
        """x：缺值已補 0 的特徵；m：觀測遮罩。缺值的指標權重會在該公司內重新正規化。"""
        out = []
        for p, w in self.weights().items():
            ix = self.idx[p]
            wm = w * m[:, ix]
            denom = wm.sum(1).clamp_min(1e-8)
            out.append((wm * x[:, ix]).sum(1) / denom)
        return torch.stack(out, dim=1)

    def forward(self, x, m):
        s = self.pillar_scores(x, m)
        return self.decoder(s), s


def _masked_mse(xhat, x, m):
    return ((xhat - x) ** 2 * m).sum() / m.sum()


def fit_pwae(X, pillar_of, seed=0, epochs=1500, lr=0.01, hidden=16):
    """X：n × p 的 numpy 陣列（可含 NaN）。回傳訓練好的模型與結果字典。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = torch.tensor(~np.isnan(X), dtype=torch.float32)
    x = torch.tensor(np.nan_to_num(X, nan=0.0), dtype=torch.float32)

    model = PillarWeightedAE(pillar_of, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        xhat, _ = model(x, m)
        loss = _masked_mse(xhat, x, m)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        xhat, s = model(x, m)
        base = _masked_mse(xhat, x, m).item()
        # 面向重要性：把某一面向的分數換成它的平均值，看重建誤差上升多少
        gain = []
        for k in range(len(PILLARS)):
            s2 = s.clone()
            s2[:, k] = s[:, k].mean()
            gain.append(_masked_mse(model.decoder(s2), x, m).item() - base)
        gain = np.clip(np.array(gain), 0, None)
        pillar_w = gain / gain.sum() if gain.sum() > 0 else np.full(len(PILLARS), 1 / len(PILLARS))

        s_np = s.numpy()
        s_std = (s_np - s_np.mean(0)) / s_np.std(0).clip(1e-8)
        composite = s_std @ pillar_w
        w = model.weights()
        indicator_w = np.zeros(len(pillar_of))
        for p in PILLARS:
            indicator_w[model.idx[p]] = w[p].numpy()

    return {
        "model": model,
        "recon_mse": base,
        "pillar_scores": s_np,
        "pillar_weights": pillar_w,
        "indicator_weights": indicator_w,  # 面向內權重（每個面向內加總為 1）
        "composite": composite,
    }


def fit_pwae_ensemble(X, pillar_of, seeds=range(10), **kw):
    """多個隨機種子各訓練一次，取平均；同時回傳跨種子的標準差當作穩定度。"""
    runs = [fit_pwae(X, pillar_of, seed=s, **kw) for s in seeds]
    stack = lambda k: np.stack([r[k] for r in runs])
    return {
        "composite": stack("composite").mean(0),
        "composite_runs": stack("composite"),
        "pillar_scores": stack("pillar_scores").mean(0),
        "pillar_weights": stack("pillar_weights").mean(0),
        "pillar_weights_sd": stack("pillar_weights").std(0),
        "indicator_weights": stack("indicator_weights").mean(0),
        "indicator_weights_sd": stack("indicator_weights").std(0),
        "recon_mse": float(np.mean([r["recon_mse"] for r in runs])),
    }


# ---------- 對照組 ----------

def equal_weight_score(X):
    return np.nanmean(X, axis=1)


def pca_score(X):
    """以欄平均（標準化後約為 0）補值，取第一主成分。"""
    Xf = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    Xc = Xf - Xf.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    v = vt[0]
    if v.sum() < 0:  # 讓多數指標的負荷量為正，分數方向才是「越大越好」
        v = -v
    return Xc @ v, v
