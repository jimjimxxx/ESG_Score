"""把證交所原始 JSON 整理成「公司 × ESG 指標」資料表。

輸出：
  data/processed/indicators_raw.csv   原始尺度的指標（方便檢查）
  data/processed/features.csv         轉換後的特徵：方向統一（越大越好）、產業內標準化
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# 證交所產業別代碼；家數太少（< 20 家）的產業併成較大的同業群組，否則產業內比較沒有意義
INDUSTRY = {
    "01": "水泥工業", "02": "食品工業", "03": "塑膠工業", "04": "紡織纖維", "05": "電機機械",
    "06": "電器電纜", "08": "玻璃陶瓷", "09": "造紙工業", "10": "鋼鐵工業", "11": "橡膠工業",
    "12": "汽車工業", "14": "建材營造", "15": "航運業", "16": "觀光餐旅", "17": "金融保險",
    "18": "貿易百貨", "20": "其他業", "21": "化學工業", "22": "生技醫療", "23": "油電燃氣",
    "24": "半導體業", "25": "電腦及週邊設備", "26": "光電業", "27": "通信網路", "28": "電子零組件",
    "29": "電子通路", "30": "資訊服務", "31": "其他電子", "35": "綠能環保", "36": "數位雲端",
    "37": "運動休閒", "38": "居家生活",
}
PEER_GROUP = {
    "01": "原物料（水泥、玻璃、造紙、橡膠、油電）", "08": "原物料（水泥、玻璃、造紙、橡膠、油電）",
    "09": "原物料（水泥、玻璃、造紙、橡膠、油電）", "11": "原物料（水泥、玻璃、造紙、橡膠、油電）",
    "23": "原物料（水泥、玻璃、造紙、橡膠、油電）",
    "05": "電機機械與電器電纜", "06": "電機機械與電器電纜",
    "30": "資訊服務與數位雲端", "36": "資訊服務與數位雲端",
    "16": "消費與服務（觀光、百貨、運動、居家）", "18": "消費與服務（觀光、百貨、運動、居家）",
    "37": "消費與服務（觀光、百貨、運動、居家）", "38": "消費與服務（觀光、百貨、運動、居家）",
}

# 指標定義：(欄名, 面向, 方向)；方向 +1 代表越大越好，-1 代表越小越好
INDICATORS = [
    ("ghg_intensity", "E", -1),        # (範疇一 + 範疇二) / 營收
    ("ghg_verified", "E", +1),         # 溫室氣體盤查取得外部驗證
    ("renewable_ratio", "E", +1),      # 再生能源使用率
    ("energy_certified", "E", +1),     # 能源管理取得驗證（如 ISO 50001）
    ("water_intensity", "E", -1),      # 用水量 / 營收
    ("waste_intensity", "E", -1),      # 廢棄物總量 / 營收
    ("avg_benefit", "S", +1),          # 員工福利平均數
    ("median_salary", "S", +1),        # 非主管全時員工薪資中位數
    ("female_manager_ratio", "S", +1), # 女性主管比例
    ("injury_rate", "S", -1),          # 職業災害比率
    ("fire_casualty", "S", -1),        # 是否有火災死傷
    ("indep_director_ratio", "G", +1), # 獨立董事比例
    ("female_director_ratio", "G", +1),# 女性董事比例
    ("board_attendance", "G", +1),     # 董事會出席率
    ("director_training", "G", +1),    # 董監事進修符合比率
    ("investor_conferences", "G", +1), # 年度法說會次數
    ("audit_attendance", "G", +1),     # 審計委員會出席率
    ("comp_indep_ratio", "G", +1),     # 薪酬委員會獨董比例
]
# 偏態嚴重的連續指標先取 log
LOG_COLS = ["ghg_intensity", "water_intensity", "waste_intensity", "avg_benefit",
            "median_salary", "investor_conferences"]


def load(name):
    df = pd.DataFrame(json.loads((RAW / f"{name}.json").read_text(encoding="utf-8")))
    return df.set_index("公司代號")


def num(s):
    """把 '12.3%'、'1,234'、'N/A' 這類字串轉成數字；無法轉換者為 NaN。"""
    s = s.astype(str).str.replace(",", "").str.replace("%", "").str.strip()
    return pd.to_numeric(s, errors="coerce")


def build_raw():
    e1, e2, e3, e4 = load("esg_01"), load("esg_02"), load("esg_03"), load("esg_04")
    s5, g6, g7, g9, s21 = load("esg_05"), load("esg_06"), load("esg_07"), load("esg_09"), load("esg_21")
    info, rev = load("company_info"), load("monthly_revenue")

    df = pd.DataFrame(index=e1.index)
    df["name"] = e1["公司名稱"]
    df["industry_code"] = info["產業別"].reindex(df.index)
    # 2025 年 1–9 月累計營收（仟元）：所有公司期間一致，當作規模分母
    df["revenue"] = num(rev["累計營業收入-去年累計營收"]).reindex(df.index)

    scope12 = num(e1["範疇一排放量(噸CO2e)"]) + num(e1["範疇二排放量(噸CO2e)"])
    df["ghg_intensity"] = scope12 / df["revenue"]
    df["ghg_verified"] = (e1["範疇一取得驗證"] == "是").astype(float)
    df["renewable_ratio"] = num(e2["使用率(再生能源/總能源)"])
    df["energy_certified"] = (~e2["取得驗證"].isin(["N/A", "", "無", "否"])).astype(float)
    df["water_intensity"] = num(e3["用水量(公噸)"]) / df["revenue"]
    df["waste_intensity"] = num(e4["總重量(有害+非有害)-數據(公噸)"]) / df["revenue"]

    df["avg_benefit"] = num(s5["員工福利平均數(仟元/人)"])
    # 欄名結尾有空白，用 startswith 找
    med_col = [c for c in s5.columns if c.startswith("非擔任主管之全時員工薪資中位數")][0]
    df["median_salary"] = num(s5[med_col])
    df["female_manager_ratio"] = num(s5["管理職女性主管佔比"])
    df["injury_rate"] = num(s21["職業災害人數及比率-比率"])
    df["fire_casualty"] = (num(s21["火災死傷人數(人)"]) > 0).astype(float)

    board = num(g6["董事席次(含獨立董事)(席)"])
    df["indep_director_ratio"] = num(g6["獨立董事席次(席)"]) / board
    df["female_director_ratio"] = num(g6["女性董事席次及比率-比率"])
    df["board_attendance"] = num(g6["董事出席董事會出席率"])
    df["director_training"] = num(g6["董監事進修時數符合進修要點比率"])
    df["investor_conferences"] = num(g7["公司年度召開法說會次數(次)"])
    df["audit_attendance"] = num(g9["審計委員會出席率"])
    comp = num(g9["薪酬委員會席次"])
    df["comp_indep_ratio"] = num(g9["薪酬委員會獨立董事席次"]) / comp.where(comp > 0)

    # 營收為 0 或缺漏時密集度無意義
    for c in ["ghg_intensity", "water_intensity", "waste_intensity"]:
        df.loc[~(df["revenue"] > 0), c] = np.nan

    # 排除存託憑證（91）與查無產業別者
    df = df[df["industry_code"].notna() & (df["industry_code"] != "91")].copy()
    df["industry"] = df["industry_code"].map(INDUSTRY)
    df["peer_group"] = df["industry_code"].map(PEER_GROUP).fillna(df["industry"])
    return df


def standardize_within(x, groups):
    """產業內穩健標準化（中位數 / MAD），截尾到 ±3。缺值保留 NaN。
    真實資料與模擬資料共用這個函式，確保前處理完全相同。"""

    def robust_z(g):
        med = g.median()
        mad = (g - med).abs().median() * 1.4826
        sd = g.std()
        # MAD 為 0（例如大多數公司都是 100% 出席）時退回標準差
        scale = mad.where(mad > 1e-9, sd).replace(0, np.nan)
        return ((g - med) / scale).clip(-3, 3)

    z = x.groupby(groups, group_keys=False).apply(robust_z)
    # 同業內完全沒有變異（例如整個產業都沒有火災死傷）→ 有資料的公司都在中位數，z = 0；
    # 只有原始資料本來就缺的才保留 NaN
    return z.mask(z.isna() & x.notna(), 0.0)


def to_features(df):
    """取 log、方向統一（越大越好），再做產業內標準化。"""
    cols = [c for c, _, _ in INDICATORS]
    x = df[cols].copy()
    for c in LOG_COLS:
        x[c] = np.log1p(x[c].clip(lower=0))
    for c, _, sign in INDICATORS:
        x[c] = sign * x[c]

    z = standardize_within(x, df["peer_group"])
    out = pd.concat([df[["name", "industry", "peer_group"]], z], axis=1)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_raw()
    df.to_csv(OUT / "indicators_raw.csv", encoding="utf-8-sig")
    feats = to_features(df)
    feats.to_csv(OUT / "features.csv", encoding="utf-8-sig")

    cols = [c for c, _, _ in INDICATORS]
    print(f"公司數：{len(df)}，同業群組數：{df['peer_group'].nunique()}")
    print(df["peer_group"].value_counts().to_string())
    print("\n各指標缺值比例：")
    print(feats[cols].isna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
