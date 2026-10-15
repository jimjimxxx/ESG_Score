"""從證交所 OpenAPI 下載上市公司 ESG 揭露資料與公司基本資料（含產業別）。"""
import json
import time
from pathlib import Path

import requests

BASE = "https://openapi.twse.com.tw/v1"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# 21 個 ESG 揭露主題（t187ap46_L_1 ~ L_21）＋ 上市公司基本資料（t187ap03_L，含產業別）
# ＋ 每月營收彙總（t187ap05_L，用「去年累計營收」當規模分母）
ENDPOINTS = {f"esg_{i:02d}": f"/opendata/t187ap46_L_{i}" for i in range(1, 22)}
ENDPOINTS["company_info"] = "/opendata/t187ap03_L"
ENDPOINTS["monthly_revenue"] = "/opendata/t187ap05_L"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in ENDPOINTS.items():
        resp = requests.get(BASE + path, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        (RAW_DIR / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"{name:13s} {len(data):5d} 筆")
        time.sleep(0.5)  # 避免連續請求過快


if __name__ == "__main__":
    main()
