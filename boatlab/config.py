"""設定。環境変数で上書きできる。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("BOATLAB_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = Path(os.environ.get("BOATLAB_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
DATABASE_URL = os.environ.get("BOATLAB_DATABASE_URL", f"sqlite:///{DATA_DIR / 'lab.db'}")


@dataclass(frozen=True)
class SourceSpec:
    """外部ソースの定義。URL テンプレートと最小アクセス間隔（秒）。"""

    name: str
    url_template: str
    min_interval_sec: float
    max_per_day: int | None = None
    headers: dict[str, str] = field(default_factory=dict)


# --- 過去データ（Boatrace Open API v3, gh-pages/docs 配下） ---
# raw.githubusercontent.com と github.io の両方で同じ内容。環境によって到達できる方を使う。
OPENAPI_V3_BASE = os.environ.get(
    "BOATLAB_OPENAPI_V3_BASE",
    "https://raw.githubusercontent.com/BoatraceOpenAPI/{repo}/gh-pages/docs/v3/{yyyy}/{yyyymmdd}.json",
)
OPENAPI_V3_BASE_ALT = "https://boatraceopenapi.github.io/{repo}/v3/{yyyy}/{yyyymmdd}.json"

# --- 2026-01〜 出走表+直前+オッズ+結果（turnmark/api v1, 前日まで） ---
TURNMARK_BASE = os.environ.get(
    "BOATLAB_TURNMARK_BASE",
    "https://raw.githubusercontent.com/turnmark/api/gh-pages/docs/v1/{yyyy}/{yyyymmdd}.json",
)
TURNMARK_BASE_ALT = "https://turnmark.github.io/api/v1/{yyyy}/{yyyymmdd}.json"

# --- 当日（BoatraceOpenAPI/api v1, 約3分更新） ---
OPENAPI_API_TODAY = os.environ.get(
    "BOATLAB_OPENAPI_API_TODAY", "https://boatraceopenapi.github.io/api/v1/today.json"
)
OPENAPI_API_TODAY_ALT = "https://raw.githubusercontent.com/BoatraceOpenAPI/api/gh-pages/docs/v1/today.json"
OPENAPI_API_DAY = "https://raw.githubusercontent.com/BoatraceOpenAPI/api/gh-pages/docs/v1/{yyyy}/{yyyymmdd}.json"

# --- 公式サイト（オッズのみ。低頻度） ---
OFFICIAL_ODDS3T = "https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={yyyymmdd}"
OFFICIAL_BEFOREINFO = "https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd:02d}&hd={yyyymmdd}"

SOURCES: dict[str, SourceSpec] = {
    "openapi_v3": SourceSpec("openapi_v3", OPENAPI_V3_BASE, min_interval_sec=0.4),
    "turnmark": SourceSpec("turnmark", TURNMARK_BASE, min_interval_sec=0.4),
    "openapi_api": SourceSpec("openapi_api", OPENAPI_API_TODAY, min_interval_sec=1.0),
    # 公式サイトは 3 秒間隔・1 日 1,000 回をハードリミットにする（docs/07 §2）
    "official_web": SourceSpec(
        "official_web",
        OFFICIAL_ODDS3T,
        min_interval_sec=3.0,
        max_per_day=1000,
        headers={"User-Agent": "boatlab/0.1 (personal research; low-frequency)"},
    ),
}

STADIUMS: dict[int, str] = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖", 7: "蒲郡", 8: "常滑",
    9: "津", 10: "三国", 11: "びわこ", 12: "住之江", 13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島",
    17: "宮島", 18: "徳山", 19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}

# Open API / turnmark の番号 → 表記（turnmark の _source から実測で確認、docs/00 参照）
WEATHER = {1: "晴", 2: "曇り", 3: "雨"}
WIND_DIR = {
    1: "北", 2: "北北東", 3: "北東", 4: "東北東", 5: "東", 6: "東南東", 7: "南東", 8: "南南東",
    9: "南", 10: "南南西", 11: "南西", 12: "西南西", 13: "西", 14: "西北西", 15: "北西", 16: "北北西", 17: "無風",
}
TECHNIQUE = {1: "逃げ", 2: "差し", 3: "まくり", 4: "まくり差し", 5: "抜き", 6: "恵まれ"}
GRADE = {1: "SG", 2: "G1", 3: "G2", 4: "G3", 5: "一般"}
RACER_CLASS = {1: "A1", 2: "A2", 3: "B1", 4: "B2"}
PLACE_CODE = {
    7: "妨", 8: "エ", 9: "転", 10: "落", 11: "沈", 12: "不", 13: "失", 14: "F", 15: "L", 16: "欠", 99: "他",
}
