"""公式サイト HTML からの取得（オッズのみ・低頻度）。docs/00 §4, docs/07 §2。

注意：本作業環境（サンドボックス）からは boatrace.jp に到達できないため、実ページでは未検証。
      想定構造（3連単オッズ表）：
        <table class="is-w495"> ヘッダ行に 1着艇番 ×6、
        本体行は 2着艇（rowspan=4, ブロック先頭行のみ）・3着艇・オッズ の繰り返し ×6列。
      構造が想定と異なる場合は parse_odds3t が不完全な dict を返す → 呼び出し側で件数検査し NULL 扱い。
"""
from __future__ import annotations

import re
from datetime import date

from boatlab.config import OFFICIAL_ODDS3T
from boatlab.ingest.base import Fetcher
from boatlab.ingest.records import OddsRec, make_race_id
from boatlab.util import now_jst

_TAG = re.compile(r"<[^>]+>")


def _cells(row_html: str) -> list[str]:
    cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, flags=re.S)
    return [_TAG.sub("", c).replace("\n", "").strip() for c in cells]


def parse_odds3t(html: str) -> dict[str, float | None]:
    text = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    tables = re.findall(r"<table[^>]*class=\"[^\"]*is-w495[^\"]*\"[^>]*>(.*?)</table>", text, flags=re.S)
    out: dict[str, float | None] = {}
    for tb in tables:
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tb, flags=re.S)
        first_boats: list[int] = []
        carry: dict[int, int] = {}
        for r in rows:
            vals = _cells(r)
            if not first_boats:
                if len(vals) == 6 and all(re.fullmatch(r"\d", v) for v in vals):
                    first_boats = [int(v) for v in vals]
                continue
            seq: list[tuple[str, float | int | None]] = []
            for v in vals:
                if re.fullmatch(r"\d", v):
                    seq.append(("b", int(v)))
                elif re.fullmatch(r"\d+\.\d+", v):
                    seq.append(("o", float(v)))
                else:
                    seq.append(("o", None))  # 欠場・空欄
            i = 0
            for k in range(6):
                if i >= len(seq):
                    break
                second = None
                if seq[i][0] == "b" and i + 1 < len(seq) and seq[i + 1][0] == "b":
                    second = int(seq[i][1]); i += 1
                if i >= len(seq) or seq[i][0] != "b":
                    break
                third = int(seq[i][1]); i += 1
                odds = seq[i][1] if i < len(seq) and seq[i][0] == "o" else None
                i += 1
                if second is None:
                    second = carry.get(k)
                else:
                    carry[k] = second
                if second is not None:
                    out[f"{first_boats[k]}-{second}-{third}"] = odds
    return out


def fetch_odds3t(fetcher: Fetcher, d: date, stadium: int, rno: int) -> OddsRec | None:
    url = OFFICIAL_ODDS3T.format(rno=rno, jcd=stadium, yyyymmdd=d.strftime("%Y%m%d"))
    key = f"odds3t/{d:%Y%m%d}/{stadium:02d}_{rno:02d}_{now_jst():%H%M}.html"
    res = fetcher.fetch("official_web", url, key, use_cache=False)
    odds = parse_odds3t(res.content.decode("utf-8", errors="replace"))
    if len(odds) < 100:
        return None
    return OddsRec(race_id=make_race_id(d, stadium, rno), bet_type="3t", captured_at=now_jst(), source="official_web", odds=odds)
