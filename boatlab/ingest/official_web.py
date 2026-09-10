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

from boatlab.config import OFFICIAL_ODDS3T, OFFICIAL_ODDSTF
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


# ---------------------------------------------------------------- 単勝・複勝（oddstf）
_BOATCOLOR = re.compile(r"is-boatColor([1-6])")
_NUM = re.compile(r"\d+\.\d+")
_ROWSPLIT = re.compile(r"<tr[^>]*>", re.I)


def _boat_blocks(html: str) -> list[tuple[int, list[float]]]:
    """`is-boatColor{n}` の出現ごとに、その直後〜次の艇までに現れる小数を集める。

    表の入れ子や class 名の増減に影響されないよう、艇番の色クラスだけを手がかりにする。
    """
    out: list[tuple[int, list[float]]] = []
    marks = list(_BOATCOLOR.finditer(html))
    for k, m in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(html)
        seg = _TAG.sub(" ", html[m.end():end])
        out.append((int(m.group(1)), [float(x) for x in _NUM.findall(seg)]))
    return out


def _runs(blocks: list[tuple[int, list[float]]]) -> list[list[list[float]]]:
    """艇番が 1→6 と並ぶ塊に切り出す（1つの塊＝1つのオッズ表）。数値の無い塊は捨てる。"""
    runs, cur = [], []
    for b, nums in blocks:
        if b == len(cur) + 1:
            cur.append(nums)
        else:
            if len(cur) == 6:
                runs.append(cur)
            cur = [nums] if b == 1 else []
        if len(cur) == 6:
            runs.append(cur)
            cur = []
    return [r for r in runs if all(len(x) >= 1 for x in r)]


def parse_oddstf(html: str) -> dict[str, dict]:
    """単勝・複勝オッズ。戻り値 {'win': {'1': 1.5, ...}, 'place': {'1': {'lo':..,'hi':..}, ...}}。

    ページ内の「艇番1〜6が並ぶ塊」を全部拾い、数値が1つだけの塊＝単勝、2つ以上の塊＝複勝とみなす
    （複勝は下限・上限の2つが出る）。見出しの位置や表のクラス名には依存しない。
    6艇そろった単勝が取れなければ空 dict を返す（呼び出し側で NULL 扱い）。
    """
    text = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    runs = _runs(_boat_blocks(text))
    win_run = next((r for r in runs if max(len(x) for x in r) == 1), None)
    place_run = next((r for r in runs if min(len(x) for x in r) >= 2), None)
    if win_run is None and len(runs) >= 1:
        win_run = runs[0]                       # 単勝側にも余分な数値が入っていた場合
    if win_run is None:
        return {}
    win = {str(i + 1): win_run[i][0] for i in range(6)}
    place = {}
    if place_run is not None and place_run is not win_run:
        place = {str(i + 1): {"lo": place_run[i][0], "hi": place_run[i][1]} for i in range(6)}
    return {"win": win, "place": place}


def digest_oddstf(html: str) -> str:
    """パーサが外れたときに構造を報告するための要約（1画面に収まる短さ）。"""
    text = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    marks = _BOATCOLOR.findall(text)
    lines = [f"len={len(html)} is-boatColor={len(marks)}{'' if not marks else ' 順=' + ''.join(marks[:14])}"]
    classes = sorted({c for c in re.findall(r'class="([^"]{0,60})"', text)
                      if re.search(r"boat|odds|oddsPoint|numberSet", c, re.I)})[:12]
    lines.append("class候補: " + (" | ".join(classes) if classes else "なし"))
    blocks = _boat_blocks(text)
    for b, nums in blocks[:8]:
        lines.append(f"  艇{b}: {nums[:4]}")
    lines.append(f"runs={[[len(x) for x in r] for r in _runs(blocks)][:4]}")
    if not marks:
        i = max(text.find("単勝"), 0)
        lines.append("単勝付近: " + _TAG.sub(" ", text[i:i + 200]).replace("\n", " ")[:160])
    return "\n".join(lines)


def fetch_oddstf(fetcher: Fetcher, d: date, stadium: int, rno: int, tag: str = "") -> list[OddsRec]:
    """単勝・複勝オッズを1回の取得で両方取る。失敗時は空リスト。"""
    url = OFFICIAL_ODDSTF.format(rno=rno, jcd=stadium, yyyymmdd=d.strftime("%Y%m%d"))
    key = f"oddstf/{d:%Y%m%d}/{stadium:02d}_{rno:02d}_{now_jst():%H%M}{tag}.html"
    res = fetcher.fetch("official_web", url, key, use_cache=False)
    parsed = parse_oddstf(res.content.decode("utf-8", errors="replace"))
    if not parsed:
        return []
    now = now_jst()
    rid = make_race_id(d, stadium, rno)
    recs = [OddsRec(race_id=rid, bet_type="win", captured_at=now, source="official_web", odds=parsed["win"])]
    if parsed.get("place"):
        recs.append(OddsRec(race_id=rid, bet_type="place", captured_at=now, source="official_web", odds=parsed["place"]))
    return recs


def digest_odds3t(html: str) -> str:
    """3連単オッズ表が読めないときに構造を報告する（1画面に収まる短さ）。"""
    text = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    tables = re.findall(r"<table([^>]*)>(.*?)</table>", text, flags=re.S)
    lines = [f"len={len(html)} tables={len(tables)} 小数の総数={len(_NUM.findall(_TAG.sub(' ', text)))}"
             f" is-w495={len(re.findall(r'is-w495', text))} parse={len(parse_odds3t(html))}"]
    for i, (attr, body) in enumerate(tables[:4]):
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", body, flags=re.S)
        cls = (re.search(r'class="([^"]*)"', attr) or [None, "-"])[1]
        lines.append(f"[表{i}] class={cls[:40]} tr={len(rows)}")
        for r in rows[:3]:
            lines.append("   " + str(_cells(r))[:110])
    return "\n".join(lines)
