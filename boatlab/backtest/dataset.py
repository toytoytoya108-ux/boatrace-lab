"""バックテスト用データセット：全レースの as-of 特徴量＋ラベル＋レース情報（払戻・返還・実オッズ）。

特徴量は parquet にキャッシュする（feature_set_version をファイル名に含める）。
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from boatlab.config import DATA_DIR
from boatlab.features.build import FEATURE_SET_VERSION, attach_labels, build_features
from boatlab.features.history import HistoryFrames, load_history
from boatlab.model.trifecta import PERM_LABELS, combo_index
from boatlab.store.db import get_engine


def _cache_path(d0: date, d1: date) -> Path:
    p = DATA_DIR / "features"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{FEATURE_SET_VERSION}_{d0}_{d1}.parquet"


def _downcast(x: pd.DataFrame) -> pd.DataFrame:
    for c in x.columns:
        if x[c].dtype == "float64":
            x[c] = x[c].astype("float32")
    return x


def build_entry_dataset(d0: date, d1: date, use_cache: bool = True, columns: list[str] | None = None) -> pd.DataFrame:
    """艇単位の特徴量＋ラベル。

    メモリ対策：累積テーブル（as-of）は全期間から1回だけ作り、対象は半年ごとのチャンクで
    処理して float32 でチャンク別 parquet に保存する（7GB RAM の環境で全期間を扱うため）。
    """
    cp = _cache_path(d0, d1)
    if use_cache and cp.exists():
        return pd.read_parquet(cp, columns=columns)
    from boatlab.features.asof import AsofTables, perf_log_ext
    from boatlab.features.perf_history import load_ext_frames, load_perf_frames

    races, entries_min, result_entries = load_perf_frames(d0, d1)
    results_k, previews_e, conditions_w = load_ext_frames(d0, d1)
    tables = AsofTables(perf_log_ext(races, entries_min, result_entries, results_k, previews_e, conditions_w))
    del races, entries_min, results_k, previews_e, conditions_w

    from datetime import timedelta

    chunks: list[Path] = []
    start = d0
    while start <= d1:
        end = min((pd.Timestamp(start) + pd.DateOffset(months=6) - pd.Timedelta(days=1)).date(), d1)
        ccp = _cache_path(start, end)
        if not ccp.exists():
            target = load_history(start, end)
            if not len(target.races):  # データが無い期間（空チャンク）はスキップ
                start = end + timedelta(days=1)
                continue
            tf = HistoryFrames(target.races, target.entries, target.previews, target.conditions, pd.DataFrame(),
                               pd.DataFrame(columns=target.result_entries.columns))
            x = build_features(tf, tables)
            x = attach_labels(x, target.result_entries)
            x = x.drop(columns=["name"], errors="ignore")
            x = _downcast(x.sort_values(["race_date", "race_id", "lane"]).reset_index(drop=True))
            x.to_parquet(ccp, index=False)
            del target, tf, x
        chunks.append(ccp)
        start = end + timedelta(days=1)
    del tables
    import gc
    frames = []
    for c in chunks:
        f = pd.read_parquet(c, columns=columns)
        for col in f.columns:
            if f[col].dtype == "object" and col not in ("series_results", "parts"):
                f[col] = f[col].astype("category")
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    return out


def build_race_dataset(d0: date, d1: date) -> pd.DataFrame:
    """レース単位：結果・払戻・返還・実オッズ（あれば）。payouts/odds の JSON はチャンクで処理（メモリ対策）。"""
    eng = get_engine()
    r = pd.read_sql_query(text("""
        SELECT r.id AS race_id, r.race_date, r.stadium_code, r.race_no, r.closed_at, r.grade, r.status,
               res.trifecta, res.trifecta_payout, res.is_irregular, res.refunds
        FROM races r LEFT JOIN results res ON res.race_id = r.id
        WHERE r.race_date BETWEEN :d0 AND :d1 ORDER BY r.race_date, r.id"""), eng, params={"d0": str(d0), "d1": str(d1)})
    r["race_date"] = pd.to_datetime(r["race_date"])
    r["tri_idx"] = r["trifecta"].map(lambda s: -1 if s is None or pd.isna(s) else (combo_index(s) if combo_index(s) is not None else -1))
    r["refund_lanes"] = r["refunds"].map(lambda s: json.loads(s) if isinstance(s, str) else (s or []))
    r = r.drop(columns=["refunds"])
    r["tri_amount"] = r["trifecta_payout"]

    def _payout(d, bt):
        for p in (d.get(bt) or []):
            if p.get("combination") and p.get("amount") is not None:
                return p["combination"], int(p["amount"])
        return None, None

    pay = {}
    with eng.connect() as conn:
        cur = conn.execution_options(stream_results=True).execute(text(
            """SELECT res.race_id, res.payouts FROM results res JOIN races x ON x.id=res.race_id
               WHERE x.race_date BETWEEN :d0 AND :d1"""), {"d0": str(d0), "d1": str(d1)})
        while True:
            rows = cur.fetchmany(20000)
            if not rows:
                break
            for rid, js in rows:
                try:
                    d = json.loads(js) if isinstance(js, str) else (js or {})
                except Exception:
                    d = {}
                w = _payout(d, "win")
                e = _payout(d, "exacta")
                pay[rid] = (int(w[0]) if w[0] and str(w[0]).isdigit() else np.nan, w[1],
                            int(e[0].split("-")[0]) if e[0] and "-" in str(e[0]) else np.nan,
                            int(e[0].split("-")[1]) if e[0] and "-" in str(e[0]) else np.nan, e[1])
    pdf = pd.DataFrame.from_dict(pay, orient="index", columns=["win_lane", "win_amount", "ex_a", "ex_b", "ex_amount"])
    pdf.index.name = "race_id"
    r = r.merge(pdf.reset_index(), on="race_id", how="left")
    del pay, pdf

    real = {}
    with eng.connect() as conn:
        cur = conn.execution_options(stream_results=True).execute(text(
            """SELECT o.race_id, o.odds FROM odds_snapshots o JOIN races x ON x.id=o.race_id
               WHERE o.bet_type='3t' AND o.source='turnmark_final' AND x.race_date BETWEEN :d0 AND :d1"""),
            {"d0": str(d0), "d1": str(d1)})
        while True:
            rows = cur.fetchmany(5000)
            if not rows:
                break
            for rid, js in rows:
                try:
                    d = json.loads(js) if isinstance(js, str) else js
                    real[rid] = np.array([np.nan if d.get(k) is None else d[k] for k in PERM_LABELS], dtype=np.float32)
                except Exception:
                    pass
    r["real_odds"] = r["race_id"].map(real)
    return r
