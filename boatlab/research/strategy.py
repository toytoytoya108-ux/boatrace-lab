"""買い方（絞り込み型のパラメータ）の自動研究。夜間ジョブで実行し、結果を JSON で保存する。

方針（多重検定で「たまたま良かった買い方」を選ばないための縛り）:
  - 候補は事前に固定（GRID）。増やさない。
  - 3つの独立した期間で評価する:
      bt_explore : 2026-01〜05 のバックテスト（実オッズ、研究データ研究_data/bt2026_top20.npz）
      bt_confirm : 2026-06〜08 のバックテスト（同上。探索で選んだ候補の確認用）
      live       : 本番運用で保存した確定予想（実オッズ取得分）× 実結果
  - 順送り検証（rolling）: 各月について「その月より前のデータで最良だった候補」をその月に当てはめ、
    つなげた成績を出す。自動選択に従い続けた場合に実際に得られたはずの成績＝最も正直な1つの数字。
  - 設定は変えない。候補の印付けまで。採用はユーザーが設定タブで行う。
"""
from __future__ import annotations

import itertools
import json
import logging
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from boatlab.backtest.metrics import roi_bootstrap
from boatlab.config import DATA_DIR, ROOT
from boatlab.model.selection import FocusedParams, select_focused
from boatlab.model.trifecta import PERM_LABELS, combo_index
from boatlab.store.db import get_engine
from boatlab.util import now_jst

log = logging.getLogger(__name__)

BT_FILE = Path(ROOT) / "research_data" / "bt2026_top20.npz"
OUT_FILE = Path(DATA_DIR) / "research" / "strategy_research.json"
EXPLORE_END = "2026-05-31"
MIN_ROLL_N = 300      # 順送りで「前期間の最良」を選ぶのに必要な最小レース数（回収率95%区間の下限で選ぶ）

# 候補グリッド（固定）: 期待値下限 × オッズ上限 × 自信下限 × 最大点数
GRID = [dict(ev_min=e, odds_hi=h, s15_min=s, max_points=m)
        for e, h, s, m in itertools.product((0.8, 0.9, 1.0), (15, 20, 50), (0.70, 0.73), (3, 5))]


def label_of(v: dict) -> str:
    return f"EV≥{v['ev_min']}・5〜{v['odds_hi']}倍・自信≥{v['s15_min']}・最大{v['max_points']}点"


# ---------------------------------------------------------------- データ
def load_backtest() -> dict | None:
    if not BT_FILE.exists():
        return None
    z = np.load(BT_FILE, allow_pickle=False)
    n, k = z["top"].shape
    P = np.zeros((n, 120), np.float32)
    O = np.full((n, 120), np.nan, np.float32)
    idx = np.arange(n)[:, None]
    P[idx, z["top"]] = z["p"].astype(np.float32)
    O[idx, z["top"]] = z["odds"]
    return dict(kind="bt", date=z["date"].astype(str), stadium=z["stadium"], P=P, O=O, tri=z["tri"].astype(int),
                payout=z["payout"].astype(float), model=str(z["model"]))


def load_live() -> dict:
    """本番の確定予想（active・final・実オッズ）と結果。"""
    rows = pd.read_sql_query(text("""
        SELECT p.race_id, r.race_date, r.stadium_code, p.probs, p.odds_used, p.flags, res.trifecta, res.trifecta_payout
        FROM predictions p JOIN races r ON r.id = p.race_id JOIN results res ON res.race_id = p.race_id
        WHERE p.role = 'active' AND p.stage = 'final' AND r.status != 'cancelled' AND res.trifecta IS NOT NULL
        ORDER BY r.race_date, p.race_id"""), get_engine())
    keep, P, O, tri, pay = [], [], [], [], []
    for _, x in rows.iterrows():
        flags = json.loads(x["flags"]) if isinstance(x["flags"], str) else (x["flags"] or {})
        if flags.get("odds_estimated"):
            continue
        probs = json.loads(x["probs"]) if isinstance(x["probs"], str) else x["probs"]
        odds = json.loads(x["odds_used"]) if isinstance(x["odds_used"], str) else x["odds_used"]
        t = combo_index(x["trifecta"])
        if t is None or t < 0:
            continue
        P.append([float(probs.get(k, 0.0)) for k in PERM_LABELS])
        O.append([np.nan if odds.get(k) is None else float(odds[k]) for k in PERM_LABELS])
        tri.append(int(t)); pay.append(float(x["trifecta_payout"] or 0)); keep.append(x)
    if not keep:
        return dict(kind="live", date=np.array([], str), stadium=np.array([], int), P=np.zeros((0, 120), np.float32),
                    O=np.zeros((0, 120), np.float32), tri=np.zeros(0, int), payout=np.zeros(0))
    return dict(kind="live", date=np.array([str(x["race_date"])[:10] for x in keep]), stadium=np.array([int(x["stadium_code"]) for x in keep]),
                P=np.array(P, np.float32), O=np.array(O, np.float32), tri=np.array(tri), payout=np.array(pay))


# ---------------------------------------------------------------- シミュレーション
def simulate(ds: dict, prm: FocusedParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """各レースの (投資, 払戻, 的中)。買わないレースは投資0。"""
    n = len(ds["tri"])
    stake = np.zeros(n); pay = np.zeros(n); hit = np.zeros(n, bool)
    for i in range(n):
        f = select_focused(ds["P"][i], ds["O"][i], prm)
        if f.decision != "buy":
            continue
        stake[i] = sum(f.stakes)
        t = int(ds["tri"][i])
        if t in f.points:
            hit[i] = True
            pay[i] = ds["payout"][i] * f.stakes[f.points.index(t)] / 100
    return stake, pay, hit


def summarize(stake, pay, hit, sel=None) -> dict:
    m = stake > 0
    if sel is not None:
        m &= sel
    n = int(m.sum())
    if n == 0:
        return dict(n=0, hit=None, roi=None, ci=None, pnl=0, avg_stake=None, share=0.0)
    lo, hi = roi_bootstrap(stake[m], pay[m])
    tot = int(sel.sum()) if sel is not None else len(stake)
    return dict(n=n, hit=round(float(hit[m].mean()), 4), roi=round(float(pay[m].sum() / stake[m].sum()), 4),
                ci=[round(lo, 3), round(hi, 3)], pnl=int(pay[m].sum() - stake[m].sum()), avg_stake=int(stake[m].mean()),
                share=round(n / max(tot, 1), 3))


def _current_params() -> dict | None:
    try:
        row = pd.read_sql_query(text("SELECT extra FROM settings_versions ORDER BY id DESC LIMIT 1"), get_engine())
        if not len(row):
            return None
        ex = row.iloc[0]["extra"]
        ex = json.loads(ex) if isinstance(ex, str) else (ex or {})
        fc = ex.get("focused") or {}
        return {k: fc[k] for k in ("ev_min", "odds_hi", "s15_min", "max_points") if k in fc} or None
    except Exception as e:
        log.warning("current settings unavailable: %r", e)
        return None


def run_research(bt: dict | None = None, live: dict | None = None, current: dict | None = None, out: Path | None = None) -> dict:
    bt = bt if bt is not None else load_backtest()
    live = live if live is not None else load_live()
    current = current if current is not None else _current_params()
    variants = [dict(v) for v in GRID]
    cur_label = None
    if current:
        cur = {**GRID[0], **current}
        hit_ = [v for v in variants if all(abs(float(v[k]) - float(cur[k])) < 1e-9 for k in cur)]
        if hit_:
            cur_label = label_of(hit_[0])
        else:
            variants.append(cur)
            cur_label = label_of(cur)
    # 期間ごとの成績 + 月別（順送り用）
    datasets = [d for d in (bt, live) if d is not None and len(d["tri"])]
    rows = []
    monthly: dict[str, dict[str, tuple]] = {}   # label -> month -> (stake, pay, hit) arrays
    for v in variants:
        prm = replace(FocusedParams(), **v)
        lab = label_of(v)
        row = dict(label=lab, params=v, is_current=(lab == cur_label))
        for ds in datasets:
            stake, pay, hit = simulate(ds, prm)
            if ds["kind"] == "bt":
                expl = ds["date"] <= EXPLORE_END
                row["bt_explore"] = summarize(stake, pay, hit, expl)
                row["bt_confirm"] = summarize(stake, pay, hit, ~expl)
            else:
                row["live"] = summarize(stake, pay, hit)
            months = np.array([d[:7] for d in ds["date"]])
            for mo in np.unique(months):
                m = months == mo
                cur_m = monthly.setdefault(lab, {}).get(mo)
                arr = (stake[m], pay[m], hit[m])
                monthly[lab][mo] = arr if cur_m is None else tuple(np.concatenate([a, b]) for a, b in zip(cur_m, arr))
        rows.append(row)
    # 順送り検証: 各月、その月より前の累計で「回収率95%区間の下限」が最大（n>=MIN_ROLL_N）の候補をその月に当てる
    # （平均回収率で選ぶと、たまたま当たった月に引きずられて毎月候補が入れ替わる）
    all_months = sorted({mo for d in monthly.values() for mo in d})
    roll_stake, roll_pay, roll_hit, picks = [], [], [], []
    for i, mo in enumerate(all_months):
        prev = all_months[:i]
        best, best_lo, best_roi = None, -1.0, None
        for lab, d in monthly.items():
            st = np.concatenate([d[m][0] for m in prev if m in d]) if prev else np.zeros(0)
            py = np.concatenate([d[m][1] for m in prev if m in d]) if prev else np.zeros(0)
            k = st > 0
            if int(k.sum()) < MIN_ROLL_N:
                continue
            lo, _ = roi_bootstrap(st[k], py[k], n_boot=200)
            if lo > best_lo:
                best, best_lo, best_roi = lab, lo, float(py.sum() / st.sum())
        if best is None or mo not in monthly[best]:
            continue
        st, py, ht = monthly[best][mo]
        roll_stake.append(st); roll_pay.append(py); roll_hit.append(ht)
        s = summarize(st, py, ht)
        picks.append(dict(month=str(mo), picked=best, prev_roi=round(best_roi, 4), prev_ci_lo=round(float(best_lo), 3), n=s["n"], roi=s["roi"], pnl=s["pnl"]))
    rolling = summarize(np.concatenate(roll_stake), np.concatenate(roll_pay), np.concatenate(roll_hit)) if roll_stake else dict(n=0)
    # 候補の印: 現在設定より 確認期間と実運用の両方で回収率が高い（実運用が100R未満なら確認期間のみで判定し「仮」）
    cur_row = next((r for r in rows if r["is_current"]), None)
    for r in rows:
        r["flag"] = ""
        if cur_row is None or r is cur_row:
            continue
        bc, bl = r.get("bt_confirm", {}), r.get("live", {})
        cc, cl = cur_row.get("bt_confirm", {}), cur_row.get("live", {})
        better_bt = bc.get("roi") is not None and cc.get("roi") is not None and bc["roi"] > cc["roi"]
        if bl.get("n", 0) >= 100 and cl.get("n", 0) >= 100:
            if better_bt and bl["roi"] > cl["roi"]:
                r["flag"] = "候補"
        elif better_bt:
            r["flag"] = "仮候補"
    rows.sort(key=lambda r: -(r.get("bt_confirm", {}).get("roi") or 0))
    report = dict(generated_at=now_jst().isoformat(timespec="minutes"), model_bt=bt["model"] if bt else None,
                  n_bt=int(len(bt["tri"])) if bt else 0, n_live=int(len(live["tri"])) if live else 0,
                  explore_end=EXPLORE_END, current=cur_label, grid_size=len(GRID), rows=rows,
                  rolling=dict(summary=rolling, picks=picks),
                  note="候補は固定。順送り＝各月その月より前の最良候補を当てはめた成績。設定は自動では変えません。")
    out = out or OUT_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def load_report(path: Path | None = None) -> dict | None:
    p = path or OUT_FILE
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
