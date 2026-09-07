"""Model 1.2（fs2: form + exh_trust）の本検証。四半期ウォークフォワード、期ごとにチェックポイント。

使い方（前面で何度も呼ぶ。10分で打ち切られても次回は続きから）:
  python scripts/run_model12_wf.py valid     # 2024-01〜2025-06（基準値 fs2_base と比較）
  python scripts/run_model12_wf.py test      # 2025-07〜2026-08（封印テスト。参考値：市場検証で一度使用済み）
完了すると reports/backtest/model12_<stage>.md を書く。
"""
from __future__ import annotations
import gc
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boatlab.backtest.dataset import build_entry_dataset, build_race_dataset
from boatlab.backtest.metrics import summarize
from boatlab.backtest.walkforward import ProbStore, WFConfig, evaluate, probs_quality, run_probs
from boatlab.features.build import CATEGORICAL_FEATURES, FEATURE_GROUPS, NUMERIC_FEATURES
from boatlab.model.selection import FocusedParams, SelectionParams, select_focused

OUT = Path("reports/backtest")
STORES = Path("data/probstores")
GROUPS = ("form", "exh_trust")
STAGES = {
    "valid": dict(vs="2024-01-01", ve="2025-06-30", r0=date(2023, 9, 1), base="fs2_base.pkl"),
    "test": dict(vs="2025-07-01", ve="2026-08-30", r0=date(2025, 3, 1), base="test_lgb.pkl"),
}


def load_X(groups):
    extra = [f for g in groups for f in FEATURE_GROUPS[g]]
    keep = list(dict.fromkeys(NUMERIC_FEATURES + extra + CATEGORICAL_FEATURES +
                              ["race_id", "race_date", "lane", "regno", "finish_pos", "y_win", "y_top2", "y_top3", "completeness"]))
    return build_entry_dataset(date(2018, 1, 1), date(2026, 8, 30), columns=keep)


def focused_roi(store: ProbStore, R: pd.DataFrame, prm: FocusedParams, use_real: bool) -> dict:
    """絞り込み型で買った場合の成績（推定/実オッズ）。"""
    Ri = R.set_index("race_id")
    stake = payout = n = hits = 0
    for per in store.periods:
        for i, rid in enumerate(per.test_ids):
            row = Ri.loc[rid]
            real = row["real_odds"] if use_real else None
            odds = real if isinstance(real, np.ndarray) and np.isfinite(real).sum() >= 100 else per.test_odds_est[i]
            f = select_focused(per.test_p[i], odds, prm)
            tri = int(per.test_tri[i])
            if f.decision != "buy" or tri < 0 or row["status"] == "cancelled":
                continue
            n += 1
            stake += sum(f.stakes)
            if tri in f.points:
                hits += 1
                payout += (row["trifecta_payout"] or 0) * f.stakes[f.points.index(tri)] / 100
    return dict(n=n, hit=hits / max(n, 1), roi=payout / max(stake, 1), pnl=payout - stake, avg_stake=stake / max(n, 1))


def main(stage: str):
    cfg_s = STAGES[stage]
    path = STORES / f"m12_{stage}.pkl"
    ckpt = STORES / f"m12_{stage}.ckpt.pkl"
    R = build_race_dataset(cfg_s["r0"], date.fromisoformat(cfg_s["ve"]))
    if not path.exists():
        X = load_X(GROUPS)
        cfg = WFConfig(period_start=cfg_s["vs"], period_end=cfg_s["ve"], freq="QS", holdout_months=3,
                       half_life_years=None, num_rounds=400, train_max_rows=1_200_000,
                       label=f"m12:{stage}", feature_groups=GROUPS)
        t = time.time()
        store = run_probs(X, R, cfg, checkpoint=ckpt)
        store.save(path)
        ckpt.unlink(missing_ok=True)
        print(f"run_probs done in {time.time() - t:.0f}s", flush=True)
        del X
        gc.collect()
    store = ProbStore.load(path)
    base = ProbStore.load(STORES / cfg_s["base"])
    comp = build_entry_dataset(date(2018, 1, 1), date(2026, 8, 30), columns=["race_id", "completeness"]).groupby("race_id")["completeness"].first()
    sel = json.loads((OUT / "chosen_selection.json").read_text())
    prm = SelectionParams(hole_min_odds=sel["hole_min_odds"], beta=sel["beta"])
    use_real = stage == "test"
    rows = []
    for label, st in (("Model 1.0", base), ("Model 1.2", store)):
        q = probs_quality(st)
        rec = evaluate(st, R, prm, use_real_odds=use_real, X_completeness=comp)
        s = summarize(rec, label)
        fr = focused_roi(st, R, FocusedParams(), use_real)
        rows.append(dict(model=label, logloss=round(q["trifecta_logloss"], 4), hit15=round(s["all"]["hit_rate"], 4), roi15=round(s["all"]["roi"], 4),
                         buy_n=s["buy"]["n"], buy_roi=round(s["buy"]["roi"], 4) if s["buy"]["n"] else None,
                         focused_n=fr["n"], focused_hit=round(fr["hit"], 4), focused_roi=round(fr["roi"], 4), focused_avg_stake=int(fr["avg_stake"])))
        del rec
        gc.collect()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"model12_{stage}.csv", index=False)
    md = [f"# Model 1.2（form + exh_trust）本検証 — {stage}（{cfg_s['vs']}〜{cfg_s['ve']}、四半期WF、{'実オッズ' if use_real else '推定オッズ'}）", "",
          "| モデル | 3連単log-loss | 15点的中率 | 15点回収率 | 購入候補N | 購入候補回収率 | 絞り込み型N | 絞り込み型的中率 | 絞り込み型回収率 | 平均投資 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['logloss']:.4f} | {r['hit15']*100:.1f}% | {r['roi15']*100:.1f}% | {r['buy_n']} | {('%.1f%%' % (r['buy_roi']*100)) if r['buy_roi'] is not None else '–'} | {r['focused_n']} | {r['focused_hit']*100:.1f}% | {r['focused_roi']*100:.1f}% | {r['focused_avg_stake']:,}円 |")
    (OUT / f"model12_{stage}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "valid")
