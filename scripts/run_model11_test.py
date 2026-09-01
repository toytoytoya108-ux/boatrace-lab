"""Model 1.1 候補（3-seed アンサンブル・1.2M行・400round・減衰なし）の封印テスト。1回のみ。

比較対象: Model 1.0 の封印テスト（reports/backtest/test_summary.json）
  3連単 log-loss 3.7983 / 全レース 的中率 55.9% 回収率 78.9% / 購入候補 1,903R 72.5% 76.6%
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_backtest_campaign import OUT, STORES, TEST, _fmt, stage_dataset  # noqa: E402

from boatlab.backtest.metrics import monthly, set_calibration_table, summarize  # noqa: E402
from boatlab.backtest.walkforward import ProbStore, WFConfig, evaluate, probs_quality, run_probs  # noqa: E402
from boatlab.model.selection import SelectionParams  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("m11test")
PRM = SelectionParams(hole_min_odds=20, beta=0.3)
SEEDS = (7, 17, 27)

if __name__ == "__main__":
    X, R = stage_dataset()
    comp = X.groupby("race_id")["completeness"].first()
    stores = []
    for sd in SEEDS:
        cfg = WFConfig(*TEST, freq="MS", refit_every=3, model="lgb", half_life_years=None, num_rounds=400,
                       train_max_rows=1_200_000, label=f"test_m11_s{sd}",
                       lgb_params={"seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd})
        path = STORES / f"{cfg.label}.pkl"
        if path.exists():
            stores.append(ProbStore.load(path))
        else:
            t = time.time()
            st = run_probs(X, R, cfg)
            st.save(path)
            stores.append(st)
            log.info("%s probs in %.0fs", cfg.label, time.time() - t)
    base = stores[0]
    for k, per in enumerate(base.periods):
        for attr in ("hold_p", "test_p"):
            ps = [getattr(st.periods[k], attr).astype(np.float64) for st in stores]
            g = np.exp(np.mean([np.log(np.clip(p, 1e-12, None)) for p in ps], axis=0))
            setattr(per, attr, (g / g.sum(axis=1, keepdims=True)).astype(np.float32))
    base.save(STORES / "test_m11_ens3.pkl")
    rec = evaluate(base, R, PRM, X_completeness=comp)
    rec.to_parquet(OUT / "test_m11_records.parquet", index=False)
    s = summarize(rec, "test_m11")
    out = {"quality": probs_quality(base), "summary": _fmt({k: v for k, v in s.items() if isinstance(v, dict)}),
           "buy_rate": s["buy_rate"], "params": {"seeds": SEEDS, "train_max_rows": 1_200_000, "num_rounds": 400,
                                                 "half_life": None, "hole_min_odds": 20, "beta": 0.3}}
    for src in ("real", "estimated"):
        sub = rec[rec["odds_source"] == src]
        if len(sub):
            out[f"by_odds_source_{src}"] = _fmt({k: v for k, v in summarize(sub).items() if isinstance(v, dict)})
    (OUT / "test_m11_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    m10 = json.loads((OUT / "test_summary.json").read_text())
    cmp = pd.DataFrame([
        {"model": "1.0", "logloss": m10["quality"]["trifecta_logloss"], **{k: m10["summary"]["all"][k] for k in ("hit_rate", "roi", "max_losing_streak", "max_drawdown")},
         "buy_n": m10["summary"]["buy"]["n"], "buy_hit": m10["summary"]["buy"]["hit_rate"], "buy_roi": m10["summary"]["buy"]["roi"]},
        {"model": "1.1(ens3)", "logloss": out["quality"]["trifecta_logloss"], **{k: out["summary"]["all"][k] for k in ("hit_rate", "roi", "max_losing_streak", "max_drawdown")},
         "buy_n": out["summary"]["buy"]["n"], "buy_hit": out["summary"]["buy"]["hit_rate"], "buy_roi": out["summary"]["buy"]["roi"]},
    ])
    md = ["# Model 1.1 封印テスト（2025-07〜2026-08）— 1回のみ評価\n", cmp.round(4).to_markdown(index=False),
          "\n\n## 月別（全レース仮想）\n", monthly(rec, decision=None).round(4).to_markdown(),
          "\n\n## 信頼度帯別の実セット的中率\n", set_calibration_table(rec).round(4).to_markdown(index=False)]
    (OUT / "test_m11_summary.md").write_text("\n".join(md) + "\n")
    log.info("ALL DONE\n%s", cmp.round(4).to_string(index=False))
