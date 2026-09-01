"""Model 1.1 実験（検証期間 2024-01〜2025-06 のみ。封印テストは勝者だけが後で1回）。

比較対象（現行）: lgb_hNone = log-loss 3.8204 / hit .579 / roi .803（480k行・250round・既定パラメータ）

実験:
  tuned1  : lr0.03 leaves63 min_data100 rounds700（480k行）… 容量を増やす
  tuned2  : lr0.05 leaves63 min_data100 rounds400（480k行）… 中間
  data12  : 既定パラメータ rounds400（1.2M行）        … データ量の効果
  ens3    : 最良設定を seed 3 本の幾何平均（艇別確率を平均）  … 分散低減
選定規則: log-loss 主・僅差(±0.002)なら単純な方。採用判断はこの後の shadow/テストで。
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
from run_backtest_campaign import OUT, STORES, VALID, stage_dataset  # noqa: E402

from boatlab.backtest.metrics import summarize  # noqa: E402
from boatlab.backtest.walkforward import ProbStore, WFConfig, evaluate, probs_quality, run_probs  # noqa: E402
from boatlab.model.selection import SelectionParams  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("m11")

PRM = SelectionParams(hole_min_odds=20, beta=0.3)


def run_one(X, R, comp, cfg: WFConfig) -> dict:
    path = STORES / f"valid_{cfg.label}.pkl"
    if path.exists():
        store = ProbStore.load(path)
    else:
        t = time.time()
        store = run_probs(X, R, cfg)
        store.save(path)
        log.info("%s probs in %.0fs", cfg.label, time.time() - t)
    q = probs_quality(store)
    s = summarize(evaluate(store, R, PRM, X_completeness=comp), cfg.label)
    out = {"label": cfg.label, "logloss": q["trifecta_logloss"], "hit": s["all"]["hit_rate"],
           "roi": s["all"]["roi"], "buy_n": s["buy"]["n"], "buy_hit": s["buy"]["hit_rate"], "buy_roi": s["buy"]["roi"]}
    log.info("RESULT %s", out)
    return out


def run_ensemble(X, R, comp, base_kwargs: dict, seeds=(7, 17, 27), label="ens3") -> dict:
    """seed 違いの3モデルの艇別確率を平均してから PL に入れる簡易アンサンブル。"""
    stores = []
    for sd in seeds:
        cfg = WFConfig(*VALID, freq="QS", refit_every=1, model="lgb", label=f"{label}_s{sd}",
                       lgb_params={**base_kwargs.get("lgb_params", {}), "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                       **{k: v for k, v in base_kwargs.items() if k != "lgb_params"})
        path = STORES / f"valid_{cfg.label}.pkl"
        if path.exists():
            stores.append(ProbStore.load(path))
        else:
            t = time.time()
            st = run_probs(X, R, cfg)
            st.save(path)
            stores.append(st)
            log.info("%s probs in %.0fs", cfg.label, time.time() - t)
    # 120通り確率を幾何平均して正規化（期間・レース順は同一）
    base = stores[0]
    for k, per in enumerate(base.periods):
        for attr in ("hold_p", "test_p"):
            ps = [getattr(st.periods[k], attr).astype(np.float64) for st in stores]
            g = np.exp(np.mean([np.log(np.clip(p, 1e-12, None)) for p in ps], axis=0))
            g = g / g.sum(axis=1, keepdims=True)
            setattr(per, attr, g.astype(np.float32))
    q = probs_quality(base)
    s = summarize(evaluate(base, R, PRM, X_completeness=comp), label)
    out = {"label": label, "logloss": q["trifecta_logloss"], "hit": s["all"]["hit_rate"], "roi": s["all"]["roi"],
           "buy_n": s["buy"]["n"], "buy_hit": s["buy"]["hit_rate"], "buy_roi": s["buy"]["roi"]}
    log.info("RESULT %s", out)
    return out


if __name__ == "__main__":
    X, R = stage_dataset()
    comp = X.groupby("race_id")["completeness"].first()
    results = [{"label": "baseline lgb_hNone(既存)", "logloss": 3.8204, "hit": 0.5792, "roi": 0.8028}]
    results.append(run_one(X, R, comp, WFConfig(*VALID, freq="QS", refit_every=1, model="lgb", num_rounds=700,
                                                train_max_rows=480_000, label="m11_tuned1",
                                                lgb_params={"learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 100})))
    results.append(run_one(X, R, comp, WFConfig(*VALID, freq="QS", refit_every=1, model="lgb", num_rounds=400,
                                                train_max_rows=480_000, label="m11_tuned2",
                                                lgb_params={"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100})))
    results.append(run_one(X, R, comp, WFConfig(*VALID, freq="QS", refit_every=1, model="lgb", num_rounds=400,
                                                train_max_rows=1_200_000, label="m11_data12")))
    # 上の3つで最良の設定を seed アンサンブル
    best = min(results[1:], key=lambda r: r["logloss"])
    kw = {"m11_tuned1": dict(num_rounds=700, train_max_rows=480_000, lgb_params={"learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 100}),
          "m11_tuned2": dict(num_rounds=400, train_max_rows=480_000, lgb_params={"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100}),
          "m11_data12": dict(num_rounds=400, train_max_rows=1_200_000)}[best["label"]]
    results.append(run_ensemble(X, R, comp, kw, label=f"m11_ens3_{best['label'][-6:]}"))
    df = pd.DataFrame(results)
    df.to_csv(OUT / "m11_experiments.csv", index=False)
    (OUT / "m11_experiments.md").write_text("# Model 1.1 実験（検証期間のみ）\n\n" + df.round(4).to_markdown(index=False) + "\n")
    log.info("ALL DONE\n%s", df.round(4).to_string(index=False))
