"""バックテスト・キャンペーン（docs/05）。

stage 1: データセット構築（2018-01-01〜END）
stage 2: 検証期間（2024-01〜2025-06, 四半期）でモデル比較：m0 / m0b / lgb(減衰 h ∈ {None,3,2,1})
stage 3: 最良モデルの ProbStore に対して選定パラメータ探索（穴オッズ × β）
stage 4: 封印テスト（2025-07〜END, 月次）を 1 回だけ実行（選んだ設定で）
stage 5: 市場オッズ推定の精度検証（2026 実オッズ）
出力: reports/backtest/*.md, *.json, *.parquet
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boatlab.backtest.dataset import build_entry_dataset, build_race_dataset  # noqa: E402
from boatlab.backtest.metrics import monthly, recent, set_calibration_table, summarize  # noqa: E402
from boatlab.backtest.walkforward import ProbStore, WFConfig, evaluate, probs_quality, run_probs  # noqa: E402
from boatlab.model.market import evaluate_against_real  # noqa: E402
from boatlab.model.selection import SelectionParams  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("campaign")

D0 = date(2018, 1, 1)
END = date(2026, 8, 30)
VALID = ("2024-01-01", "2025-06-30")
TEST = ("2025-07-01", "2026-08-30")
OUT = Path("reports/backtest")
OUT.mkdir(parents=True, exist_ok=True)
STORES = Path("data/probstores")
STORES.mkdir(parents=True, exist_ok=True)


def _fmt(s: dict) -> dict:
    def f(v):
        if isinstance(v, float):
            return round(v, 4)
        if isinstance(v, tuple):
            return [round(x, 4) for x in v]
        return v
    return {k: ({kk: f(vv) for kk, vv in v.items()} if isinstance(v, dict) else f(v)) for k, v in s.items()}


def stage_dataset():
    t = time.time()
    from boatlab.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES
    keep = list(dict.fromkeys(NUMERIC_FEATURES + CATEGORICAL_FEATURES +
                              ["race_id", "race_date", "lane", "regno", "finish_pos",
                               "y_win", "y_top2", "y_top3", "completeness"]))
    X = build_entry_dataset(D0, END, columns=keep)
    R = build_race_dataset(D0, END)
    log.info("dataset X=%s R=%s in %.0fs", X.shape, R.shape, time.time() - t)
    return X, R


def stage_model_compare(X, R, comp):
    results = {}
    configs = [
        WFConfig(*VALID, freq="QS", refit_every=1, model="m0", label="m0", train_max_rows=480_000),
        WFConfig(*VALID, freq="QS", refit_every=1, model="m0b", label="m0b", train_max_rows=480_000),
    ]
    for h in (None, 3.0, 2.0, 1.0):
        configs.append(WFConfig(*VALID, freq="QS", refit_every=1, model="lgb", half_life_years=h, num_rounds=250,
                                train_max_rows=480_000, label=f"lgb_h{h}"))
    for cfg in configs:
        path = STORES / f"valid_{cfg.label}.pkl"
        if path.exists():
            store = ProbStore.load(path)
        else:
            t = time.time()
            store = run_probs(X, R, cfg)
            store.save(path)
            log.info("%s probs in %.0fs", cfg.label, time.time() - t)
        q = probs_quality(store)
        rec = evaluate(store, R, SelectionParams(), X_completeness=comp)
        s = summarize(rec, cfg.label)
        results[cfg.label] = {"quality": q, "all": _fmt(s["all"]), "buy": _fmt(s["buy"]), "buy_rate": s["buy_rate"],
                              "periods": [{"period": p.period, "lam": p.lam, "market": p.market_fit} for p in store.periods]}
        log.info("%s: %s hit=%.3f roi=%.3f", cfg.label, q, s["all"]["hit_rate"], s["all"]["roi"])
    (OUT / "valid_model_compare.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str))
    rows = [{"model": k, "trifecta_logloss": v["quality"]["trifecta_logloss"], "hit_rate": v["all"]["hit_rate"],
             "roi": v["all"]["roi"], "main_roi": v["all"].get("main_roi"), "hole_roi": v["all"].get("hole_roi"),
             "max_dd": v["all"]["max_drawdown"], "buy_rate": v["buy_rate"]} for k, v in results.items()]
    df = pd.DataFrame(rows).sort_values("trifecta_logloss")
    (OUT / "valid_model_compare.md").write_text("# 検証期間 モデル比較（2024-01〜2025-06, 推定オッズ）\n\n" + df.to_markdown(index=False) + "\n")
    return df


def stage_selection_sweep(R, comp, best_label: str):
    store = ProbStore.load(STORES / f"valid_{best_label}.pkl")
    rows = []
    recs = {}
    for h in (10, 15, 20, 25, 30, 40, 50):
        for beta in (0.0, 0.3, 0.5):
            prm = SelectionParams(hole_min_odds=h, beta=beta)
            rec = evaluate(store, R, prm, X_completeness=comp)
            s = summarize(rec)
            a = s["all"]
            rows.append({"hole_min_odds": h, "beta": beta, "n": a["n"], "hit_rate": a["hit_rate"], "roi": a["roi"],
                         "roi_ci_lo": a["roi_ci"][0], "roi_ci_hi": a["roi_ci"][1], "main_roi": a.get("main_roi"),
                         "hole_roi": a.get("hole_roi"), "hole_hits": a["hole_hits"], "avg_payout_on_hit": a["avg_payout_on_hit"],
                         "max_losing_streak": a["max_losing_streak"], "max_dd": a["max_drawdown"],
                         "avg_er": a["avg_expected_return"], "relaxed_rate": float(rec["hole_relaxed"].mean()),
                         "buy_n": s["buy"]["n"], "buy_hit_rate": s["buy"]["hit_rate"], "buy_roi": s["buy"]["roi"]})
            recs[(h, beta)] = rec
            log.info("sweep h=%s beta=%s hit=%.3f roi=%.3f hole_roi=%.3f buy_n=%d", h, beta, a["hit_rate"], a["roi"], a.get("hole_roi", 0), s["buy"]["n"])
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "valid_selection_sweep.csv", index=False)
    (OUT / "valid_selection_sweep.md").write_text("# 検証期間 穴オッズ×β 探索（推定オッズ）\n\n" + df.round(4).to_markdown(index=False) + "\n")
    # 代替ゲート案の参考表示
    base = recs[(20, 0.3)]
    alt = []
    for cmin in (0.6, 0.65, 0.7, 0.75):
        for emin in (0.8, 0.9, 1.0, 1.1):
            v = base[(base["valid"] == True) & (base["confidence"] >= cmin) & (base["expected_return"] >= emin)]  # noqa: E712
            if len(v):
                alt.append({"confidence_min": cmin, "er_min": emin, "n": len(v), "rate": len(v) / (base["valid"] == True).sum(),  # noqa: E712
                            "hit_rate": v["hit"].mean(), "roi": v["payout_total"].sum() / max(v["stake_total"].sum(), 1)})
    pd.DataFrame(alt).round(4).to_csv(OUT / "valid_gate_alternatives.csv", index=False)
    return df


def stage_test(X, R, comp, half_life, hole_min_odds, beta):
    cfg = WFConfig(*TEST, freq="MS", refit_every=3, model="lgb", half_life_years=half_life, num_rounds=400, train_max_rows=1_200_000, label="test_lgb")
    path = STORES / "test_lgb.pkl"
    if path.exists():
        store = ProbStore.load(path)
    else:
        t = time.time()
        store = run_probs(X, R, cfg)
        store.save(path)
        log.info("test probs in %.0fs", time.time() - t)
    prm = SelectionParams(hole_min_odds=hole_min_odds, beta=beta)
    rec = evaluate(store, R, prm, X_completeness=comp)
    rec.to_parquet(OUT / "test_records.parquet", index=False)
    s = summarize(rec, "test")
    out = {"quality": probs_quality(store), "summary": _fmt({k: v for k, v in s.items() if isinstance(v, dict)}),
           "buy_rate": s["buy_rate"], "skip_would_hit": s["skip_would_hit"],
           "recent100_buy": recent(rec, 100), "recent100_all": recent(rec, 100, decision="skip"),
           "params": {"half_life": half_life, "hole_min_odds": hole_min_odds, "beta": beta}}
    # 実オッズ期間（2026）と推定オッズ期間を分けて表示
    for src in ("real", "estimated"):
        sub = rec[rec["odds_source"] == src]
        if len(sub):
            out[f"by_odds_source_{src}"] = _fmt({k: v for k, v in summarize(sub).items() if isinstance(v, dict)})
    (OUT / "test_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    md = ["# 封印テスト（2025-07〜2026-08）結果 — 1回のみ評価\n",
          f"パラメータ: {out['params']}\n", f"3連単 log-loss: {out['quality']}\n",
          "## 全レース（仮想購入）\n", pd.DataFrame([out["summary"]["all"]]).T.to_markdown(),
          "\n## 購入候補のみ\n", pd.DataFrame([out["summary"]["buy"]]).T.to_markdown(),
          "\n## 月別（全レース仮想）\n", monthly(rec, decision=None).round(4).to_markdown(),
          "\n## 信頼度帯別の実セット的中率（校正検証）\n", set_calibration_table(rec).round(4).to_markdown(index=False)]
    for src in ("real", "estimated"):
        if f"by_odds_source_{src}" in out:
            md += [f"\n## オッズ種別 = {src}\n", pd.DataFrame([out[f"by_odds_source_{src}"]["all"]]).T.to_markdown()]
    (OUT / "test_summary.md").write_text("\n".join(md) + "\n")
    return rec, store


def stage_market_eval(store: ProbStore, R):
    Rr = R.set_index("race_id")
    est, real = [], []
    for per in store.periods:
        for i, rid in enumerate(per.test_ids):
            ro = Rr.loc[rid, "real_odds"]
            if isinstance(ro, np.ndarray):
                est.append(per.test_odds_est[i]); real.append(ro)
    if not est:
        return {}
    ev = evaluate_against_real(np.concatenate(est), np.concatenate(real))
    (OUT / "market_odds_eval.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1, default=str))
    return ev


if __name__ == "__main__":
    stages = sys.argv[1:] or ["dataset", "compare", "sweep", "test", "market"]
    X, R = stage_dataset()
    comp = X.groupby("race_id")["completeness"].first()
    best = "lgb_hNone"
    if "compare" in stages:
        df = stage_model_compare(X, R, comp)
        lgb_rows = df[df["model"].str.startswith("lgb")]
        best = lgb_rows.sort_values("trifecta_logloss").iloc[0]["model"]
        log.info("best model by logloss: %s", best)
    if "sweep" in stages:
        stage_selection_sweep(R, comp, best)
    if "test" in stages:
        hl = None if best.endswith("None") else float(best.split("_h")[1])
        sel = json.loads((OUT / "chosen_selection.json").read_text()) if (OUT / "chosen_selection.json").exists() else {"hole_min_odds": 20, "beta": 0.3}
        rec, store = stage_test(X, R, comp, hl, sel["hole_min_odds"], sel["beta"])
        if "market" in stages:
            log.info("market eval: %s", stage_market_eval(store, R))
