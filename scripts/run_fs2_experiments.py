"""fs2 特徴量グループの1グループずつ効果検証（検証期間 2024-01〜2025-06、四半期WF）。

判定: 3連単 log-loss がベースラインより 0.001 以上改善したグループを combo に採用。
出力: reports/backtest/fs2_experiments.{csv,md}（実行のたび追記保存）
"""
from __future__ import annotations
import gc, json, logging, sys, time
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boatlab.backtest.dataset import build_entry_dataset, build_race_dataset
from boatlab.backtest.walkforward import ProbStore, WFConfig, evaluate, probs_quality, run_probs
from boatlab.features.build import ALL_GROUP_FEATURES, CATEGORICAL_FEATURES, FEATURE_GROUPS, NUMERIC_FEATURES
from boatlab.model.selection import SelectionParams
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fs2")
OUT = Path("reports/backtest"); STORES = Path("data/probstores")
D0, END = date(2018, 1, 1), date(2026, 8, 30)
VS, VE = "2024-01-01", "2025-06-30"

def load_data():
    keep = list(dict.fromkeys(NUMERIC_FEATURES + ALL_GROUP_FEATURES + CATEGORICAL_FEATURES +
                              ["race_id", "race_date", "lane", "regno", "finish_pos",
                               "y_win", "y_top2", "y_top3", "completeness"]))
    X = build_entry_dataset(D0, END, columns=keep)
    # run_probs は period_start の holdout_months 前からの結果を参照する
    R = build_race_dataset(date(2023, 9, 1), date(2025, 6, 30))
    log.info("X=%s R=%s", X.shape, R.shape)
    return X, R

def one_run(X, R, label, groups):
    path = STORES / f"fs2_{label}.pkl"
    if path.exists():
        store = ProbStore.load(path)
    else:
        cfg = WFConfig(period_start=VS, period_end=VE, freq="QS", holdout_months=3,
                       half_life_years=None, num_rounds=400, train_max_rows=1_200_000,
                       label=f"fs2:{label}", feature_groups=tuple(groups))
        t = time.time()
        store = run_probs(X, R, cfg)
        store.save(path)
        log.info("[%s] run_probs done in %.0fs", label, time.time() - t)
    q = probs_quality(store)
    comp = X.groupby("race_id")["completeness"].first()
    sel = json.loads((OUT / "chosen_selection.json").read_text())
    rec = evaluate(store, R, SelectionParams(hole_min_odds=sel["hole_min_odds"], beta=sel["beta"]),
                   use_real_odds=False, X_completeness=comp)
    v = rec[rec["valid"] == True]
    hit = float(v["hit"].mean()); roi = float(v["payout_total"].sum() / max(v["stake_total"].sum(), 1))
    row = dict(label=label, groups="+".join(groups) if groups else "(base)", n=q["n"],
               logloss=round(q["trifecta_logloss"], 4), hit=round(hit, 4), roi=round(roi, 4))
    del rec, v
    gc.collect()
    # 追記保存
    csv = OUT / "fs2_experiments.csv"
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    df = pd.concat([df[df["label"] != label] if len(df) else df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv, index=False)
    log.info("RESULT %s", row)
    return row

if __name__ == "__main__":
    X, R = load_data()
    base = one_run(X, R, "base", [])
    rows = [base]
    order = ["entry", "st", "kimarite", "weather", "form", "exh_trust", "parts"]
    for g in order:
        rows.append(one_run(X, R, g, [g]))
    good = [r["label"] for r in rows[1:] if r["logloss"] <= base["logloss"] - 0.001]
    log.info("improving groups: %s", good)
    if good:
        rows.append(one_run(X, R, "combo", good))
        # combo から1つずつ抜く後退検証（グループが2つ以上のとき）
        if len(good) > 2:
            for g in good:
                rows.append(one_run(X, R, f"combo-{g}", [x for x in good if x != g]))
    md = ["# fs2 特徴量グループの効果検証（検証期間 2024-01〜2025-06・四半期WF・推定オッズ）", "",
          "| 実験 | 追加グループ | log-loss | Δ vs base | 15点的中率 | 回収率 |", "|---|---|---|---|---|---|"]
    for r in pd.read_csv(OUT / "fs2_experiments.csv").to_dict("records"):
        md.append(f"| {r['label']} | {r['groups']} | {r['logloss']:.4f} | {r['logloss']-base['logloss']:+.4f} | {r['hit']*100:.1f}% | {r['roi']*100:.1f}% |")
    (OUT / "fs2_experiments.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("ALL DONE")
