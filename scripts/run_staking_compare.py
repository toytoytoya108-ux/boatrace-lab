"""資金配分方式の比較（検証期間のみで決定 → 選んだ方式だけ封印テストで1回確認）。

買い目の選定は同一なので的中率は方式に依らず同じ。違いは回収率・損益・DD・的中時払戻の分布。
"""
from __future__ import annotations
import json, logging, sys, time
from datetime import date
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boatlab.backtest.dataset import build_race_dataset
from boatlab.backtest.metrics import summarize, max_drawdown
from boatlab.backtest.walkforward import ProbStore, evaluate
from boatlab.model.selection import SelectionParams
from boatlab.model.staking import StakingParams
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("staking")
OUT = Path("reports/backtest"); STORES = Path("data/probstores")

METHODS = {
    "uniform_200x15": StakingParams(method="uniform"),
    "payout_equal": StakingParams(method="payout_equal"),
    "prob_p1": StakingParams(method="prob", prob_power=1.0),
    "prob_p2": StakingParams(method="prob", prob_power=2.0),
    "group_100/400": StakingParams(method="group", main_stake=100, hole_stake=400),  # 100円単位で成立する唯一の非均等グループ配分
}

def completeness(d0, d1):
    parts = []
    for f in sorted(Path("data/features").glob("fs1_*.parquet")):
        a, b = f.stem.split("_")[1:3]
        if b < str(d0) or a > str(d1):
            continue
        parts.append(pd.read_parquet(f, columns=["race_id", "completeness"]).groupby("race_id")["completeness"].first())
    return pd.concat(parts) if parts else pd.Series(dtype=float)

def row(rec, name):
    s = summarize(rec, name)
    v = rec[rec["valid"] == True].sort_values(["race_date", "race_id"])
    out = {"method": name}
    for k in ("all", "buy"):
        sub = v if k == "all" else v[v["decision"] == "buy"]
        d = s[k]
        hits = sub[sub["hit"] == True]
        out.update({f"{k}_n": d["n"], f"{k}_hit": d["hit_rate"], f"{k}_roi": d["roi"], f"{k}_roi_lo": d["roi_ci"][0], f"{k}_roi_hi": d["roi_ci"][1],
                    f"{k}_pnl": d["pnl"], f"{k}_maxdd": max_drawdown(sub["pnl"].values) if len(sub) else 0,
                    f"{k}_pay_med": float(hits["payout_total"].median()) if len(hits) else np.nan,
                    f"{k}_pay_p10": float(hits["payout_total"].quantile(0.1)) if len(hits) else np.nan,
                    f"{k}_pay_lt_stake": float((hits["payout_total"] < hits["stake_total"]).mean()) if len(hits) else np.nan,
                    f"{k}_hole_roi": d.get("hole_roi", np.nan)})
    return out

def run(tag, store_name, d0, d1, methods):
    store = ProbStore.load(STORES / store_name)
    R = build_race_dataset(d0, d1)
    comp = completeness(d0, d1)
    sel = json.loads((OUT / "chosen_selection.json").read_text())
    prm = SelectionParams(hole_min_odds=sel["hole_min_odds"], beta=sel["beta"])
    rows = []
    for name, sp in methods.items():
        t = time.time()
        rec = evaluate(store, R, prm, X_completeness=comp, staking=sp)
        rows.append(row(rec, name))
        log.info("[%s] %s: all roi=%.4f buy roi=%.4f (%.0fs)", tag, name, rows[-1]["all_roi"], rows[-1]["buy_roi"], time.time() - t)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{tag}_staking_compare.csv", index=False)
    return df

def md(df, title):
    lines = [f"## {title}", "", "| 方式 | 対象 | N | 的中率 | 回収率 | 95%CI | 損益 | 最大DD | 的中時払戻 中央値 | 的中時払戻 下位10% | 的中でも赤字の割合 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        for k, lab in (("all", "全レース"), ("buy", "購入候補")):
            lines.append(f"| {r['method']} | {lab} | {int(r[k+'_n']):,} | {r[k+'_hit']*100:.1f}% | **{r[k+'_roi']*100:.1f}%** | {r[k+'_roi_lo']*100:.1f}–{r[k+'_roi_hi']*100:.1f}% | {r[k+'_pnl']:,.0f}円 | {r[k+'_maxdd']:,.0f}円 | {r[k+'_pay_med']:,.0f}円 | {r[k+'_pay_p10']:,.0f}円 | {r[k+'_pay_lt_stake']*100:.1f}% |")
    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    stages = sys.argv[1:] or ["valid"]
    parts = ["# 資金配分方式の比較（合計3,000円固定・100円単位・各点100円以上/900円以下）", "",
             "買い目（15点）の選定は全方式で同一。したがって的中率は同じで、差が出るのは回収率・損益・DD・的中時の払戻分布のみ。", ""]
    if "valid" in stages:
        df = run("valid", "valid_lgb_hNone.pkl", date(2024, 1, 1), date(2025, 6, 30), METHODS)
        parts.append(md(df, "検証期間 2024-01〜2025-06（推定オッズ、Model 1.0 設定）"))
    if "test" in stages:
        chosen = json.loads((OUT / "chosen_staking.json").read_text())
        m = {chosen["name"]: StakingParams.from_dict(chosen["params"]), "uniform_200x15": StakingParams()}
        df = run("test", "test_lgb.pkl", date(2025, 7, 1), date(2026, 8, 30), m)
        parts.append(md(df, "封印テスト 2025-07〜2026-08（選んだ方式のみ1回確認。2026年は実オッズ）"))
    (OUT / "staking_compare.md").write_text("\n".join(parts), encoding="utf-8")
    print((OUT / "staking_compare.md").read_text())
