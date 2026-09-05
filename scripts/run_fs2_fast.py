"""fs2 グループ効果の高速アブレーション（単一分割・前面実行前提）。

重い四半期ウォークフォワード（1グループ40分）は待機停止環境で完走しないため、
1回の train/holdout/test 分割で log-loss を比較する（グループ間の相対順位が目的）。
  train : race_date < 2023-10-01
  holdout(校正): 2023-10-01 〜 2023-12-31
  test  : 2024-01-01 〜 2025-06-30
判定: 3連単 log-loss が base より 0.001 以上小さいグループを「効いた」とする。
1グループずつ引数で実行し、結果を reports/backtest/fs2_fast.csv に追記。
"""
from __future__ import annotations
import gc
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boatlab.backtest.dataset import build_entry_dataset, build_race_dataset
from boatlab.features.build import CATEGORICAL_FEATURES, FEATURE_GROUPS, NUMERIC_FEATURES
from boatlab.model.calibration import ComboCalibrator
from boatlab.model.strength import StrengthModel
from boatlab.model.trifecta import fit_lambdas, race_matrix, trifecta_probs

OUT = Path("reports/backtest")
OUT.mkdir(parents=True, exist_ok=True)
TR, HS, HE, TS, TE = "2023-10-01", "2023-10-01", "2023-12-31", "2024-01-01", "2025-06-30"
ROUNDS, MAXROWS = 250, 700_000


def _mats(model, x):
    pred = model.predict(x)
    pw, ids = race_matrix(pred, x, "p_win")
    p2, _ = race_matrix(pred, x, "p_top2")
    p3, _ = race_matrix(pred, x, "p_top3")
    return (pw, p2, p3), ids


def run(label, groups):
    extra = [f for g in groups for f in FEATURE_GROUPS[g]]
    keep = list(dict.fromkeys(NUMERIC_FEATURES + extra + CATEGORICAL_FEATURES +
               ["race_id", "race_date", "lane", "regno", "finish_pos", "y_win", "y_top2", "y_top3"]))
    X = build_entry_dataset(date(2018, 1, 1), date(2026, 8, 30), columns=keep)
    R = build_race_dataset(date(2023, 10, 1), date(2025, 6, 30)).set_index("race_id")
    t0 = time.time()
    tr = X[(X["race_date"] < TR) & X["finish_pos"].notna()]
    if len(tr) > MAXROWS:
        rids = tr["race_id"].unique()
        keepids = set(pd.Series(rids).sample(n=MAXROWS // 6, random_state=0))
        tr = tr[tr["race_id"].isin(keepids)]
    hold = X[(X["race_date"] >= HS) & (X["race_date"] <= HE) & X["finish_pos"].notna()]
    test = X[(X["race_date"] >= TS) & (X["race_date"] <= TE)]
    model = StrengthModel(num_rounds=ROUNDS, half_life_years=None,
                          feature_names=NUMERIC_FEATURES + extra + CATEGORICAL_FEATURES).fit(tr, asof=pd.Timestamp(TR))
    mats_h, ids_h = _mats(model, hold)
    tri_h = R.loc[ids_h, "tri_idx"].values.astype(int)
    lam = fit_lambdas(*mats_h, tri_h)[:2]
    cal = ComboCalibrator().fit(trifecta_probs(*mats_h, *lam), tri_h)
    mats_t, ids_t = _mats(model, test)
    p_t = cal.transform(trifecta_probs(*mats_t, *lam))
    tri_t = R.loc[ids_t, "tri_idx"].values.astype(int)
    idx = np.arange(len(tri_t))
    ok = tri_t >= 0
    ll = float(-np.log(np.clip(p_t[idx[ok], tri_t[ok]], 1e-12, None)).mean())
    top15 = np.argsort(-p_t, axis=1)[:, :15]
    hit = float(np.mean([tri_t[i] in top15[i] for i in range(len(tri_t)) if tri_t[i] >= 0]))
    row = dict(label=label, groups="+".join(groups) if groups else "(base)", n=int(ok.sum()),
               logloss=round(ll, 4), top15_hit=round(hit, 4), secs=int(time.time() - t0))
    csv = OUT / "fs2_fast.csv"
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    df = pd.concat([df[df["label"] != label] if len(df) else df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(csv, index=False)
    print("RESULT", row, flush=True)
    del X, R, tr, hold, test, model, p_t
    gc.collect()
    return row


if __name__ == "__main__":
    label = sys.argv[1]
    groups = [] if label == "base" else sys.argv[2].split(",")
    run(label, groups)
