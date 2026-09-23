"""展示評価（◎×）の較正: 貯まった評価と結果から「評価の効き（beta）」を推定する（2026-09-23）。

設定は自動では変えない。ここで出す推定値は画面に「提案」として出すだけで、採用するかはユーザーが決める。

推定の考え方: 評価なしの確定予想（role=katai_t）の120通り確率 p を土台に、
  p'(beta) = 正規化(p × exp(beta·(1.0·r_1着 + 0.5·r_2着 + 0.25·r_3着)))
として、実際の3連単の対数尤度 Σ log p'(beta)[実際の目] を最大にする beta を格子で探す。
beta=0（評価を無視）との尤度差が 1.92（χ²(1) の 95%）を超えなければ「まだ言えない」。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sqlalchemy import text

from boatlab.model.modes import adjust_probs_for_ratings
from boatlab.model.trifecta import PERM_LABELS, combo_index
from boatlab.store.db import get_engine

_PL = list(PERM_LABELS)
GRID = np.round(np.arange(0.0, 1.51, 0.05), 2)


def load_rated_pairs(model_version: str | None = None, base_role: str = "katai_t") -> pd.DataFrame:
    """評価あり版（role=<base>R）と評価なし版（role=<base>）が両方あり、結果も出ているレース。"""
    q = text(f"""
        SELECT b.race_id, r.race_date, r.stadium_code, r.race_no, b.probs AS probs, a.flags AS flags_r,
               res.trifecta, res.trifecta_payout,
               sb.valid AS valid_b, sb.hit AS hit_b, sb.stake_total AS stake_b, sb.payout_total AS payout_b,
               sa.valid AS valid_r, sa.hit AS hit_r, sa.stake_total AS stake_r, sa.payout_total AS payout_r,
               b.decision AS decision_b, a.decision AS decision_r
        FROM predictions a
        JOIN predictions b ON b.race_id = a.race_id AND b.model_version = a.model_version AND b.stage = a.stage
                          AND b.role = :base
        JOIN races r ON r.id = a.race_id
        LEFT JOIN results res ON res.race_id = a.race_id
        LEFT JOIN scoring sa ON sa.prediction_id = a.id
        LEFT JOIN scoring sb ON sb.prediction_id = b.id
        WHERE a.role = :rated AND a.stage = 'final' AND (:mv IS NULL OR a.model_version = :mv)
        ORDER BY r.race_date, r.closed_at""")
    with get_engine().connect() as c:
        df = pd.read_sql(q, c, params={"base": base_role, "rated": base_role + "R", "mv": model_version})
    for col in ("probs", "flags_r"):
        if len(df) and isinstance(df[col].iloc[0], str):
            df[col] = df[col].map(json.loads)
    return df


def estimate_beta(df: pd.DataFrame) -> dict:
    """結果の出た評価ありレースから beta を推定する。"""
    rows = df[df["trifecta"].notna()] if len(df) else df
    n = len(rows)
    out = {"n": int(n), "beta_hat": None, "ll_gain": None, "ci": None, "verdict": "few",
           "good": None, "bad": None, "grid": None}
    if n == 0:
        return out
    P = np.array([[float(p[k]) for k in _PL] for p in rows["probs"]])
    tri = np.array([combo_index(t) for t in rows["trifecta"]])
    R = [ {int(k): int(v) for k, v in (f.get("ratings") or {}).items()} for f in rows["flags_r"] ]
    ll = []
    for b in GRID:
        v = 0.0
        for i in range(n):
            v += float(np.log(max(adjust_probs_for_ratings(P[i], R[i], float(b))[tri[i]], 1e-12)))
        ll.append(v)
    ll = np.array(ll)
    j = int(np.argmax(ll))
    gain = float(ll[j] - ll[0])
    inside = GRID[ll >= ll[j] - 1.92]
    out.update(beta_hat=float(GRID[j]), ll_gain=gain, ci=[float(inside.min()), float(inside.max())],
               grid=[{"beta": float(b), "ll": float(v - ll[0])} for b, v in zip(GRID, ll)])
    # ◎の艇・×の艇の「モデルの1着確率」と「実際の1着率」
    A = np.array([[int(x[0]) - 1 for x in PERM_LABELS]])  # 1着艇 (0-based)
    a_idx = np.array([int(PERM_LABELS[t][0]) - 1 for t in tri])
    for name, val in (("good", 1), ("bad", -1)):
        pm, hit = [], []
        for i in range(n):
            for lane, rr in R[i].items():
                if rr == val:
                    pw = float(P[i][A[0] == lane - 1].sum())
                    pm.append(pw); hit.append(1.0 if a_idx[i] == lane - 1 else 0.0)
        if pm:
            out[name] = {"n": len(pm), "model_win": float(np.mean(pm)), "actual_win": float(np.mean(hit))}
    if n < 100:
        out["verdict"] = "few"            # 100レース未満は推定値を出すだけ
    elif gain < 1.92:
        out["verdict"] = "no_evidence"    # beta=0 と区別できない
    else:
        out["verdict"] = "evidence"
    return out


def paired_summary(df: pd.DataFrame) -> dict:
    """同じレースの 評価なし／評価あり を対で集計（両方とも有効採点のレースだけ）。"""
    if not len(df):
        return {"n": 0}
    d = df[(df["valid_b"] == 1) & (df["valid_r"] == 1)]
    def agg(prefix):
        # 見送り行も仮想採点されているので、同じレースを「仮に買った」として対で比べる
        st = int(d[f"stake_{prefix}"].fillna(0).sum()); pay = int(d[f"payout_{prefix}"].fillna(0).sum())
        return {"stake": st, "payout": pay, "pnl": pay - st, "roi": (pay / st if st else None),
                "hits": int(d[f"hit_{prefix}"].fillna(0).sum()), "buys": int((d[f"decision_{prefix}"] == "buy").sum())}
    same = int(((d["hit_b"].fillna(0) == d["hit_r"].fillna(0))).sum())
    return {"n": int(len(d)), "base": agg("b"), "rated": agg("r"), "same_outcome": same,
            "n_rated_total": int(len(df))}
