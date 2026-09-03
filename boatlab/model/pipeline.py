"""学習済み成果物の束（Predictor）：学習・保存・読込・レース予想。

当日予想（ops/daily.py）とバックテスト（backtest/walkforward.py）は同じ部品を使う。
成果物: data/models/<version>/ に pickle（LightGBM は model_to_string で保存）。
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from boatlab.config import DATA_DIR
from boatlab.features.build import FEATURE_SET_VERSION, NUMERIC_FEATURES
from boatlab.model.calibration import ComboCalibrator, SetCalibrator
from boatlab.model.market import MarketOddsModel
from boatlab.model.selection import SELECTION_VERSION, SelectionParams, decide, labels, select_points
from boatlab.model.staking import StakingParams, allocate
from boatlab.model.strength import BaselineProgramLogit, StrengthModel
from boatlab.model.trifecta import PERM_LABELS, PERMS, fit_lambdas, race_matrix, trifecta_probs


@dataclass
class Predictor:
    version: str
    strength: StrengthModel
    market_pub: BaselineProgramLogit
    lam: tuple[float, float]
    combo_cal: ComboCalibrator
    market: MarketOddsModel
    set_cal: SetCalibrator
    selection: SelectionParams
    trained_until: date
    holdout_from: date
    params: dict = field(default_factory=dict)

    # ---------------- 学習
    @classmethod
    def train(cls, X: pd.DataFrame, R: pd.DataFrame, version: str, until: pd.Timestamp,
              selection: SelectionParams, half_life: float | None = 2.0, num_rounds: int = 400,
              holdout_months: int = 3, train_max_rows: int | None = None, seeds: tuple[int, ...] = (7,)) -> "Predictor":
        Rr = R.set_index("race_id")
        hs = until - pd.DateOffset(months=holdout_months)
        train = X[(X["race_date"] < hs) & X["finish_pos"].notna()]
        hold = X[(X["race_date"] >= hs) & (X["race_date"] <= until) & X["finish_pos"].notna()]
        if train_max_rows and len(train) > train_max_rows:
            keep = pd.Series(train["race_id"].unique()).sample(n=train_max_rows // 6, random_state=0)
            train = train[train["race_id"].isin(keep)]
        strength = StrengthModel(num_rounds=num_rounds, half_life_years=half_life, seeds=tuple(seeds)).fit(train, asof=hs)
        pub = BaselineProgramLogit().fit(train)
        pred = strength.predict(hold)
        mats = tuple(race_matrix(pred, hold, c)[0] for c in ("p_win", "p_top2", "p_top3"))
        ids = race_matrix(pred, hold, "p_win")[1]
        tri = Rr.loc[ids, "tri_idx"].values.astype(int)
        lam = fit_lambdas(*mats, tri)[:2]
        p_raw = trifecta_probs(*mats, *lam)
        combo_cal = ComboCalibrator().fit(p_raw, tri)
        p_h = combo_cal.transform(p_raw)
        mk = MarketOddsModel()
        pp = pub.predict(hold)
        q = mk.q_from_public(*[race_matrix(pp, hold, c)[0] for c in ("p_win", "p_top2", "p_top3")])
        mk.fit(q, Rr.loc[ids, ["win_lane", "win_amount", "ex_a", "ex_b", "ex_amount", "tri_idx", "tri_amount"]].reset_index(drop=True))
        odds_h = mk.odds(q)
        S, hit = [], []
        for i in range(len(ids)):
            sel = select_points(p_h[i], odds_h[i], selection)
            S.append(sel.S)
            hit.append(1.0 if tri[i] in (sel.main + sel.hole) else 0.0)
        set_cal = SetCalibrator().fit(np.array(S), np.array(hit))
        return cls(version, strength, pub, lam, combo_cal, mk, set_cal, selection, until.date(), hs.date(),
                   params={"half_life": half_life, "num_rounds": num_rounds, "holdout_months": holdout_months,
                           "seeds": list(seeds), "train_max_rows": train_max_rows,
                           "feature_set_version": FEATURE_SET_VERSION, "selection_version": SELECTION_VERSION,
                           "selection": selection.__dict__, "lam": lam, "market_fit": mk.fit_report})

    # ---------------- 保存/読込
    def save(self, root: Path | None = None) -> Path:
        d = (root or DATA_DIR / "models") / self.version
        d.mkdir(parents=True, exist_ok=True)
        boosters = {k: b.model_to_string() for k, b in self.strength.boosters.items()}
        payload = {k: v for k, v in self.__dict__.items() if k != "strength"}
        payload["strength_meta"] = {"num_rounds": self.strength.num_rounds, "params": self.strength.params,
                                    "half_life_years": self.strength.half_life_years, "seeds": list(self.strength.seeds)}
        with open(d / "predictor.pkl", "wb") as f:
            pickle.dump(payload, f)
        (d / "boosters.json").write_text(json.dumps(boosters))
        (d / "params.json").write_text(json.dumps(self.params, ensure_ascii=False, default=str, indent=1))
        return d

    @classmethod
    def load(cls, version: str, root: Path | None = None) -> "Predictor":
        d = (root or DATA_DIR / "models") / version
        with open(d / "predictor.pkl", "rb") as f:
            payload = pickle.load(f)
        meta = payload.pop("strength_meta")
        st = StrengthModel(num_rounds=meta["num_rounds"], params=meta["params"], half_life_years=meta["half_life_years"],
                           seeds=tuple(meta.get("seeds", (7,))))
        st.boosters = {k: lgb.Booster(model_str=s) for k, s in json.loads((d / "boosters.json").read_text()).items()}
        return cls(strength=st, **payload)

    # ---------------- 予想
    def predict_races(self, x: pd.DataFrame, odds_by_race: dict[int, np.ndarray] | None = None,
                      staking: StakingParams | None = None) -> list[dict]:
        """x: build_features の出力（対象レース群）。odds_by_race: 実オッズ(120)。無ければ推定。

        staking: 15点への資金配分（設定値）。None なら均等（selection.stake 円×点数）。
        """
        odds_by_race = odds_by_race or {}
        staking = staking or StakingParams(method="uniform", total=self.selection.stake * 15)
        pred = self.strength.predict(x)
        mats = tuple(race_matrix(pred, x, c)[0] for c in ("p_win", "p_top2", "p_top3"))
        ids = race_matrix(pred, x, "p_win")[1]
        p = self.combo_cal.transform(trifecta_probs(*mats, *self.lam))
        pp = self.market_pub.predict(x)
        q = self.market.q_from_public(*[race_matrix(pp, x, c)[0] for c in ("p_win", "p_top2", "p_top3")])
        odds_est = self.market.odds(q)
        comp = x.groupby("race_id")["completeness"].first()
        absent_by_race = x.groupby("race_id")["is_absent"].apply(lambda s: s.values) if "is_absent" in x else None
        out = []
        for i, rid in enumerate(ids):
            real = odds_by_race.get(rid)
            use_real = real is not None and np.isfinite(real).sum() >= 100
            odds = real if use_real else odds_est[i]
            sel = select_points(p[i], odds, self.selection)
            C = float(self.set_cal.transform(np.array([sel.S]))[0])
            flags = {"odds_estimated": not use_real, "hole_relaxed": sel.hole_relaxed, "staking": staking.method}
            pts = sel.main + sel.hole
            stakes = allocate(pts, sel.main, p[i], odds, staking)
            decision, reason = decide(C, sel.expected_return, float(comp.get(rid, 0.0)), flags, self.selection)
            xr = x[x["race_id"] == rid].sort_values("lane")
            boat_eval = _boat_eval(xr, pred.loc[xr.index], mats, i)
            out.append({
                "race_id": int(rid), "probs": {PERM_LABELS[j]: float(p[i][j]) for j in range(120)},
                "odds_used": {PERM_LABELS[j]: (None if not np.isfinite(odds[j]) else float(odds[j])) for j in range(120)},
                "odds_source": "real" if use_real else "estimated",
                "ev": {PERM_LABELS[j]: float(sel.ev[j]) for j in range(120)},
                "main": labels(sel.main), "hole": labels(sel.hole),
                "selections": [{"combo": PERM_LABELS[j], "rank": r + 1, "kind": "main" if r < len(sel.main) else "hole",
                                "stake": int(stakes[r]), "prob": float(p[i][j]),
                                "odds": (None if not np.isfinite(odds[j]) else float(odds[j])), "ev": float(sel.ev[j])}
                               for r, j in enumerate(pts)],
                "stake_total": int(sum(stakes)), "staking": staking.to_dict(),
                "S": sel.S, "confidence": C, "expected_return": sel.expected_return,
                "decision": decision, "skip_reason": reason, "completeness": float(comp.get(rid, 0.0)),
                "flags": flags, "boat_eval": boat_eval,
                "rationale": _rationale(xr, boat_eval, sel, p[i]),
                "input_hash": hashlib.sha256(pd.util.hash_pandas_object(xr[NUMERIC_FEATURES].fillna(-999)).values.tobytes()).hexdigest()[:16],
            })
        return out


def _boat_eval(xr: pd.DataFrame, pr: pd.DataFrame, mats, i) -> dict:
    ev = {}
    for k, (_, row) in enumerate(xr.iterrows()):
        ev[str(int(row["lane"]))] = {
            "regno": None if pd.isna(row["regno"]) else int(row["regno"]), "name": row.get("name"),
            "klass": row.get("klass"), "course_pred": int(row["course_pred"]),
            "p_win": float(pr.iloc[k]["p_win"]), "p_top2": float(pr.iloc[k]["p_top2"]), "p_top3": float(pr.iloc[k]["p_top3"]),
            "nat_win_rate": _f(row["nat_win_rate"]), "loc_win_rate": _f(row["loc_win_rate"]), "avg_st": _f(row["avg_st"]),
            "motor_rate2": _f(row["motor_rate2"]), "exhibition_time": _f(row["exhibition_time"]),
            "exhibition_rank": _f(row["exhibition_time_rank"]), "st_exh": _f(row["st_exh"]),
            "course_win_rate_2y": _f(row["rc_2y_win_shr"]), "course_n_2y": _f(row["rc_2y_n"]),
            "stadium_course_win_rate": _f(row["rsc_all_win_shr"]), "stadium_course_n": _f(row["rsc_all_n"]),
            "stadium_course_base_1y": _f(row["sc_1y_win_rate"]), "motor_stadium_90d_win": _f(row["ms_90d_win_shr"]),
            "recent30d_avg_finish": _f(row["r_30d_avg_fin"]),
        }
    return ev


def _f(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else float(v)


def _rationale(xr: pd.DataFrame, be: dict, sel, p: np.ndarray) -> dict:
    top = sorted(be.items(), key=lambda kv: -kv[1]["p_win"])
    axis = top[0][0]
    p1 = sum(p[j] for j in range(120) if PERMS[j][0] == int(axis) - 1)
    return {
        "axis_lane": int(axis), "axis_p_win": round(top[0][1]["p_win"], 3), "axis_trifecta_mass": round(float(p1), 3),
        "rivals": [int(k) for k, _ in top[1:3]],
        "main_mass": round(float(sum(p[j] for j in sel.main)), 3), "hole_mass": round(float(sum(p[j] for j in sel.hole)), 3),
        "summary": f"{axis}号艇（1着確率{top[0][1]['p_win']:.0%}）を軸、相手本線は{top[1][0]}・{top[2][0]}号艇。"
                   f"本線10点の合計確率{sum(p[j] for j in sel.main):.0%}、穴5点{sum(p[j] for j in sel.hole):.0%}。",
    }
