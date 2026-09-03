"""ウォークフォワード・バックテスト（docs/05 §2）。

2段階に分ける:
 1) run_probs   : 期間ごとに 強さモデル → λ → 校正 → 市場オッズ推定 を行い、holdout/test の
                  120通り確率・オッズ・正解を ProbStore に保存（重い処理。モデル設定ごとに1回）
 2) evaluate    : ProbStore に対して選定パラメータ（穴オッズ・β・閾値…）を当てて採点
                  （軽い処理。パラメータ探索はこちらを回す）

各期間 m について:
  train    = race_date < holdout_start（拡張ウィンドウ・時間減衰）
  holdout  = [period_start − holdout_months, period_start)  … λ・校正器・市場オッズ推定器・セット校正の学習
  period   = [period_start, next_period_start)              … 予想→記録→採点
予想は本番 predictions ではなく records に書く。
"""
from __future__ import annotations

import gc
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from boatlab.backtest.metrics import score_race
from boatlab.model.calibration import ComboCalibrator, SetCalibrator
from boatlab.model.market import MarketOddsModel
from boatlab.model.selection import SelectionParams, decide, select_points
from boatlab.model.strength import BaselineCourseRate, BaselineProgramLogit, StrengthModel
from boatlab.model.trifecta import fit_lambdas, race_matrix, trifecta_probs

log = logging.getLogger(__name__)


@dataclass
class WFConfig:
    period_start: str            # 'YYYY-MM-01'
    period_end: str              # 含む
    freq: str = "MS"             # 予測期間の刻み（MS=月, QS=四半期）
    refit_every: int = 1         # 何期間ごとに強さモデルを再学習するか
    holdout_months: int = 3
    model: str = "lgb"           # lgb / m0 / m0b
    half_life_years: float | None = 2.0
    num_rounds: int = 400
    lgb_params: dict = field(default_factory=dict)
    lam_fixed: tuple[float, float] | None = None   # None なら holdout で探索
    train_max_rows: int | None = None              # 探索時の間引き（None=全件）
    label: str = "lgb"


@dataclass
class PeriodProbs:
    period: str
    hold_ids: list
    hold_p: np.ndarray          # (H,120) 校正後
    hold_odds_est: np.ndarray   # (H,120)
    hold_tri: np.ndarray
    test_ids: list
    test_p: np.ndarray
    test_odds_est: np.ndarray
    test_tri: np.ndarray
    lam: tuple
    market_fit: dict
    train_rows: int


@dataclass
class ProbStore:
    cfg: WFConfig
    periods: list[PeriodProbs]

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "ProbStore":
        with open(path, "rb") as f:
            return pickle.load(f)


def _make_model(cfg: WFConfig):
    if cfg.model == "m0":
        return BaselineCourseRate()
    if cfg.model == "m0b":
        return BaselineProgramLogit()
    return StrengthModel(num_rounds=cfg.num_rounds, half_life_years=cfg.half_life_years,
                         params={**StrengthModel().params, **cfg.lgb_params})


def _mats(model, x: pd.DataFrame):
    pred = model.predict(x)
    pw, ids = race_matrix(pred, x, "p_win")
    p2, _ = race_matrix(pred, x, "p_top2")
    p3, _ = race_matrix(pred, x, "p_top3")
    return (pw, p2, p3), ids


def run_probs(X: pd.DataFrame, R: pd.DataFrame, cfg: WFConfig) -> ProbStore:
    R = R.set_index("race_id")
    starts = pd.date_range(cfg.period_start, cfg.period_end, freq=cfg.freq)
    out: list[PeriodProbs] = []
    model = None
    market_pub = None
    for k, ps in enumerate(starts):
        pe = min((ps + pd.tseries.frequencies.to_offset(cfg.freq)) - pd.Timedelta(days=1), pd.Timestamp(cfg.period_end))
        hs = ps - pd.DateOffset(months=cfg.holdout_months)
        mask_train = (X["race_date"] < hs) & X["finish_pos"].notna()
        if cfg.train_max_rows and int(mask_train.sum()) > cfg.train_max_rows:
            # レース単位で間引く（艇6行を壊さない）。大きな中間コピーを作らない
            rids = X.loc[mask_train, "race_id"].unique()
            keep = set(pd.Series(rids).sample(n=cfg.train_max_rows // 6, random_state=0))
            mask_train &= X["race_id"].isin(keep)
        train = X[mask_train]
        hold = X[(X["race_date"] >= hs) & (X["race_date"] < ps) & X["finish_pos"].notna()]
        test = X[(X["race_date"] >= ps) & (X["race_date"] <= pe)]
        if len(train) == 0 or len(test) == 0:
            continue
        if model is None or k % cfg.refit_every == 0:
            model = _make_model(cfg).fit(train, asof=hs)
            market_pub = BaselineProgramLogit().fit(train)
        mats_h, ids_h = _mats(model, hold)
        tri_h = R.loc[ids_h, "tri_idx"].values.astype(int)
        lam = cfg.lam_fixed or fit_lambdas(*mats_h, tri_h)[:2]
        p_raw_h = trifecta_probs(*mats_h, *lam)
        combo_cal = ComboCalibrator().fit(p_raw_h, tri_h)
        p_h = combo_cal.transform(p_raw_h)
        mk = MarketOddsModel()
        q_h = mk.q_from_public(*_mats(market_pub, hold)[0])
        mk.fit(q_h, R.loc[ids_h, ["win_lane", "win_amount", "ex_a", "ex_b", "ex_amount", "tri_idx", "tri_amount"]].reset_index(drop=True))
        odds_h = mk.odds(q_h)
        mats_t, ids_t = _mats(model, test)
        p_t = combo_cal.transform(trifecta_probs(*mats_t, *lam))
        odds_t = mk.odds(mk.q_from_public(*_mats(market_pub, test)[0]))
        tri_t = R.loc[ids_t, "tri_idx"].values.astype(int)
        out.append(PeriodProbs(str(ps.date()), ids_h, p_h.astype(np.float32), odds_h.astype(np.float32), tri_h,
                               ids_t, p_t.astype(np.float32), odds_t.astype(np.float32), tri_t, lam, mk.fit_report, int(len(train))))
        log.info("[%s] period %s: train=%d hold=%d test=%d lam=%s market=%s", cfg.label, ps.date(), len(train), len(ids_h), len(ids_t), lam, mk.fit_report)
        del train, hold, test, mats_h, mats_t, p_raw_h, p_h, p_t, q_h, odds_h, odds_t
        gc.collect()
    return ProbStore(cfg, out)


def evaluate(store: ProbStore, R: pd.DataFrame, prm: SelectionParams, use_real_odds: bool = True,
             X_completeness: pd.Series | None = None, staking: "StakingParams | None" = None) -> pd.DataFrame:
    """選定パラメータを当てて採点。実オッズ（2026〜）があればそれを使う。staking で点ごとの配分を変える。"""
    from boatlab.model.staking import allocate
    R = R.set_index("race_id")
    records: list[dict] = []
    for per in store.periods:
        S_h, hit_h = [], []
        for i in range(len(per.hold_ids)):
            sel = select_points(per.hold_p[i], per.hold_odds_est[i], prm)
            S_h.append(sel.S)
            hit_h.append(1.0 if per.hold_tri[i] in (sel.main + sel.hole) else 0.0)
        set_cal = SetCalibrator().fit(np.array(S_h), np.array(hit_h))
        for i, rid in enumerate(per.test_ids):
            row = R.loc[rid]
            real = row["real_odds"] if use_real_odds else None
            use_real = isinstance(real, np.ndarray) and np.isfinite(real).sum() >= 100
            odds = real if use_real else per.test_odds_est[i]
            sel = select_points(per.test_p[i], odds, prm)
            C = float(set_cal.transform(np.array([sel.S]))[0])
            comp = float(X_completeness.get(rid, 1.0)) if X_completeness is not None else 1.0
            decision, reason = decide(C, sel.expected_return, comp, {}, prm)
            tri = int(per.test_tri[i])
            pts = sel.main + sel.hole
            stakes = allocate(pts, sel.main, per.test_p[i], odds, staking) if staking is not None else None
            sc = score_race(pts, sel.main, tri, row["trifecta_payout"], row["refund_lanes"], prm.stake,
                            cancelled=(row["status"] == "cancelled"), stakes=stakes)
            stake_hit = (stakes[pts.index(tri)] if (stakes is not None and tri in pts) else (prm.stake if tri in pts else 0))
            records.append(dict(
                race_id=rid, race_date=row["race_date"], stadium_code=row["stadium_code"], grade=row["grade"],
                period=per.period, decision=decision, skip_reason=reason, confidence=C, S=sel.S,
                expected_return=sel.expected_return, odds_source=("real" if use_real else "estimated"),
                hole_relaxed=sel.hole_relaxed, p_lane1_win=float(per.test_p[i][:20].sum()),
                main=sel.main, hole=sel.hole, actual_idx=tri, actual_payout=row["trifecta_payout"],
                hole_odds_mean=float(np.nanmean(odds[sel.hole])) if sel.hole else np.nan,
                completeness=comp, logp_actual=(float(np.log(max(per.test_p[i][tri], 1e-12))) if tri >= 0 else np.nan),
                stake_hit=stake_hit, stake_min=(min(stakes) if stakes else prm.stake), stake_max=(max(stakes) if stakes else prm.stake),
                main_stake=(sum(stakes[:len(sel.main)]) if stakes else prm.stake * len(sel.main)),
                hole_stake=(sum(stakes[len(sel.main):]) if stakes else prm.stake * len(sel.hole)),
                **sc,
            ))
    return pd.DataFrame(records)


def probs_quality(store: ProbStore) -> dict:
    """モデル品質：3連単 log-loss（テスト期間の平均 log p(正解)）と 1着 log-loss。"""
    lp = []
    for per in store.periods:
        ok = per.test_tri >= 0
        lp.append(np.log(np.clip(per.test_p[ok, per.test_tri[ok]], 1e-12, None)))
    lp = np.concatenate(lp) if lp else np.array([])
    return {"n": int(len(lp)), "trifecta_logloss": float(-lp.mean()) if len(lp) else float("nan")}
