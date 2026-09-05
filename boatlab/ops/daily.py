"""日次運用ジョブ（docs/07）。

ingest_today   : BoatraceOpenAPI/api today.json → 出走表・直前情報・結果（fetched_at=now）
fetch_odds     : 公式 odds3t（締切前のレースのみ、1レース最大2回）
predict        : 締切前のレースに対して予想を生成し predictions に追記（stage=program/final）
score          : 結果が揃った予想を採点（created_at < post_time_at_pred 等を検証）
train          : 前日までのデータで Predictor を学習して保存・登録
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select, text

from boatlab.backtest.dataset import build_race_dataset
from boatlab.backtest.metrics import score_race
from boatlab.config import OPENAPI_API_DAY, OPENAPI_API_TODAY, OPENAPI_API_TODAY_ALT
from boatlab.features.build import attach_labels, build_features
from boatlab.features.history import HistoryFrames, load_history
from boatlab.ingest.base import Fetcher, NotFound
from boatlab.ingest.parsers import parse_v1_day
from boatlab.model.pipeline import Predictor
from boatlab.model.selection import FocusedParams, SelectionParams, select_focused
from boatlab.model.trifecta import PERM_LABELS as _PL
from boatlab.model.staking import StakingParams
from boatlab.model.trifecta import PERM_LABELS, combo_index
from boatlab.store.db import session_scope
from boatlab.store.models import (
    ModelVersion, OddsSnapshot, Prediction, PredictionSelection, Race, Result, Scoring, SettingsVersion,
)
from boatlab.store.writer import write_bundle
from boatlab.util import now_jst

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- ingest
def ingest_today(fetcher: Fetcher, d: date | None = None) -> dict:
    d = d or now_jst().date()
    now = now_jst()
    key = f"today/{d:%Y%m%d}_{now:%H%M%S}.json"
    doc = None
    for url in (OPENAPI_API_TODAY, OPENAPI_API_TODAY_ALT, OPENAPI_API_DAY.format(yyyy=d.year, yyyymmdd=d.strftime("%Y%m%d"))):
        try:
            doc, _ = fetcher.fetch_json("openapi_api", url, key, use_cache=False)
            break
        except NotFound:
            continue
        except Exception as e:  # 次の URL へ
            log.warning("today fetch failed %s: %r", url, e)
    if doc is None:
        return {"date": str(d), "error": "no source reachable"}
    bundle = parse_v1_day(d, doc, source_prefix="openapi_api", fetched_at=now)
    # 前日分のまま未更新のファイルを弾く（race_id はファイル内日付で振られる）
    keep = {r.race_id for r in bundle.races if r.race_date == d}
    for attr in ("races", "entries", "previews", "conditions", "results", "result_entries", "odds"):
        setattr(bundle, attr, [x for x in getattr(bundle, attr) if x.race_id in keep])
    if not keep:
        return {"date": str(d), "stale_file": True}
    with session_scope() as s:
        counts = write_bundle(s, bundle)
    return {"date": str(d), **counts}


# ---------------------------------------------------------------- predict
def _settings_row(s) -> SettingsVersion:
    row = s.execute(select(SettingsVersion).order_by(SettingsVersion.id.desc())).scalars().first()
    if row is None:
        row = SettingsVersion()
        s.add(row)
        s.flush()
    return row


def _settings_id(s) -> int:
    return _settings_row(s).id


def focused_from_settings(row: SettingsVersion) -> FocusedParams:
    """設定の extra.focused（絞り込み型）。無ければ既定値（検証済みの値）。"""
    return FocusedParams.from_dict((row.extra or {}).get("focused"))


def staking_from_settings(row: SettingsVersion) -> StakingParams:
    """設定の extra.staking（無ければ均等 stake_per_point×points）。合計は points×stake_per_point に固定。"""
    d = dict((row.extra or {}).get("staking") or {})
    d.setdefault("total", int(row.points or 15) * int(row.stake_per_point or 200))
    return StakingParams.from_dict(d)


def predict_pending(predictor: Predictor, stage: str, role: str = "active", d: date | None = None,
                    min_minutes_before_close: int = 4, max_minutes_before_close: int | None = None,
                    now: datetime | None = None, hist_cache: HistoryFrames | None = None) -> dict:
    """締切まで min_minutes 以上あるレースに予想を保存する（同一 stage/role は1回のみ）。

    now はテスト用（過去日をシミュレート）。created_at は常に実時刻（トリガで担保）なので、
    過去日のシミュレーションは採点時に invalid（created_after_close）になる。
    """
    d = d or now_jst().date()
    now = now or now_jst()
    with session_scope() as s:
        races = s.execute(select(Race).where(Race.race_date == d, Race.status != "cancelled")).scalars().all()
        done = {rid for (rid,) in s.execute(select(Prediction.race_id).where(
            Prediction.model_version == predictor.version, Prediction.stage == stage, Prediction.role == role))}
        srow = _settings_row(s)
        settings_id = srow.id
        staking = staking_from_settings(srow)
        focused = focused_from_settings(srow)
        # 絞り込み型は本体（role）と独立に保存する（role='focused'、確定予想のみ）
        focused_role = f"{role}_focused" if role != "active" else "focused"
        done_f = {rid for (rid,) in s.execute(select(Prediction.race_id).where(
            Prediction.model_version == predictor.version, Prediction.stage == stage, Prediction.role == focused_role))}
    targets = []
    for r in races:
        if r.id in done or r.closed_at is None:
            continue
        mins = (r.closed_at - now).total_seconds() / 60
        if mins < min_minutes_before_close:
            continue
        if max_minutes_before_close is not None and mins > max_minutes_before_close:
            continue
        targets.append(r)
    if not targets:
        return {"predicted": 0}
    ids = [r.id for r in targets]
    # 履歴（前日まで）と対象（当日）。hist_cache はスケジューラが日単位で使い回す
    hist = hist_cache if hist_cache is not None else load_history(d - timedelta(days=3 * 365), d - timedelta(days=1))
    today = load_history(d, d)
    tf = HistoryFrames(today.races[today.races["id"].isin(ids)], today.entries[today.entries["race_id"].isin(ids)],
                       today.previews[today.previews["race_id"].isin(ids)] if len(today.previews) else today.previews,
                       today.conditions[today.conditions["race_id"].isin(ids)] if len(today.conditions) else today.conditions,
                       pd.DataFrame(), pd.DataFrame(columns=hist.result_entries.columns))
    x = build_features(tf, hist)
    x["is_absent"] = False
    # 予想時オッズ（公式・15分以内に取得したもの）
    odds_by_race = {}
    with session_scope() as s:
        for o in s.execute(select(OddsSnapshot).where(OddsSnapshot.race_id.in_(ids), OddsSnapshot.bet_type == "3t",
                                                      OddsSnapshot.source == "official_web")
                           .order_by(OddsSnapshot.captured_at)).scalars():
            if (now - o.captured_at).total_seconds() <= 15 * 60:
                odds_by_race[o.race_id] = (o.id, np.array([np.nan if o.odds.get(k) is None else float(o.odds[k]) for k in PERM_LABELS]))
    outs = predictor.predict_races(x, {k: v[1] for k, v in odds_by_race.items()}, staking=staking)
    n = 0
    with session_scope() as s:
        race_map = {r.id: r for r in targets}
        for o in outs:
            r = race_map[o["race_id"]]
            xr = x[x["race_id"] == o["race_id"]]
            feats = xr.set_index("lane")[[c for c in xr.columns if c not in ("lane", "race_date", "closed_at", "fetched_at")]] \
                .astype(object).where(lambda d: d.notna(), None).to_dict("index")
            p = Prediction(
                race_id=o["race_id"], model_version=predictor.version, settings_id=settings_id, stage=stage, role=role,
                created_at=now_jst(), asof_ts=now, post_time_at_pred=r.closed_at,
                features_used={str(k): {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) and vv is not None else (None if vv is None else str(vv))) for kk, vv in v.items()} for k, v in feats.items()},
                odds_snapshot_id=odds_by_race.get(o["race_id"], (None,))[0],
                completeness=o["completeness"], missing_fields=None, flags=o["flags"], boat_eval=o["boat_eval"],
                probs=o["probs"], odds_used=o["odds_used"], ev=o["ev"], confidence=o["confidence"],
                expected_return=o["expected_return"], decision=o["decision"], skip_reason=o["skip_reason"],
                rationale=o["rationale"], rationale_text=o["rationale"]["summary"], input_hash=o["input_hash"],
            )
            s.add(p)
            s.flush()
            for sel in o["selections"]:
                s.add(PredictionSelection(prediction_id=p.id, combo=sel["combo"], rank=sel["rank"], kind=sel["kind"],
                                          stake=sel["stake"], prob=sel["prob"], odds_at_pred=sel["odds"],
                                          odds_source=o["odds_source"], ev=sel["ev"]))
            n += 1
            # ---- 絞り込み型（同じ確率・オッズから別の買い目を作り、role='focused' として追記）
            if focused.enabled and stage == "final" and o["race_id"] not in done_f:
                parr = np.array([o["probs"][k] for k in _PL], dtype=float)
                oarr = np.array([np.nan if o["odds_used"][k] is None else float(o["odds_used"][k]) for k in _PL])
                f = select_focused(parr, oarr, focused, completeness=float(o["completeness"]),
                                   odds_estimated=bool(o["flags"].get("odds_estimated")))
                pf = Prediction(
                    race_id=o["race_id"], model_version=predictor.version, settings_id=settings_id, stage=stage, role=focused_role,
                    created_at=now_jst(), asof_ts=now, post_time_at_pred=r.closed_at,
                    features_used=None, odds_snapshot_id=odds_by_race.get(o["race_id"], (None,))[0],
                    completeness=o["completeness"], missing_fields=None,
                    flags={**o["flags"], "mode": "focused", "S15": round(f.S15, 4), "n_points": len(f.points),
                           "stake_total": int(sum(f.stakes))},
                    boat_eval=o["boat_eval"], probs=o["probs"], odds_used=o["odds_used"], ev=o["ev"],
                    confidence=o["confidence"], expected_return=f.expected_return, decision=f.decision,
                    skip_reason=f.skip_reason, rationale=o["rationale"],
                    rationale_text=(o["rationale"]["summary"] + f"（絞り込み型: {len(f.points)}点・{int(sum(f.stakes))}円）"),
                    input_hash=o["input_hash"],
                )
                s.add(pf)
                s.flush()
                for rnk, (j, stk) in enumerate(zip(f.points, f.stakes)):
                    s.add(PredictionSelection(prediction_id=pf.id, combo=_PL[j], rank=rnk + 1, kind="main",
                                              stake=int(stk), prob=float(parr[j]),
                                              odds_at_pred=(None if not np.isfinite(oarr[j]) else float(oarr[j])),
                                              odds_source=o["odds_source"], ev=float(f.ev[j])))
    return {"predicted": n, "stage": stage, "date": str(d)}


# ---------------------------------------------------------------- score
def score_pending() -> dict:
    n = 0
    with session_scope() as s:
        rows = s.execute(text("""
            SELECT p.id FROM predictions p
            JOIN results res ON res.race_id = p.race_id
            LEFT JOIN scoring sc ON sc.prediction_id = p.id
            WHERE sc.prediction_id IS NULL""")).all()
        for (pid,) in rows:
            p = s.get(Prediction, pid)
            r = s.get(Race, p.race_id)
            res = s.get(Result, p.race_id)
            sels = s.execute(select(PredictionSelection).where(PredictionSelection.prediction_id == pid)).scalars().all()
            invalid = None
            if p.created_at >= p.post_time_at_pred or (r.closed_at and p.created_at >= r.closed_at):
                invalid = "created_after_close"
            elif res.fetched_at is not None and r.closed_at is not None and res.fetched_at <= r.closed_at:
                invalid = "result_before_close"
            elif r.status == "cancelled":
                invalid = "cancelled"
            tri = combo_index(res.trifecta) if res.trifecta else None
            sel_idx = [combo_index(x.combo) for x in sels]
            main_idx = [combo_index(x.combo) for x in sels if x.kind == "main"]
            stakes = [int(x.stake) for x in sels]
            sc = score_race(sel_idx, main_idx, -1 if tri is None else tri, res.trifecta_payout, res.refunds or [],
                            sels[0].stake if sels else 200, cancelled=(r.status == "cancelled"), stakes=stakes or None)
            valid = invalid is None and sc["valid"]
            hit = sc["hit"] if valid else None
            cat = None
            if valid:
                cat = ("buy_hit" if hit else "buy_miss") if p.decision == "buy" else ("skip_would_hit" if hit else "skip_correct")
            pred_odds = next((x.odds_at_pred for x in sels if tri is not None and x.combo == PERM_LABELS[tri]), None)
            s.add(Scoring(prediction_id=pid, race_id=p.race_id, valid=valid, invalid_reason=invalid,
                          actual_trifecta=res.trifecta, actual_payout=res.trifecta_payout, hit=hit, hit_kind=sc["hit_kind"] if valid else None,
                          refunded_points=sc["refunded_points"], refunded_stake=sc["refunded_stake"],
                          stake_total=sc["stake_total"], payout_total=sc["payout_total"], pnl=sc["pnl"],
                          roi=(sc["payout_total"] / sc["stake_total"] if sc["stake_total"] else None), category=cat,
                          odds_final_ratio=((res.trifecta_payout / 100) / pred_odds if (pred_odds and res.trifecta_payout) else None)))
            n += 1
    return {"scored": n}


# ---------------------------------------------------------------- train
def train_and_register(version: str, until: date, selection: SelectionParams, description: str = "",
                       half_life: float | None = 2.0, num_rounds: int = 400, train_max_rows: int | None = None,
                       status: str = "candidate", parent: str | None = None, years: int = 8,
                       seeds: tuple[int, ...] = (7,)) -> Predictor:
    from boatlab.backtest.dataset import build_entry_dataset
    d0 = date(until.year - years, 1, 1)
    X = build_entry_dataset(d0, until)
    R = build_race_dataset(d0, until)
    pr = Predictor.train(X, R, version, pd.Timestamp(until), selection, half_life, num_rounds, train_max_rows=train_max_rows,
                         seeds=seeds)
    path = pr.save()
    with session_scope() as s:
        mv = s.get(ModelVersion, version)
        if mv is None:
            mv = ModelVersion(version=version, feature_set_version=pr.params["feature_set_version"],
                              selection_version=pr.params["selection_version"], params=pr.params, parent_version=parent)
            s.add(mv)
        mv.description = description or mv.description
        mv.artifact_path = str(path)
        mv.trained_until = until
        mv.status = status
        mv.params = {k: v for k, v in pr.params.items()}
    return pr
