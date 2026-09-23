"""日次運用ジョブ（docs/07）。

ingest_today   : BoatraceOpenAPI/api today.json → 出走表・直前情報・結果（fetched_at=now）
fetch_odds     : 公式 odds3t（締切前のレースのみ、1レース最大2回）
predict        : 締切前のレースに対して予想を生成し predictions に追記（stage=program/final）
score          : 結果が揃った予想を採点（created_at < post_time_at_pred 等を検証）
train          : 前日までのデータで Predictor を学習して保存・登録
"""
from __future__ import annotations
import json
import logging
import re
from dataclasses import replace
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import func, select, text

from boatlab.backtest.dataset import build_race_dataset
from boatlab.backtest.metrics import score_race
from boatlab.config import OPENAPI_API_DAY, OPENAPI_API_TODAY, OPENAPI_API_TODAY_ALT
from boatlab.features.build import attach_labels, build_features
from boatlab.features.history import HistoryFrames, load_history
from boatlab.ingest.base import Fetcher, NotFound
from boatlab.ingest.parsers import parse_v1_day
from boatlab.model.pipeline import Predictor
from boatlab.model.modes import (HONMEI_MULTIPLE, MODES_VERSION, Q_MAN, ModeParams, adjust_probs_for_ratings, honmei_grid_for,
                                 market_probs, race_tags, select_ana, select_ev1, select_honmei, select_katai_top,
                                 select_place)
from boatlab.model.selection import FocusedParams, SelectionParams, select_focused
from boatlab.model.trifecta import PERM_LABELS as _PL
from boatlab.model.staking import StakingParams
from boatlab.model.trifecta import PERM_LABELS, combo_index
from boatlab.store.db import session_scope
from boatlab.store.models import (
    ExhibitionRating, ModelVersion, OddsSnapshot, PoolGapPick, Prediction, PredictionSelection, Race, Result, Scoring,
    SettingsVersion,
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
        dropped = _drop_unchanged_snapshots(s, bundle)
        counts = write_bundle(s, bundle)
    _prune_today_raw(fetcher, d)
    return {"date": str(d), **counts, "unchanged_dropped": dropped}


_PREVIEW_KEYS = ("course", "st_exh", "weight", "weight_adj", "exhibition_time", "tilt", "propeller", "parts")
_COND_KEYS = ("weather", "temp_c", "water_temp_c", "wind_dir", "wind_speed_m", "wave_cm")


def _drop_unchanged_snapshots(s, bundle) -> int:
    """直前情報・気象が前回取得と同じ内容なら追記しない（数分おきの再取込でスナップショットが膨らむのを防ぐ）。
    値が変わったときだけ新しい行が入るので「締切前の最新値」は従来どおり得られる。"""
    from sqlalchemy import text
    ids = sorted({p.race_id for p in bundle.previews} | {c.race_id for c in bundle.conditions})
    if not ids:
        return 0
    idl = ",".join(str(i) for i in ids)
    latest_p = {}
    for row in s.execute(text(f"""
        SELECT race_id, lane, source, {", ".join(_PREVIEW_KEYS)} FROM (
          SELECT p.*, ROW_NUMBER() OVER (PARTITION BY race_id, lane, source ORDER BY fetched_at DESC, id DESC) AS rn
          FROM preview_snapshots p WHERE race_id IN ({idl})) WHERE rn = 1""")).mappings():
        latest_p[(row["race_id"], row["lane"], row["source"])] = tuple(_norm(row[k]) for k in _PREVIEW_KEYS)
    latest_c = {}
    for row in s.execute(text(f"""
        SELECT race_id, source, phase, {", ".join(_COND_KEYS)} FROM (
          SELECT c.*, ROW_NUMBER() OVER (PARTITION BY race_id, source, phase ORDER BY observed_at DESC, id DESC) AS rn
          FROM race_conditions c WHERE race_id IN ({idl})) WHERE rn = 1""")).mappings():
        latest_c[(row["race_id"], row["source"], row["phase"])] = tuple(_norm(row[k]) for k in _COND_KEYS)
    n0 = len(bundle.previews) + len(bundle.conditions)
    bundle.previews = [p for p in bundle.previews
                       if latest_p.get((p.race_id, p.lane, p.source)) != tuple(_norm(getattr(p, k, None)) for k in _PREVIEW_KEYS)]
    bundle.conditions = [c for c in bundle.conditions
                         if latest_c.get((c.race_id, c.source, c.phase)) != tuple(_norm(getattr(c, k, None)) for k in _COND_KEYS)]
    return n0 - len(bundle.previews) - len(bundle.conditions)


def _norm(v):
    if v is None:
        return None
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    if isinstance(v, str):
        try:
            return json.dumps(json.loads(v), ensure_ascii=False, sort_keys=True)
        except Exception:
            return v
    if isinstance(v, float):
        return round(v, 4)
    return v


def _prune_today_raw(fetcher: Fetcher, d: date, keep: int = 3) -> None:
    """today.json の原本は数分おきに保存されるので、各日の最新 keep 件だけ残す（1日で数百MBに膨らむため）。"""
    try:
        base = fetcher.raw_path("openapi_api", "today")
        if not base.exists():
            return
        by_day: dict[str, list] = {}
        for f in base.glob("*.json"):
            by_day.setdefault(f.name.split("_")[0], []).append(f)
        for files in by_day.values():
            for f in sorted(files)[:-keep]:
                f.unlink(missing_ok=True)
    except Exception as e:  # 掃除の失敗で取込を止めない
        log.warning("raw prune failed: %r", e)


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


def modes_from_settings(row: SettingsVersion) -> ModeParams:
    """設定の extra.modes（各モードの条件）。無ければ既定値。

    **倍率を変えたら、しきい値表と成立率も一緒に差し替える。** 倍率を上げると
    「Σ(1/オッズ) ≤ 1/倍率」を満たすレースだけが残り、その母集団の確率合計は下にずれる。
    古い表を使い回すとしきい値が高すぎて枠が埋まらなくなる（静かに壊れる）ので、
    設定で倍率だけ動かされた場合はここで対応する表に付け替える。
    表を明示的に保存してある設定（グリッドを自分で入れた場合）はそのまま尊重する。"""
    d = dict((row.extra or {}).get("modes") or {})
    prm = ModeParams.from_dict(d)
    if "honmei_conf_grid" not in d and float(prm.honmei_multiple) != HONMEI_MULTIPLE:
        grid, rate = honmei_grid_for(float(prm.honmei_multiple))
        prm = replace(prm, honmei_conf_grid=grid,
                      honmei_feasible_rate=(rate if "honmei_feasible_rate" not in d
                                            else prm.honmei_feasible_rate))
    return prm


# 2026-09-23: 4モード化。穴狙い(ana)・堅い保証つき(katai) は記録を止めた（選定関数とテストは残す）。
MODE_ROLES = ("katai_t", "honmei", "place", "ev1")
# 展示評価（◎×）を加味した版を並べて記録する対象。role は末尾に "R"（例: katai_tR）。複勝は市場だけで決めるので対象外。
RATED_ROLES = ("katai_t", "honmei", "ev1")
RATED_SUFFIX = "R"


def load_ratings(s, race_id: int, before: datetime) -> dict[int, int]:
    """その時点までに入力されていた最新の評価（lane → +1/-1）。0（無印に戻した）は除く。"""
    rows = s.execute(select(ExhibitionRating).where(ExhibitionRating.race_id == race_id,
                                                    ExhibitionRating.created_at <= before)
                     .order_by(ExhibitionRating.created_at)).scalars().all()
    latest: dict[int, int] = {}
    for x in rows:
        latest[int(x.lane)] = int(x.rating)
    return {k: v for k, v in latest.items() if v}


def rated_output(o: dict, ratings: dict[int, int], beta: float) -> dict:
    """評価を加味した確率で o を作り直す。本線10点＋穴5点の並びも作り直す（堅い・本命はこの並びを使う）。"""
    parr = np.array([float(o["probs"][k]) for k in _PL])
    q = adjust_probs_for_ratings(parr, ratings, beta)
    order = np.argsort(-q)
    o2 = dict(o)
    o2["probs"] = {_PL[i]: float(q[i]) for i in range(120)}
    o2["selections"] = [dict(combo=_PL[int(order[k])], rank=k + 1, kind=("main" if k < 10 else "hole"), stake=100,
                             prob=float(q[order[k]]), odds=o["odds_used"].get(_PL[int(order[k])]), ev=None) for k in range(15)]
    o2["flags"] = dict(o["flags"])
    return o2


def staking_from_settings(row: SettingsVersion) -> StakingParams:
    """設定の extra.staking（無ければ均等 stake_per_point×points）。合計は points×stake_per_point に固定。"""
    d = dict((row.extra or {}).get("staking") or {})
    d.setdefault("total", int(row.points or 15) * int(row.stake_per_point or 200))
    return StakingParams.from_dict(d)


def predict_pending(predictor: Predictor, stage: str, role: str = "active", d: date | None = None,
                    min_minutes_before_close: float = 4, max_minutes_before_close: float | None = None,
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
        modes = modes_from_settings(srow)
        done_m = {mr: {rid for (rid,) in s.execute(select(Prediction.race_id).where(
            Prediction.model_version == predictor.version, Prediction.stage == stage,
            Prediction.role == (mr if role == "active" else f"{role}_{mr}")))}
                  for mr in MODE_ROLES + tuple(x + RATED_SUFFIX for x in RATED_ROLES)}
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
    hist = hist_cache if hist_cache is not None else load_history(d - timedelta(days=3 * 365), d - timedelta(days=1), slim=True)
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
            # ---- 4モード（堅い厚め・本命・複勝単勝・期待値1以上）。確定予想のみ。市場ベースのものは推定オッズでは動かさない
            if stage == "final" and role == "active":
                _record_modes(s, o, r, predictor.version, settings_id, now, odds_by_race, modes, done_m)
                # ---- 展示評価（◎×）があれば、加味した版を role 末尾 R で並べて記録する（対で採点するため）
                ratings = load_ratings(s, o["race_id"], now_jst())
                if ratings:
                    o_r = rated_output(o, ratings, float(modes.rating_beta))
                    _record_modes(s, o_r, r, predictor.version, settings_id, now, odds_by_race, modes, done_m,
                                  suffix=RATED_SUFFIX, only=RATED_ROLES,
                                  extra_flags={"rated": True, "ratings": {str(k): v for k, v in sorted(ratings.items())},
                                               "rating_beta": float(modes.rating_beta)})
    return {"predicted": n, "stage": stage, "date": str(d)}


def honmei_context(s, r: Race, prm: ModeParams, role: str = "honmei") -> tuple[int, int]:
    """本命10点モードの（残り枠、この先に残る成立見込みレース数）。

    **先読みは一切使わない。** 分かっているのは「今日ここまでに何本買ったか」と
    「番組表の上で、この先に何レース残っているか」だけ。成立するかはオッズ次第なので、
    過去の成立率（honmei_feasible_rate）を掛けて見積もる（`online5.md` R3）。"""
    bought = s.execute(
        select(func.count(Prediction.id)).join(Race, Race.id == Prediction.race_id).where(
            Race.race_date == r.race_date, Prediction.role == role, Prediction.decision == "buy")
    ).scalar() or 0
    remaining = s.execute(
        select(func.count(Race.id)).where(
            Race.race_date == r.race_date, Race.closed_at.is_not(None),
            Race.closed_at >= r.closed_at)
    ).scalar() or 1
    slots_left = max(0, int(prm.honmei_slots) - int(bought))
    races_left = max(1, int(round(int(remaining) * float(prm.honmei_feasible_rate))))
    return slots_left, races_left


def _record_modes(s, o: dict, r: Race, model_version: str, settings_id: int, now: datetime,
                  odds_by_race: dict, prm: ModeParams, done_m: dict, suffix: str = "", only=None,
                  extra_flags: dict | None = None) -> None:
    """4モードを role='katai_t'/'honmei'/'place'/'ev1' として predictions に追記する（追記専用・自動購入なし）。

    複勝単勝は市場のオッズだけで選ぶので、公式オッズが無い（推定）ときは skip として記録する。
    suffix/only/extra_flags は展示評価を加味した版（role 末尾 R）用。flags.mode は常に元のモード名、
    flags.role に実際の role を入れる（採点はモード名で分岐する）。"""
    rid = o["race_id"]
    only = tuple(only) if only else MODE_ROLES
    extra_flags = extra_flags or {}
    real = not bool(o["flags"].get("odds_estimated"))
    oarr = np.array([np.nan if o["odds_used"][k] is None else float(o["odds_used"][k]) for k in _PL])
    main_idx = [combo_index(x["combo"]) for x in sorted(o["selections"], key=lambda x: x["rank"]) if x["kind"] == "main"]
    # 荒れ理由タグ・堅い注意タグ（選定には使わない。表示と前向き検証のため。市場ベースのタグは実オッズのときだけ）
    q_mkt = market_probs(oarr) if real else None
    tags = race_tags(o["boat_eval"], r.stadium_code, getattr(r, "title", None), getattr(r, "race_type", None), q_mkt)
    # 穴狙いの記録は止めたが「市場が荒れると見ているか」（万舟確率）は全モードに残す（2026-09-23）
    q_man = float(np.where(q_mkt <= Q_MAN, q_mkt, 0.0).sum()) if q_mkt is not None else None
    rough_market = bool(q_man is not None and q_man >= float(prm.ana_qman_min))
    common = dict(race_id=rid, model_version=model_version, settings_id=settings_id, stage="final",
                  created_at=now_jst(), asof_ts=now, post_time_at_pred=r.closed_at, features_used=None,
                  odds_snapshot_id=odds_by_race.get(rid, (None,))[0], completeness=o["completeness"],
                  missing_fields=None, boat_eval=o["boat_eval"], probs=o["probs"], odds_used=o["odds_used"],
                  ev=o["ev"], confidence=o["confidence"], rationale=o["rationale"], input_hash=o["input_hash"])

    def _add(mode: str, fired: bool, reason: str | None, flags: dict, sels: list[dict], text: str, er: float = 0.0):
        role = mode + suffix
        p = Prediction(**common, role=role, expected_return=er, decision=("buy" if fired else "skip"),
                       skip_reason=(None if fired else reason),
                       flags={**o["flags"], "mode": mode, "role": role, "modes_version": MODES_VERSION, "params": prm.to_dict(),
                              "tags": tags["rough"], "solid_tags": tags["solid"], "q_man": q_man, "rough_market": rough_market,
                              **extra_flags, **flags},
                       rationale_text=(("◎×加味: " if suffix else "") + text))
        s.add(p)
        s.flush()
        for k, x in enumerate(sels):
            s.add(PredictionSelection(prediction_id=p.id, combo=x["combo"], rank=k + 1, kind=x["kind"],
                                      stake=int(x["stake"]), prob=x.get("prob"), odds_at_pred=x.get("odds"),
                                      odds_source=o["odds_source"], ev=None))

    if "katai_t" in only and rid not in done_m["katai_t" + suffix]:
        kt = select_katai_top(main_idx, oarr, float(o["confidence"]), prm)
        _add("katai_t", kt["fired"], kt["reason"],
             dict(n_points=len(kt["points"]), stake_total=int(kt.get("stake_total", 0))),
             [dict(combo=_PL[j], kind="katai_t", stake=st, prob=float(o["probs"][_PL[j]]), odds=od)
              for j, st, od in zip(kt["points"], kt["stakes"], kt.get("odds", []))],
             (f"堅い予想（上位厚め）: 本線{len(kt['points'])}点・{kt.get('stake_total', 0)}円（1点目に厚く）"
              if kt["fired"] else f"堅い予想（上位厚め）: 見送り（{kt['reason']}）"))
    if "honmei" in only and rid not in done_m["honmei" + suffix]:
        if real:
            slots_left, races_left = honmei_context(s, r, prm, role="honmei" + suffix)
            hm = select_honmei(main_idx, oarr, o["probs"], slots_left, races_left, prm)
        else:
            hm = dict(fired=False, reason="odds_estimated", points=[], stakes=[])
        _add("honmei", hm["fired"], hm["reason"],
             dict(n_points=len(hm["points"]), stake_total=int(hm.get("stake_total", 0)),
                  min_payout=hm.get("min_payout"), mult=hm.get("mult"), p10=hm.get("p10"),
                  threshold=hm.get("threshold"), slots_left=hm.get("slots_left"), races_left=hm.get("races_left")),
             [dict(combo=_PL[j], kind="honmei", stake=st, prob=float(o["probs"][_PL[j]]), odds=od)
              for j, st, od in zip(hm["points"], hm["stakes"], hm.get("odds", []))],
             (f"本命{len(hm['points'])}点: {hm.get('stake_total', 0)}円（当たれば{hm.get('min_payout')}円以上＝"
              f"×{hm.get('mult', 0):.2f}）／確率合計{hm.get('p10', 0):.3f} ≥ しきい値{hm.get('threshold') or 0:.3f}"
              if hm["fired"] else f"本命10点: 見送り（{hm['reason']}）"))
    if "place" in only and rid not in done_m["place" + suffix]:
        pl = select_place(oarr, prm) if real else dict(fukusho=dict(fired=False, reason="odds_estimated"),
                                                      tansho=dict(fired=False, reason="odds_estimated"))
        fk, tn = pl["fukusho"], pl["tansho"]
        fired = bool(fk["fired"] or tn["fired"])
        # 発火した券種だけを買い目にする。両方とも見送りなら「買うならこれ」として両方を参考記録
        # combo は主キーの一部なので複勝・単勝で同じ艇でもぶつからないよう「複1」「単1」の形にする
        sels = ([dict(combo=f"複{fk['lane']}", kind="fukusho", stake=fk["stake"], prob=fk["q"], odds=None)] if (fk["fired"] or not fired) and "lane" in fk else []) + \
               ([dict(combo=f"単{tn['lane']}", kind="tansho", stake=tn["stake"], prob=tn["q"], odds=None)] if (tn["fired"] or not fired) and "lane" in tn else [])
        _add("place", fired, (None if fired else (fk.get("reason") or "q_low")),
             dict(fukusho=fk, tansho=tn, n_points=len(sels), stake_total=int(sum(x["stake"] for x in sels))),
             sels,
             ("複勝・単勝: " + "、".join(
                 ([f"複勝 {fk['lane']}号艇（市場の2着以内確率 {fk['q']:.3f}）"] if fk["fired"] else []) +
                 ([f"単勝 {tn['lane']}号艇（市場の1着確率 {tn['q']:.3f}）"] if tn["fired"] else []))
              if fired else f"複勝・単勝: 見送り（{fk.get('reason') or 'q_low'}）"))
    if "ev1" in only and rid not in done_m["ev1" + suffix]:
        # 期待値1以上: 実績規則（オッズ不要）＋較正期待値（実オッズのときだけ）
        parr = np.array([float(o["probs"][k]) for k in _PL])
        e1 = select_ev1(o["boat_eval"], r.stadium_code, oarr if real else None, parr, prm)
        ru, ca = e1["rule"], e1["cal"]
        # 買い目 = 発火した側だけ。両方見送りのときだけ、較正の最大期待値の1点を「買うならこれ」として残す
        # （見送りの仮想採点用）。発火した行に参考点を混ぜると投資に数えられて採点が狂う（出荷前チェックで発見）。
        sels = ([dict(combo="複1", kind="fukusho", stake=ru["stake"], prob=None, odds=None)] if ru["fired"] else []) + \
               ([dict(combo=_PL[j], kind="ev3t", stake=st, prob=pc, odds=od)
                 for j, st, pc, od in zip(ca["points"], ca["stakes"], ca["p_cal"], ca["odds"])]
                if (ca["fired"] or not e1["fired"]) else [])
        _add("ev1", e1["fired"], e1["reason"],
             dict(rule=ru, cal={k: v for k, v in ca.items() if k not in ("points", "stakes", "ev", "odds", "p_cal")},
                  cal_ev=ca["ev"], n_points=len(sels), stake_total=int(sum(x["stake"] for x in sels))),
             sels,
             ("期待値1以上: " + "、".join(
                 ([f"複勝 1号艇（モーター2連率1位・展示タイム1位・上位8場）"] if ru["fired"] else []) +
                 ([f"3連単 {len(ca['points'])}点（較正期待値 {max(ca['ev']):.2f}）"] if ca["fired"] else []))
              if e1["fired"] else
              f"期待値1以上: 見送り（{e1['reason']}"
              + (f"／較正期待値の最大 {ca['max_ev']:.2f}" if ca.get("max_ev") is not None else "／較正期待値は実オッズ待ち") + "）"))


# ---------------------------------------------------------------- score
def _payout_amount(payouts: dict | None, key: str, lane: int) -> float:
    """公式の確定配当（100円あたり）。当たっていなければ 0。"""
    for e in (payouts or {}).get(key) or []:
        try:
            if int(str(e.get("combination", "")).strip()) == lane:
                return float(e.get("amount") or 0)
        except Exception:
            continue
    return 0.0


def score_place(sels, payouts: dict | None, refund_lanes: list, cancelled: bool = False,
                has_result: bool = True) -> dict:
    """複勝・単勝の採点。kind='fukusho' は payouts['place']、'tansho' は payouts['win']。
    返還艇の買い目は投資から除外。"""
    if cancelled or not has_result or not isinstance(payouts, dict):
        return dict(valid=False, hit=None, hit_kind=None, stake_total=0, payout_total=0, pnl=0,
                    refunded_points=0, refunded_stake=0)
    refund = {int(l) for l in (refund_lanes or [])}
    stake_total = payout_total = ref_pts = ref_stake = 0
    hit_kind = None
    for x in sels:
        lane = int(re.sub(r"\D", "", str(x.combo)))        # 「複1」「単1」→ 1
        if lane in refund:
            ref_pts += 1
            ref_stake += int(x.stake)
            continue
        stake_total += int(x.stake)
        amt = _payout_amount(payouts, "place" if x.kind == "fukusho" else "win", lane)
        if amt > 0:
            payout_total += int(round(amt * int(x.stake) / 100))
            hit_kind = hit_kind or x.kind
    return dict(valid=True, hit=payout_total > 0, hit_kind=hit_kind, stake_total=stake_total,
                payout_total=payout_total, pnl=payout_total - stake_total,
                refunded_points=ref_pts, refunded_stake=ref_stake)


def score_ev1(sels, res, r, tri: int | None, decision: str) -> dict:
    """期待値1以上モードの採点。複勝（kind=fukusho）と3連単（kind=ev3t）が混ざる。
    見送り行では規則側の複勝は入っていないので、較正側の参考1点だけが仮想採点される。"""
    fk = [x for x in sels if x.kind == "fukusho"]
    t3 = [x for x in sels if x.kind == "ev3t"]
    cancelled = r.status == "cancelled"
    a = score_place(fk, res.payouts, res.refunds or [], cancelled=cancelled, has_result=bool(res.trifecta)) if fk else None
    b = None
    if t3:
        b = score_race([combo_index(x.combo) for x in t3], [], -1 if tri is None else tri, res.trifecta_payout,
                       res.refunds or [], t3[0].stake, cancelled=cancelled, stakes=[int(x.stake) for x in t3])
    parts = [x for x in (a, b) if x is not None]
    if not parts or not all(x["valid"] for x in parts):
        return dict(valid=False, hit=None, hit_kind=None, stake_total=0, payout_total=0, pnl=0,
                    refunded_points=0, refunded_stake=0)
    hit_kind = "fukusho" if (a and a["hit"]) else ("ev3t" if (b and b["hit"]) else None)
    tot = {k: sum(int(x[k]) for x in parts) for k in ("stake_total", "payout_total", "refunded_points", "refunded_stake")}
    return dict(valid=True, hit=hit_kind is not None, hit_kind=hit_kind, pnl=tot["payout_total"] - tot["stake_total"], **tot)


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
            mode = (p.flags or {}).get("mode")
            if mode == "place":
                sc = score_place(sels, res.payouts, res.refunds or [], cancelled=(r.status == "cancelled"),
                                 has_result=bool(res.trifecta))
            elif mode == "ev1":
                sc = score_ev1(sels, res, r, tri, p.decision)
            else:
                sel_idx = [combo_index(x.combo) for x in sels]
                main_idx = [combo_index(x.combo) for x in sels if x.kind == "main"]
                stakes = [int(x.stake) for x in sels]
                sc = score_race(sel_idx, main_idx, -1 if tri is None else tri, res.trifecta_payout, res.refunds or [],
                                sels[0].stake if sels else 200, cancelled=(r.status == "cancelled"), stakes=stakes or None)
                if mode in ("ana", "katai", "katai_t", "honmei") and sc.get("hit"):
                    sc["hit_kind"] = mode
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


# ---------------------------------------------------------------- 単勝・複勝プールの歪み（観測）
def poolgap_from_settings(row: SettingsVersion) -> "PoolGapParams":
    """設定の extra.poolgap。無ければ既定値（2026年1〜8月の確定オッズ検証で決めた値）。"""
    from boatlab.research.poolgap import PoolGapParams
    d = (row.extra or {}).get("poolgap") or {}
    return PoolGapParams(**{k: v for k, v in d.items() if k in PoolGapParams().__dict__})


def record_pool_gap(odds3t: dict, win_odds: dict | None, place_odds: dict | None,
                    race_id: int, stage: str, minutes_before: float | None) -> int:
    """締切前オッズで見えた候補を追記する（購入はしない・仮想の記録のみ）。戻り値は追記件数。"""
    from sqlalchemy.exc import IntegrityError

    from boatlab.research.poolgap import PARAMS_VERSION, find_picks
    with session_scope() as s:
        prm = poolgap_from_settings(_settings_row(s))
    picks = find_picks(odds3t, win_odds, place_odds, prm)
    n = 0
    for p in picks:
        try:
            with session_scope() as s:
                s.add(PoolGapPick(race_id=race_id, bet_type=p["bet_type"], lane=p["lane"], stage=stage,
                                  created_at=now_jst(), minutes_before=minutes_before, odds_seen=p["odds"],
                                  p_pool=p["p_pool"], p_ref=p["p_ref"], ratio=p["ratio"], stake=prm.stake,
                                  params_version=PARAMS_VERSION, params=prm.to_dict()))
            n += 1
        except IntegrityError:
            pass          # 同じレース・券種・艇・段階は1回だけ（再実行しても増えない）
    return n


# ---------------------------------------------------------------- 直前版（締切2〜4分前のオッズで穴・複勝単勝を再選定）
