"""FastAPI：PWA 用 JSON API（docs/06 §4）。単一ユーザー認証（パスワード + 署名 Cookie）。"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, text

from boatlab.analytics import performance as perf
from boatlab.config import STADIUMS
from boatlab.store.db import get_engine, init_db, session_scope
from boatlab.store.models import JobRun, ModelVersion, SettingsVersion
from boatlab.util import now_jst

SECRET = os.environ.get("BOATLAB_SECRET", "change-me")
PASSWORD = os.environ.get("BOATLAB_PASSWORD", "")
WEB_DIR = Path(os.environ.get("BOATLAB_WEB_DIR", Path(__file__).resolve().parents[2] / "web"))

app = FastAPI(title="boatlab", docs_url=None, redoc_url=None)


# ---------------------------------------------------------------- auth
def _token(ts: int) -> str:
    return f"{ts}.{hmac.new(SECRET.encode(), str(ts).encode(), hashlib.sha256).hexdigest()}"


def require_auth(request: Request):
    if not PASSWORD:  # パスワード未設定＝ローカル開発
        return True
    tok = request.cookies.get("bl_session", "")
    try:
        ts, sig = tok.split(".")
        if hmac.compare_digest(sig, _token(int(ts)).split(".")[1]) and time.time() - int(ts) < 180 * 86400:
            return True
    except Exception:
        pass
    raise HTTPException(401, "login required")


@app.post("/api/login")
def login(body: dict, response: Response):
    if not PASSWORD or hmac.compare_digest(body.get("password", ""), PASSWORD):
        response.set_cookie("bl_session", _token(int(time.time())), max_age=180 * 86400, httponly=True, samesite="lax",
                            secure=os.environ.get("BOATLAB_INSECURE_COOKIE", "") == "")
        return {"ok": True}
    raise HTTPException(401, "wrong password")


# ---------------------------------------------------------------- helpers
def _q(sql: str, **params) -> list[dict]:
    with get_engine().connect() as c:
        rows = c.execute(text(sql), params).mappings().all()
    out = []
    for r in rows:
        d = dict(r)
        for k, v in d.items():
            if isinstance(v, str) and k in ("probs", "odds_used", "ev", "boat_eval", "rationale", "flags", "payouts", "refunds", "features_used", "params", "readiness", "summary", "odds"):
                try:
                    d[k] = json.loads(v)
                except Exception:
                    pass
        out.append(d)
    return out


ROLE_OF_MODE = {"std": "active", "focused": "focused"}


def _role(mode: str | None) -> str:
    return ROLE_OF_MODE.get(mode or "std", "active")


def _active_version() -> str | None:
    with session_scope() as s:
        mv = s.execute(select(ModelVersion).where(ModelVersion.status == "active")).scalars().first()
        return mv.version if mv else None


def _settings() -> dict:
    with session_scope() as s:
        row = s.execute(select(SettingsVersion).order_by(SettingsVersion.id.desc())).scalars().first()
        if row is None:
            row = SettingsVersion()
            s.add(row)
            s.flush()
        return {c.name: getattr(row, c.name) for c in SettingsVersion.__table__.columns}


# ---------------------------------------------------------------- endpoints
@app.get("/api/health")
def health(_=Depends(require_auth)):
    jobs = _q("SELECT job, started_at, finished_at, ok, error FROM job_run ORDER BY id DESC LIMIT 20")
    fails = _q("SELECT source, COUNT(*) AS n FROM fetch_log WHERE ok=0 AND started_at >= :d GROUP BY source",
               d=str(now_jst() - timedelta(days=7)))
    return {"now": now_jst().isoformat(), "active_model": _active_version(), "jobs": jobs, "fetch_failures_7d": fails}


def _freshness(day) -> dict:
    """画面用の更新時刻: 最終取込（morning/intraday の完了）・最終予想保存・進行中ジョブ・停滞判定。"""
    last_ok = _q("SELECT job, finished_at FROM job_run WHERE job IN ('morning','intraday') AND ok=1 ORDER BY id DESC LIMIT 1")
    # finished_at が NULL のまま残るのは、途中でプロセスが落ちた場合も同じ。2時間以上前のものは
    # 「実行中」ではなく「中断」として扱う（過去の異常終了がいつまでも実行中に見えるのを防ぐ）
    running = _q("SELECT job, started_at FROM job_run WHERE finished_at IS NULL AND ok IS NULL "
                 "AND started_at >= :t ORDER BY id DESC LIMIT 1", t=str(now_jst() - timedelta(hours=2)))
    stuck = _q("SELECT job, started_at FROM job_run WHERE finished_at IS NULL AND ok IS NULL "
               "AND started_at < :t ORDER BY id DESC LIMIT 1", t=str(now_jst() - timedelta(hours=2)))
    last_pred = _q("SELECT MAX(p.created_at) AS t FROM predictions p JOIN races r ON r.id=p.race_id WHERE r.race_date=:d", d=str(day))
    now = now_jst()
    ingest_at = last_ok[0]["finished_at"] if last_ok else None
    hm = now.strftime("%H:%M")
    racing = "07:55" <= hm <= "21:50"
    stale_min = None
    if ingest_at:
        try:
            stale_min = int((now.replace(tzinfo=None) - datetime.fromisoformat(str(ingest_at))).total_seconds() // 60)
        except Exception:
            stale_min = None
    return {"now": now.isoformat(timespec="minutes"), "ingest_at": ingest_at, "predict_at": last_pred[0]["t"] if last_pred else None,
            "running": running[0] if running else None, "stuck": stuck[0] if stuck else None, "racing_hours": racing,
            "stale": bool(racing and day == now.date() and (stale_min is None or stale_min > 15)), "stale_min": stale_min}


@app.get("/api/today")
def today(d: str | None = None, mode: str | None = None, stadium: int | None = None, _=Depends(require_auth)):
    day = date.fromisoformat(d) if d else now_jst().date()
    mv = _active_version()
    role = _role(mode)
    races = _q("""
        SELECT r.id, r.stadium_code, r.race_no, r.closed_at, r.grade, r.race_type, r.status,
               res.trifecta, res.trifecta_payout,
               p.id AS prediction_id, p.stage, p.role AS pred_role, p.confidence, p.expected_return, p.decision, p.skip_reason, p.completeness, p.flags,
               (SELECT COUNT(*) FROM prediction_selections ps WHERE ps.prediction_id = p.id) AS n_points,
               (SELECT SUM(ps.stake) FROM prediction_selections ps WHERE ps.prediction_id = p.id) AS stake_plan,
               sc.hit, sc.hit_kind, sc.pnl, sc.roi, sc.valid, sc.stake_total, sc.payout_total, sc.category
        FROM races r
        LEFT JOIN results res ON res.race_id = r.id
        LEFT JOIN predictions p ON p.id = (
            SELECT p2.id FROM predictions p2 WHERE p2.race_id = r.id
              AND (p2.role = :role OR (p2.role = 'active' AND p2.stage = 'program'))
              AND (:mv IS NULL OR p2.model_version = :mv)
            ORDER BY CASE WHEN p2.role = :role THEN 0 ELSE 1 END, CASE p2.stage WHEN 'final' THEN 0 ELSE 1 END, p2.created_at DESC LIMIT 1)
        LEFT JOIN scoring sc ON sc.prediction_id = p.id
        WHERE r.race_date = :d AND (:st IS NULL OR r.stadium_code = :st)
        ORDER BY r.closed_at, r.stadium_code, r.race_no""", d=str(day), mv=mv, role=role, st=stadium)
    for r in races:
        r["stadium"] = STADIUMS.get(r["stadium_code"])
        # 絞り込み型は確定予想（締切直前）しか保存しないため、それまでは15点固定側の暫定予想を「暫定」として見せる
        r["provisional"] = bool(r["prediction_id"]) and r["pred_role"] != role
        if r["provisional"]:
            r["decision"] = None
            r["skip_reason"] = None
            r["n_points"] = None
            r["stake_plan"] = None
    buys = [r for r in races if r["decision"] == "buy"]
    scored = [r for r in races if r["valid"]]
    day_pnl = {"n": len(scored), "stake": sum(r["stake_total"] or 0 for r in scored if r["decision"] == "buy"),
               "payout": sum(r["payout_total"] or 0 for r in scored if r["decision"] == "buy"),
               "hits": sum(1 for r in scored if r["decision"] == "buy" and r["hit"]),
               "virtual_stake": sum(r["stake_total"] or 0 for r in scored), "virtual_payout": sum(r["payout_total"] or 0 for r in scored),
               "virtual_hits": sum(1 for r in scored if r["hit"])}
    return {"date": str(day), "mode": mode or "std", "active_model": mv, "updated": _freshness(day), "n_races": len(races), "n_predicted": sum(1 for r in races if r["prediction_id"] and not r["provisional"]), "n_provisional": sum(1 for r in races if r["provisional"]),
            "n_buy": len(buys), "n_skip": sum(1 for r in races if r["decision"] == "skip"), "day": day_pnl,
            "top": sorted(buys, key=lambda r: -(r["confidence"] or 0) * (r["expected_return"] or 0))[:5], "races": races}


@app.get("/api/races/{race_id}")
def race_detail(race_id: int, _=Depends(require_auth)):
    race = _q("SELECT r.*, res.trifecta, res.trifecta_payout, res.kimarite, res.payouts, res.refunds, res.is_irregular FROM races r LEFT JOIN results res ON res.race_id=r.id WHERE r.id=:id", id=race_id)
    if not race:
        raise HTTPException(404)
    race = race[0]
    race["stadium"] = STADIUMS.get(race["stadium_code"])
    entries = _q("SELECT * FROM entries WHERE race_id=:id ORDER BY lane", id=race_id)
    previews = _q("SELECT * FROM preview_snapshots WHERE race_id=:id ORDER BY fetched_at DESC, lane", id=race_id)
    latest_prev = {}
    for p in previews:
        latest_prev.setdefault(p["lane"], p)
    cond = _q("SELECT * FROM race_conditions WHERE race_id=:id ORDER BY observed_at DESC LIMIT 1", id=race_id)
    result_entries = _q("SELECT * FROM result_entries WHERE race_id=:id ORDER BY lane", id=race_id)
    preds = _q("SELECT * FROM predictions WHERE race_id=:id ORDER BY created_at DESC", id=race_id)
    for p in preds:
        p["selections"] = _q("SELECT * FROM prediction_selections WHERE prediction_id=:pid ORDER BY rank", pid=p["id"])
        sc = _q("SELECT * FROM scoring WHERE prediction_id=:pid", pid=p["id"])
        p["scoring"] = sc[0] if sc else None
        p.pop("features_used", None)
    odds = _q("SELECT id, captured_at, source, odds FROM odds_snapshots WHERE race_id=:id AND bet_type='3t' ORDER BY captured_at DESC LIMIT 3", id=race_id)
    return {"race": race, "entries": entries, "previews": [latest_prev[k] for k in sorted(latest_prev)],
            "conditions": cond[0] if cond else None, "result_entries": result_entries, "predictions": preds, "odds": odds}


def _scored(model, mode, stadium=None):
    df = perf.load_scored(model, role=_role(mode))
    if stadium and len(df):
        df = df[df["stadium_code"] == int(stadium)]
    return df


@app.get("/api/stats")
def stats(range: str = "all", model: str | None = None, mode: str | None = None, stadium: int | None = None, _=Depends(require_auth)):
    df = _scored(model, mode, stadium)
    today_ = now_jst().date()
    since = {"d": today_, "w": today_ - timedelta(days=7), "m": today_.replace(day=1), "all": None}.get(range)
    return perf.summary(df, since=since)


@app.get("/api/stats/breakdown")
def stats_breakdown(by: str = "stadium", decision: str | None = "buy", model: str | None = None, mode: str | None = None, stadium: int | None = None, _=Depends(require_auth)):
    df = _scored(model, mode, stadium)
    if by not in ("stadium", "grade", "month", "odds_band", "confidence_band", "kind", "model_version"):
        raise HTTPException(400, "bad 'by'")
    out = perf.breakdown(df, by, None if decision in (None, "all") else decision)
    return json.loads(out.to_json(orient="records", force_ascii=False)) if len(out) else []


@app.get("/api/stats/calibration")
def stats_calibration(model: str | None = None, mode: str | None = None, _=Depends(require_auth)):
    df = _scored(model, mode)
    t = perf.calibration_check(df)
    t["band"] = t["band"].astype(str)
    return json.loads(t.to_json(orient="records"))


@app.get("/api/stats/misses")
def stats_misses(model: str | None = None, mode: str | None = None, _=Depends(require_auth)):
    return perf.miss_analysis(_scored(model, mode))


@app.get("/api/readiness")
def readiness(mode: str | None = None, _=Depends(require_auth)):
    st = _settings()
    df = _scored(None, mode)
    if (mode or "std") == "focused":
        th = {**perf.FOCUSED_READINESS_DEFAULT, **((st.get("extra") or {}).get("readiness_focused") or {})}
    else:
        th = st["readiness"]
    out = perf.readiness(df, th)
    out["mode"] = mode or "std"
    out["thresholds"] = th
    out["by_model"] = json.loads(perf.breakdown(df, "model_version").to_json(orient="records")) if len(df) else []
    return out


@app.get("/api/export.csv")
def export_csv(mode: str | None = None, _=Depends(require_auth)):
    """採点済み予想の一覧を CSV で書き出す（分析用。スマホの「ファイル」に保存してチャットに添付できる）。"""
    from fastapi.responses import Response
    df = perf.load_scored(None, role=_role(mode))
    if len(df):
        df = df.drop(columns=[c for c in ("flags",) if c in df.columns])
    body = df.to_csv(index=False)
    fn = f"boatlab_{mode or 'std'}_{now_jst():%Y%m%d}.csv"
    return Response(content=body, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{fn}"'})


@app.get("/api/stadiums")
def stadiums(_=Depends(require_auth)):
    return [{"code": k, "name": v} for k, v in sorted(STADIUMS.items())]


@app.get("/api/models")
def models(_=Depends(require_auth)):
    return _q("SELECT version, created_at, parent_version, description, status, params, trained_until, backtest_summary FROM model_versions ORDER BY created_at DESC")


@app.post("/api/models/{version}/activate")
def activate(version: str, _=Depends(require_auth)):
    with session_scope() as s:
        target = s.get(ModelVersion, version)
        if target is None:
            raise HTTPException(404)
        for mv in s.execute(select(ModelVersion)).scalars():
            if mv.version == version:
                mv.status = "active"
            elif mv.status == "active":
                mv.status = "retired"
    return {"active": version}


@app.post("/api/models/{version}/status")
def set_status(version: str, body: dict, _=Depends(require_auth)):
    """candidate ⇄ shadow ⇄ retired（active 化は /activate）。"""
    st = body.get("status")
    if st not in ("candidate", "shadow", "retired"):
        raise HTTPException(400, "status must be candidate/shadow/retired")
    with session_scope() as s:
        mv = s.get(ModelVersion, version)
        if mv is None:
            raise HTTPException(404)
        if mv.status == "active":
            raise HTTPException(400, "active model: activate another version first")
        mv.status = st
    return {"version": version, "status": st}


@app.get("/api/settings")
def get_settings(_=Depends(require_auth)):
    return _settings()


@app.put("/api/settings")
def put_settings(body: dict, _=Depends(require_auth)):
    cur = _settings()
    allowed = {"confidence_min", "ev_min", "completeness_min", "readiness", "extra"}
    with session_scope() as s:
        row = SettingsVersion(effective_from=now_jst(), **{k: body.get(k, cur[k]) for k in allowed},
                              points=cur["points"], main_points=cur["main_points"], stake_per_point=cur["stake_per_point"])
        s.add(row)
    return _settings()


# ---------------------------------------------------------------- 買い方の自動研究
_research_lock = __import__("threading").Lock()
_research_state = {"running": False, "started_at": None, "error": None}


def _run_research_bg():
    from boatlab.research.strategy import run_research
    try:
        run_research()
        _research_state["error"] = None
    except Exception as e:  # 画面に出す
        _research_state["error"] = repr(e)[:300]
    finally:
        _research_state["running"] = False


@app.get("/api/research")
def research(_=Depends(require_auth)):
    from boatlab.research.strategy import GRID, load_report
    rep = load_report()
    return {"report": rep, "grid_size": len(GRID), **_research_state}


@app.post("/api/research/run")
def research_run(_=Depends(require_auth)):
    import threading
    with _research_lock:
        if _research_state["running"]:
            return {"started": False, **_research_state}
        _research_state.update(running=True, started_at=now_jst().isoformat(timespec="minutes"), error=None)
        threading.Thread(target=_run_research_bg, daemon=True).start()
    return {"started": True, **_research_state}


@app.get("/api/poolgap")
def poolgap(day: str | None = None, _=Depends(require_auth)):
    """単勝・複勝プールの歪み（観測フェーズ）。実際の購入はしない・仮想の記録のみ。"""
    from boatlab.research.poolgap import PoolGapParams, report
    from boatlab.ops.daily import poolgap_from_settings
    from boatlab.store.models import SettingsVersion
    try:
        rep = report(day)
    except Exception as e:
        rep = {"n": 0, "today": [], "summary": [], "drift": None, "error": repr(e)[:200]}
    with session_scope() as s:
        row = s.execute(select(SettingsVersion).order_by(SettingsVersion.id.desc())).scalars().first()
        prm = poolgap_from_settings(row) if row else PoolGapParams()
    return {**rep, "params": prm.to_dict()}


_upstream_cache: dict = {"at": None, "value": None}


def _upstream_results(day) -> dict | None:
    """上流 today.json に「中身のある結果」が何レース分あるか。120秒キャッシュ。

    result ブロックは未確定でも空の入れ物として存在するため、着順か払戻があるものだけ数える。
    """
    now = now_jst()
    c = _upstream_cache
    if c["at"] and (now - c["at"]).total_seconds() < 120:
        return c["value"]
    val = None
    try:
        import httpx

        from boatlab.config import OPENAPI_API_TODAY
        doc = httpx.get(OPENAPI_API_TODAY, timeout=20).json()
        st = ((doc or {}).get("programs") or {}).get("stadiums") or {}
        n = filled = 0
        for s_key, sv in st.items():
            for _, r in (sv.get("races") or {}).items():
                if str(r.get("date") or day)[:10] != str(day):
                    continue
                n += 1
                rs = r.get("result") or {}
                pay = rs.get("payouts") or {}
                racers = rs.get("racers") or {}
                has_pay = any(v for v in pay.values())
                has_place = any((x or {}).get("place_number") for x in
                                (racers.values() if isinstance(racers, dict) else racers))
                if has_pay or has_place:
                    filled += 1
        val = {"races": n, "filled": filled}
    except Exception as e:
        val = {"error": repr(e)[:120]}
    c["at"], c["value"] = now, val
    return val


@app.get("/api/status")
def status(_=Depends(require_auth)):
    """システムの状態。「上流が遅れている」のか「取り込みが壊れている」のかを画面で判別できるようにする。"""
    day = now_jst().date()
    t = {"date": str(day)}
    row = _q("SELECT COUNT(*) AS n, SUM(CASE WHEN res.race_id IS NOT NULL THEN 1 ELSE 0 END) AS done, "
             "SUM(CASE WHEN r.closed_at < :now AND res.race_id IS NULL THEN 1 ELSE 0 END) AS waiting, "
             "SUM(CASE WHEN r.closed_at < :now THEN 1 ELSE 0 END) AS closed "
             "FROM races r LEFT JOIN results res ON res.race_id = r.id WHERE r.race_date = :d",
             d=str(day), now=str(now_jst().replace(tzinfo=None)))
    t.update(races=int(row[0]["n"] or 0), results=int(row[0]["done"] or 0), waiting=int(row[0]["waiting"] or 0))
    closed_so_far = int(row[0]["closed"] or 0)
    up = _upstream_results(day) if t["waiting"] else None
    t["upstream"] = up
    if not t["races"]:
        t["verdict"], t["note"] = "wait", "本日の出走表がまだ取り込まれていません"
    elif not t["waiting"]:
        t["verdict"], t["note"] = "ok", "締切を過ぎたレースの結果はすべて入っています"
    elif up and up.get("error"):
        t["verdict"], t["note"] = "unknown", f"上流を確認できませんでした（{up['error']}）"
    elif up and up.get("filled", 0) - t["results"] >= 3:
        t["verdict"] = "ng"
        t["note"] = f"上流には{up['filled']}件あるのにDBは{t['results']}件。取り込みに問題があります"
    else:
        t["verdict"] = "upstream"
        t["note"] = f"{t['waiting']}レースが結果待ち。上流がまだ結果を出していません（翌朝06:10の取込で埋まります）"

    o = {x["bet_type"]: int(x["n"]) for x in _q(
        "SELECT bet_type, COUNT(*) AS n FROM odds_snapshots WHERE source='official_web' "
        "AND race_id >= :lo AND race_id < :hi GROUP BY bet_type",
        lo=int(day.strftime("%Y%m%d")) * 10000, hi=(int(day.strftime("%Y%m%d")) + 1) * 10000)}
    odds = {"trifecta": o.get("3t", 0), "win": o.get("win", 0), "place": o.get("place", 0)}
    if closed_so_far < 3:
        odds["verdict"], odds["note"] = "wait", "取得は最初のレースの締切前から始まります"
    elif odds["trifecta"] and odds["win"]:
        odds["verdict"], odds["note"] = "ok", "3連単・単勝・複勝とも取得できています"
    else:
        miss = [n for n, k in (("3連単", "trifecta"), ("単勝", "win")) if not odds[k]]
        odds["verdict"] = "ng"
        odds["note"] = f"{'・'.join(miss)}のオッズが1件も取れていません（公式サイトの構造変更の可能性）"

    pg = _q("SELECT COUNT(*) AS n, COUNT(DISTINCT race_id) AS races FROM pool_gap_picks WHERE race_id >= :lo AND race_id < :hi",
            lo=int(day.strftime("%Y%m%d")) * 10000, hi=(int(day.strftime("%Y%m%d")) + 1) * 10000)
    pred = _q("SELECT COUNT(*) AS n, SUM(CASE WHEN p.flags LIKE '%odds_estimated%' THEN 1 ELSE 0 END) AS est "
              "FROM predictions p JOIN races r ON r.id=p.race_id "
              "WHERE r.race_date=:d AND p.stage='final' AND p.role='focused'", d=str(day))
    jobs = _q("SELECT job, started_at, finished_at, ok, substr(COALESCE(error,''),1,120) AS error "
              "FROM job_run ORDER BY id DESC LIMIT 12")
    fail = _q("SELECT source, COUNT(*) AS n FROM fetch_log WHERE ok=0 AND started_at >= :t GROUP BY source",
              t=str(now_jst() - timedelta(days=1)))
    return {"now": now_jst().isoformat(timespec="minutes"), "today": t, "odds": odds,
            "poolgap": {"picks": int(pg[0]["n"] or 0), "races": int(pg[0]["races"] or 0)},
            "predictions": {"final": int(pred[0]["n"] or 0), "estimated_odds": int(pred[0]["est"] or 0)},
            "jobs": jobs, "fetch_failures_24h": fail, "freshness": _freshness(day)}


@app.get("/api/backtests")
def backtests(_=Depends(require_auth)):
    root = Path("reports/backtest")
    out = {}
    for name in ("valid_model_compare.md", "valid_selection_sweep.md", "test_summary.md", "market_odds_eval.json"):
        p = root / name
        if p.exists():
            out[name] = p.read_text(encoding="utf-8")
    return out


# ---------------------------------------------------------------- static PWA
if WEB_DIR.exists():
    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/sw.js")
    def sw():
        return FileResponse(WEB_DIR / "sw.js", media_type="application/javascript")

    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.on_event("startup")
def _startup():
    init_db()
