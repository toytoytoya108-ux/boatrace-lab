"""FastAPI：PWA 用 JSON API（docs/06 §4）。単一ユーザー認証（パスワード + 署名 Cookie）。"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from datetime import date, timedelta
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


@app.get("/api/today")
def today(d: str | None = None, _=Depends(require_auth)):
    day = date.fromisoformat(d) if d else now_jst().date()
    mv = _active_version()
    races = _q("""
        SELECT r.id, r.stadium_code, r.race_no, r.closed_at, r.grade, r.race_type, r.status,
               res.trifecta, res.trifecta_payout,
               p.id AS prediction_id, p.stage, p.confidence, p.expected_return, p.decision, p.skip_reason, p.completeness, p.flags,
               sc.hit, sc.hit_kind, sc.pnl, sc.roi, sc.valid, sc.stake_total, sc.payout_total, sc.category
        FROM races r
        LEFT JOIN results res ON res.race_id = r.id
        LEFT JOIN predictions p ON p.id = (
            SELECT p2.id FROM predictions p2 WHERE p2.race_id = r.id AND p2.role='active'
              AND (:mv IS NULL OR p2.model_version = :mv)
            ORDER BY CASE p2.stage WHEN 'final' THEN 0 ELSE 1 END, p2.created_at DESC LIMIT 1)
        LEFT JOIN scoring sc ON sc.prediction_id = p.id
        WHERE r.race_date = :d ORDER BY r.closed_at, r.stadium_code, r.race_no""", d=str(day), mv=mv)
    for r in races:
        r["stadium"] = STADIUMS.get(r["stadium_code"])
    buys = [r for r in races if r["decision"] == "buy"]
    scored = [r for r in races if r["valid"]]
    day_pnl = {"n": len(scored), "stake": sum(r["stake_total"] or 0 for r in scored if r["decision"] == "buy"),
               "payout": sum(r["payout_total"] or 0 for r in scored if r["decision"] == "buy"),
               "hits": sum(1 for r in scored if r["decision"] == "buy" and r["hit"]),
               "virtual_stake": sum(r["stake_total"] or 0 for r in scored), "virtual_payout": sum(r["payout_total"] or 0 for r in scored),
               "virtual_hits": sum(1 for r in scored if r["hit"])}
    return {"date": str(day), "active_model": mv, "n_races": len(races), "n_predicted": sum(1 for r in races if r["prediction_id"]),
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


@app.get("/api/stats")
def stats(range: str = "all", model: str | None = None, _=Depends(require_auth)):
    df = perf.load_scored(model)
    today_ = now_jst().date()
    since = {"d": today_, "w": today_ - timedelta(days=7), "m": today_.replace(day=1), "all": None}.get(range)
    return perf.summary(df, since=since)


@app.get("/api/stats/breakdown")
def stats_breakdown(by: str = "stadium", decision: str | None = "buy", model: str | None = None, _=Depends(require_auth)):
    df = perf.load_scored(model)
    if by not in ("stadium", "grade", "month", "odds_band", "confidence_band", "kind", "model_version"):
        raise HTTPException(400, "bad 'by'")
    out = perf.breakdown(df, by, None if decision in (None, "all") else decision)
    return json.loads(out.to_json(orient="records", force_ascii=False)) if len(out) else []


@app.get("/api/stats/calibration")
def stats_calibration(model: str | None = None, _=Depends(require_auth)):
    df = perf.load_scored(model)
    t = perf.calibration_check(df)
    t["band"] = t["band"].astype(str)
    return json.loads(t.to_json(orient="records"))


@app.get("/api/stats/misses")
def stats_misses(model: str | None = None, _=Depends(require_auth)):
    return perf.miss_analysis(perf.load_scored(model))


@app.get("/api/readiness")
def readiness(_=Depends(require_auth)):
    st = _settings()
    df = perf.load_scored()
    out = perf.readiness(df, st["readiness"])
    out["by_model"] = json.loads(perf.breakdown(df, "model_version").to_json(orient="records")) if len(df) else []
    return out


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
