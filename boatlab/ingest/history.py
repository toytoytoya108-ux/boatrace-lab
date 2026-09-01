"""過去データの一括取込（Open API v3: 2018〜 / turnmark v1: 2026〜）。冪等。"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Iterator

from sqlalchemy import func, select

from boatlab.config import OPENAPI_V3_BASE, TURNMARK_BASE
from boatlab.ingest.base import Fetcher, NotFound
from boatlab.ingest.parsers import parse_v1_day, parse_v3_day
from boatlab.store.db import session_scope
from boatlab.store.models import FetchLog, JobRun, Race, RawFile
from boatlab.store.writer import write_bundle
from boatlab.util import now_jst

log = logging.getLogger(__name__)
V3_REPOS = ("programs", "previews", "results")


def daterange(d0: date, d1: date) -> Iterator[date]:
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def _db_log_hook(rec: dict) -> None:
    with session_scope() as s:
        s.add(FetchLog(**rec))


def make_fetcher(log_to_db: bool = True) -> Fetcher:
    return Fetcher(log_hook=_db_log_hook if log_to_db else None)


def fetch_v3(fetcher: Fetcher, repo: str, d: date) -> tuple[dict | None, object | None]:
    ymd = d.strftime("%Y%m%d")
    url = OPENAPI_V3_BASE.format(repo=repo, yyyy=d.year, yyyymmdd=ymd)
    key = f"{repo}/{d.year}/{ymd}.json"
    try:
        return fetcher.fetch_json("openapi_v3", url, key)
    except NotFound:
        return None, None


def fetch_turnmark(fetcher: Fetcher, d: date):
    ymd = d.strftime("%Y%m%d")
    url = TURNMARK_BASE.format(yyyy=d.year, yyyymmdd=ymd)
    key = f"{d.year}/{ymd}.json"
    try:
        return fetcher.fetch_json("turnmark", url, key)
    except NotFound:
        return None, None


def _record_raw(session, res, source: str):
    if res is None:
        return
    from sqlalchemy.dialects import sqlite, postgresql
    tbl = RawFile.__table__
    ins = (postgresql.insert if session.get_bind().dialect.name == "postgresql" else sqlite.insert)(tbl)
    session.execute(ins.on_conflict_do_nothing(index_elements=["source", "key"]).values(
        source=source, key=res.key, sha256=res.sha256, fetched_at=res.fetched_at.replace(tzinfo=None),
        path=str(res.key)))


def ingest_v3_day(fetcher: Fetcher, d: date, force: bool = False) -> dict:
    """1日分（programs+previews+results）。既に DB にレースがあり force でなければ previews/results の欠けだけ補う。"""
    docs = {}
    ress = {}
    for repo in V3_REPOS:
        docs[repo], ress[repo] = fetch_v3(fetcher, repo, d)
    if not any(docs.values()):
        return {"date": str(d), "skipped": "no data"}
    bundle = parse_v3_day(d, docs["programs"], docs["previews"], docs["results"])
    with session_scope() as s:
        for repo in V3_REPOS:
            _record_raw(s, ress[repo], "openapi_v3")
        counts = write_bundle(s, bundle)
    out = {"date": str(d), **counts}
    out["races_in_file"] = len(bundle.races)
    out["results_in_file"] = len(bundle.results)
    return out


def ingest_turnmark_day(fetcher: Fetcher, d: date) -> dict:
    doc, res = fetch_turnmark(fetcher, d)
    if doc is None:
        return {"date": str(d), "skipped": "no data"}
    bundle = parse_v1_day(d, doc, source_prefix="turnmark")
    with session_scope() as s:
        _record_raw(s, res, "turnmark")
        counts = write_bundle(s, bundle)
    return {"date": str(d), **counts, "odds_in_file": len(bundle.odds)}


def ingest_range(d0: date, d1: date, sources: tuple[str, ...] = ("openapi_v3",), force: bool = False,
                 fetcher: Fetcher | None = None, progress_every: int = 30) -> dict:
    fetcher = fetcher or make_fetcher()
    started = now_jst()
    with session_scope() as s:
        job = JobRun(job=f"ingest_range:{','.join(sources)}:{d0}..{d1}", started_at=started)
        s.add(job)
        s.flush()
        job_id = job.id
    totals: dict[str, int] = {}
    days = 0
    errors: list[str] = []
    for d in daterange(d0, d1):
        try:
            if "openapi_v3" in sources:
                out = ingest_v3_day(fetcher, d, force=force)
                for k, v in out.items():
                    if isinstance(v, int):
                        totals[k] = totals.get(k, 0) + v
            if "turnmark" in sources and d >= date(2026, 1, 1):
                out = ingest_turnmark_day(fetcher, d)
                for k, v in out.items():
                    if isinstance(v, int):
                        totals["tm_" + k] = totals.get("tm_" + k, 0) + v
        except Exception as e:  # 1日失敗しても続行し、最後に報告
            log.exception("ingest failed for %s", d)
            errors.append(f"{d}: {e!r}")
        days += 1
        if days % progress_every == 0:
            log.info("progress %s days=%d totals=%s", d, days, totals)
    with session_scope() as s:
        job = s.get(JobRun, job_id)
        job.finished_at = now_jst()
        job.ok = not errors
        job.summary = {"days": days, **totals, "errors": errors[:50]}
        job.error = "\n".join(errors[:50]) or None
    return {"days": days, **totals, "errors": errors}


def coverage(session) -> list[tuple]:
    """年月ごとのレース数・結果数（品質レポート用）。"""
    q = (select(func.strftime("%Y-%m", Race.race_date).label("ym"), func.count(Race.id))
         .group_by("ym").order_by("ym"))
    return list(session.execute(q))
