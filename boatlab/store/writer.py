"""正規化レコードの書き込み（冪等）。

- races / entries: 初回値を固定（ON CONFLICT DO NOTHING）。races.status は結果到着時に更新。
- preview_snapshots / race_conditions / odds_snapshots: 追記（同一キーは無視）。
- results / result_entries: 公式訂正を反映できるよう UPSERT（採点は scoring に独自コピーを持つ）。
- racers: 初出・最終出走日を更新。
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from boatlab.util import now_jst

from sqlalchemy import select, update
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session

from boatlab.ingest.records import DayBundle
from boatlab.store.models import (
    Entry, OddsSnapshot, PreviewSnapshot, Race, RaceCondition, Racer, Result, ResultEntry,
)


def _insert(session: Session, table, rows: list[dict], conflict_cols: list[str], update_cols: list[str] | None = None):
    if not rows:
        return 0
    dialect = session.get_bind().dialect.name
    ins = (postgresql.insert if dialect == "postgresql" else sqlite.insert)(table)
    if update_cols:
        stmt = ins.on_conflict_do_update(index_elements=conflict_cols,
                                         set_={c: getattr(ins.excluded, c) for c in update_cols})
    else:
        stmt = ins.on_conflict_do_nothing(index_elements=conflict_cols)
    # executemany（SQLite は 1 文ずつだが十分速い）
    session.execute(stmt, rows)
    return len(rows)


def write_bundle(session: Session, b: DayBundle, now: datetime | None = None) -> Counter:
    now = now or now_jst()
    c: Counter = Counter()

    races = [dict(id=r.race_id, race_date=r.race_date, stadium_code=r.stadium_code, race_no=r.race_no,
                  closed_at=r.closed_at, grade=r.grade, title=r.title, race_type=r.race_type,
                  distance_m=r.distance_m, day_no=r.day_no, status="scheduled", source=r.source, updated_at=now)
             for r in b.races]
    c["races"] += _insert(session, Race.__table__, races, ["id"])
    # closed_at が欠けていた行は後から補完できるようにする（値は上書きしない）
    for r in b.races:
        if r.closed_at is not None:
            session.execute(update(Race).where(Race.id == r.race_id, Race.closed_at.is_(None))
                            .values(closed_at=r.closed_at))

    entries = [dict(e.model_dump(exclude={"name"}), name=e.name, program_fetched_at=now) for e in b.entries]
    c["entries"] += _insert(session, Entry.__table__, entries, ["race_id", "lane"])

    # racers
    seen: dict[int, tuple] = {}
    for e in b.entries:
        if e.regno:
            seen[e.regno] = (e.name, e.branch, e.birthplace)
    if seen:
        dates = {r.race_id: r.race_date for r in b.races}
        d_min = min(dates.values()) if dates else None
        d_max = max(dates.values()) if dates else None
        existing = {row.regno: row for row in session.execute(select(Racer).where(Racer.regno.in_(list(seen)))).scalars()}
        new_rows = []
        for regno, (name, branch, bp) in seen.items():
            if regno in existing:
                r = existing[regno]
                if name:
                    r.name = name
                if branch is not None:
                    r.branch = branch
                if d_min and (r.first_seen is None or d_min < r.first_seen):
                    r.first_seen = d_min
                if d_max and (r.last_seen is None or d_max > r.last_seen):
                    r.last_seen = d_max
            else:
                new_rows.append(dict(regno=regno, name=name, branch=branch, birthplace=bp, first_seen=d_min, last_seen=d_max))
        c["racers"] += _insert(session, Racer.__table__, new_rows, ["regno"])

    prev = [p.model_dump() for p in b.previews]
    c["previews"] += _insert(session, PreviewSnapshot.__table__, prev, ["race_id", "lane", "source", "fetched_at"])

    conds = [x.model_dump() for x in b.conditions]
    c["conditions"] += _insert(session, RaceCondition.__table__, conds, ["race_id", "source", "phase", "observed_at"])

    odds = [o.model_dump() for o in b.odds]
    c["odds"] += _insert(session, OddsSnapshot.__table__, odds, ["race_id", "bet_type", "source", "captured_at"])

    results = [dict(race_id=r.race_id, trifecta=r.trifecta, kimarite=r.kimarite, trifecta_payout=r.trifecta_payout,
                    payouts=r.payouts, refunds=r.refunds, is_irregular=r.is_irregular,
                    irregular_note=r.irregular_note, source=r.source, fetched_at=r.fetched_at) for r in b.results]
    c["results"] += _insert(session, Result.__table__, results, ["race_id"],
                            update_cols=["trifecta", "kimarite", "trifecta_payout", "payouts", "refunds",
                                         "is_irregular", "irregular_note", "source", "fetched_at"])
    res_entries = [x.model_dump() for x in b.result_entries]
    c["result_entries"] += _insert(session, ResultEntry.__table__, res_entries, ["race_id", "lane"],
                                   update_cols=["regno", "finish_pos", "course", "st", "abnormal"])
    for r in b.results:
        session.execute(update(Race).where(Race.id == r.race_id)
                        .values(status="cancelled" if r.is_cancelled else "finished", updated_at=now))
    return c
