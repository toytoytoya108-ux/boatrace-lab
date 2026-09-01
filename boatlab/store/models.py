"""SQLAlchemy モデル（docs/03_db_schema.md に対応）。

SQLite（既定）と PostgreSQL の両方で動く型のみ使う。時刻は JST naive。
predictions / prediction_selections は追記専用（db.py でトリガを張る）。
"""
from __future__ import annotations

from datetime import date, datetime

from boatlab.util import now_jst

from sqlalchemy import (
    JSON, BigInteger, Boolean, Date, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------- core
class Stadium(Base):
    __tablename__ = "stadiums"
    code: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(16))


class Racer(Base):
    __tablename__ = "racers"
    regno: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str | None] = mapped_column(String(32))
    branch: Mapped[int | None] = mapped_column(Integer)
    birthplace: Mapped[int | None] = mapped_column(Integer)
    first_seen: Mapped[date | None] = mapped_column(Date)
    last_seen: Mapped[date | None] = mapped_column(Date)


class RacerPeriod(Base):
    """期別成績（ファン手帳）。apply_from 以降のレースにのみ結合する。"""
    __tablename__ = "racer_periods"
    regno: Mapped[int] = mapped_column(Integer, primary_key=True)
    period: Mapped[str] = mapped_column(String(8), primary_key=True)
    published_at: Mapped[date] = mapped_column(Date)
    apply_from: Mapped[date] = mapped_column(Date)
    klass: Mapped[str | None] = mapped_column(String(2))
    win_rate: Mapped[float | None] = mapped_column(Float)
    rate2: Mapped[float | None] = mapped_column(Float)
    rate3: Mapped[float | None] = mapped_column(Float)
    avg_st: Mapped[float | None] = mapped_column(Float)
    starts: Mapped[int | None] = mapped_column(Integer)
    f_count: Mapped[int | None] = mapped_column(Integer)
    l_count: Mapped[int | None] = mapped_column(Integer)
    course_stats: Mapped[dict | None] = mapped_column(JSON)


class Race(Base):
    __tablename__ = "races"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    race_date: Mapped[date] = mapped_column(Date, index=True)
    stadium_code: Mapped[int] = mapped_column(Integer, ForeignKey("stadiums.code"))
    race_no: Mapped[int] = mapped_column(Integer)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime)
    grade: Mapped[str | None] = mapped_column(String(8))
    title: Mapped[str | None] = mapped_column(String(128))
    race_type: Mapped[str | None] = mapped_column(String(32))
    distance_m: Mapped[int | None] = mapped_column(Integer)
    day_no: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="scheduled")  # scheduled/finished/cancelled
    source: Mapped[str | None] = mapped_column(String(32))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now_jst)
    __table_args__ = (UniqueConstraint("race_date", "stadium_code", "race_no"),)


class Entry(Base):
    """出走表（番組発表時点の値。初回取得値を固定し上書きしない）。"""
    __tablename__ = "entries"
    race_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("races.id"), primary_key=True)
    lane: Mapped[int] = mapped_column(Integer, primary_key=True)
    regno: Mapped[int | None] = mapped_column(Integer, index=True)
    name: Mapped[str | None] = mapped_column(String(32))
    age: Mapped[int | None] = mapped_column(Integer)
    weight: Mapped[float | None] = mapped_column(Float)
    branch: Mapped[int | None] = mapped_column(Integer)
    birthplace: Mapped[int | None] = mapped_column(Integer)
    klass: Mapped[str | None] = mapped_column(String(2))
    f_count: Mapped[int | None] = mapped_column(Integer)
    l_count: Mapped[int | None] = mapped_column(Integer)
    avg_st: Mapped[float | None] = mapped_column(Float)
    nat_win_rate: Mapped[float | None] = mapped_column(Float)
    nat_rate2: Mapped[float | None] = mapped_column(Float)
    nat_rate3: Mapped[float | None] = mapped_column(Float)
    loc_win_rate: Mapped[float | None] = mapped_column(Float)
    loc_rate2: Mapped[float | None] = mapped_column(Float)
    loc_rate3: Mapped[float | None] = mapped_column(Float)
    motor_no: Mapped[int | None] = mapped_column(Integer)
    motor_rate2: Mapped[float | None] = mapped_column(Float)
    motor_rate3: Mapped[float | None] = mapped_column(Float)
    boat_no: Mapped[int | None] = mapped_column(Integer)
    boat_rate2: Mapped[float | None] = mapped_column(Float)
    boat_rate3: Mapped[float | None] = mapped_column(Float)
    series_results: Mapped[dict | None] = mapped_column(JSON)
    program_fetched_at: Mapped[datetime | None] = mapped_column(DateTime)
    is_absent: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str | None] = mapped_column(String(32))


class PreviewSnapshot(Base):
    """直前情報（追記専用。取得のたびに行を追加）。"""
    __tablename__ = "preview_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("races.id"))
    lane: Mapped[int] = mapped_column(Integer)
    fetched_at: Mapped[datetime] = mapped_column(DateTime)
    source: Mapped[str] = mapped_column(String(32))
    course: Mapped[int | None] = mapped_column(Integer)
    st_exh: Mapped[float | None] = mapped_column(Float)
    weight: Mapped[float | None] = mapped_column(Float)
    weight_adj: Mapped[float | None] = mapped_column(Float)
    exhibition_time: Mapped[float | None] = mapped_column(Float)
    tilt: Mapped[float | None] = mapped_column(Float)
    propeller: Mapped[str | None] = mapped_column(String(8))
    parts: Mapped[list | None] = mapped_column(JSON)
    __table_args__ = (Index("ix_preview_race_fetched", "race_id", "fetched_at"),
                      UniqueConstraint("race_id", "lane", "source", "fetched_at"))


class RaceCondition(Base):
    """気象・水面（追記専用・時刻付き）。phase: preview/result"""
    __tablename__ = "race_conditions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("races.id"))
    source: Mapped[str] = mapped_column(String(32))
    observed_at: Mapped[datetime] = mapped_column(DateTime)
    phase: Mapped[str] = mapped_column(String(8))
    weather: Mapped[str | None] = mapped_column(String(8))
    temp_c: Mapped[float | None] = mapped_column(Float)
    water_temp_c: Mapped[float | None] = mapped_column(Float)
    wind_dir: Mapped[str | None] = mapped_column(String(8))
    wind_speed_m: Mapped[float | None] = mapped_column(Float)
    wave_cm: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (Index("ix_cond_race_obs", "race_id", "observed_at"),
                      UniqueConstraint("race_id", "source", "phase", "observed_at"))


class Result(Base):
    __tablename__ = "results"
    race_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("races.id"), primary_key=True)
    trifecta: Mapped[str | None] = mapped_column(String(8))
    kimarite: Mapped[str | None] = mapped_column(String(8))
    trifecta_payout: Mapped[int | None] = mapped_column(Integer)
    trifecta_popularity: Mapped[int | None] = mapped_column(Integer)
    payouts: Mapped[dict | None] = mapped_column(JSON)
    refunds: Mapped[list | None] = mapped_column(JSON)
    is_irregular: Mapped[bool] = mapped_column(Boolean, default=False)
    irregular_note: Mapped[str | None] = mapped_column(String(64))
    source: Mapped[str | None] = mapped_column(String(32))
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=now_jst)


class ResultEntry(Base):
    __tablename__ = "result_entries"
    race_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("races.id"), primary_key=True)
    lane: Mapped[int] = mapped_column(Integer, primary_key=True)
    regno: Mapped[int | None] = mapped_column(Integer)
    finish_pos: Mapped[int | None] = mapped_column(Integer)
    course: Mapped[int | None] = mapped_column(Integer)
    st: Mapped[float | None] = mapped_column(Float)
    race_time: Mapped[float | None] = mapped_column(Float)
    abnormal: Mapped[str | None] = mapped_column(String(4))


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("races.id"))
    bet_type: Mapped[str] = mapped_column(String(8), default="3t")
    captured_at: Mapped[datetime] = mapped_column(DateTime)
    source: Mapped[str] = mapped_column(String(32))  # official_web / turnmark_final / csv_import / estimated
    odds: Mapped[dict] = mapped_column(JSON)
    __table_args__ = (Index("ix_odds_race_cap", "race_id", "captured_at"),
                      UniqueConstraint("race_id", "bet_type", "source", "captured_at"))


# ---------------------------------------------------------------- model / settings
class ModelVersion(Base):
    __tablename__ = "model_versions"
    version: Mapped[str] = mapped_column(String(16), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_jst)
    parent_version: Mapped[str | None] = mapped_column(String(16))
    description: Mapped[str | None] = mapped_column(Text)
    feature_set_version: Mapped[str] = mapped_column(String(16))
    selection_version: Mapped[str] = mapped_column(String(16))
    params: Mapped[dict] = mapped_column(JSON)
    artifact_path: Mapped[str | None] = mapped_column(String(256))
    trained_until: Mapped[date | None] = mapped_column(Date)
    code_sha: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="candidate")  # candidate/active/shadow/retired
    backtest_summary: Mapped[dict | None] = mapped_column(JSON)


class SettingsVersion(Base):
    __tablename__ = "settings_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    effective_from: Mapped[datetime] = mapped_column(DateTime, default=now_jst)
    confidence_min: Mapped[float] = mapped_column(Float, default=0.70)
    ev_min: Mapped[float] = mapped_column(Float, default=1.00)
    completeness_min: Mapped[float] = mapped_column(Float, default=0.6)
    points: Mapped[int] = mapped_column(Integer, default=15)
    main_points: Mapped[int] = mapped_column(Integer, default=10)
    stake_per_point: Mapped[int] = mapped_column(Integer, default=200)
    readiness: Mapped[dict] = mapped_column(JSON, default=lambda: {
        "min_races": 1000, "hit_rate": 0.80, "roi": 1.00, "recent_n": 100, "recent_hit_rate": 0.70, "target_roi": 1.10,
    })
    extra: Mapped[dict | None] = mapped_column(JSON)


# ---------------------------------------------------------------- pred（追記専用）
class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("races.id"), index=True)
    model_version: Mapped[str] = mapped_column(String(16), ForeignKey("model_versions.version"))
    settings_id: Mapped[int] = mapped_column(Integer, ForeignKey("settings_versions.id"))
    stage: Mapped[str] = mapped_column(String(8))     # program / final
    role: Mapped[str] = mapped_column(String(8))      # active / shadow
    created_at: Mapped[datetime] = mapped_column(DateTime)  # トリガで強制
    asof_ts: Mapped[datetime] = mapped_column(DateTime)
    post_time_at_pred: Mapped[datetime] = mapped_column(DateTime)
    features_used: Mapped[dict] = mapped_column(JSON)
    preview_snapshot_ids: Mapped[list | None] = mapped_column(JSON)
    odds_snapshot_id: Mapped[int | None] = mapped_column(Integer)
    condition_id: Mapped[int | None] = mapped_column(Integer)
    completeness: Mapped[float] = mapped_column(Float)
    missing_fields: Mapped[list | None] = mapped_column(JSON)
    flags: Mapped[dict | None] = mapped_column(JSON)
    boat_eval: Mapped[dict] = mapped_column(JSON)
    probs: Mapped[dict] = mapped_column(JSON)
    odds_used: Mapped[dict] = mapped_column(JSON)
    ev: Mapped[dict] = mapped_column(JSON)
    confidence: Mapped[float] = mapped_column(Float)
    expected_return: Mapped[float] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(8))  # buy / skip
    skip_reason: Mapped[str | None] = mapped_column(String(64))
    rationale: Mapped[dict] = mapped_column(JSON)
    rationale_text: Mapped[str | None] = mapped_column(Text)
    input_hash: Mapped[str] = mapped_column(String(64))
    __table_args__ = (UniqueConstraint("race_id", "model_version", "stage", "role"),)


class PredictionSelection(Base):
    __tablename__ = "prediction_selections"
    prediction_id: Mapped[int] = mapped_column(Integer, ForeignKey("predictions.id"), primary_key=True)
    combo: Mapped[str] = mapped_column(String(8), primary_key=True)
    rank: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(8))  # main / hole
    stake: Mapped[int] = mapped_column(Integer)
    prob: Mapped[float | None] = mapped_column(Float)
    odds_at_pred: Mapped[float | None] = mapped_column(Float)
    odds_source: Mapped[str | None] = mapped_column(String(16))
    ev: Mapped[float | None] = mapped_column(Float)
    reason: Mapped[dict | None] = mapped_column(JSON)


# ---------------------------------------------------------------- score
class Scoring(Base):
    __tablename__ = "scoring"
    prediction_id: Mapped[int] = mapped_column(Integer, ForeignKey("predictions.id"), primary_key=True)
    race_id: Mapped[int] = mapped_column(BigInteger, index=True)
    scored_at: Mapped[datetime] = mapped_column(DateTime, default=now_jst)
    valid: Mapped[bool] = mapped_column(Boolean)
    invalid_reason: Mapped[str | None] = mapped_column(String(64))
    actual_trifecta: Mapped[str | None] = mapped_column(String(8))
    actual_payout: Mapped[int | None] = mapped_column(Integer)
    actual_popularity: Mapped[int | None] = mapped_column(Integer)
    hit: Mapped[bool | None] = mapped_column(Boolean)
    hit_kind: Mapped[str | None] = mapped_column(String(8))
    refunded_points: Mapped[int] = mapped_column(Integer, default=0)
    refunded_stake: Mapped[int] = mapped_column(Integer, default=0)
    stake_total: Mapped[int | None] = mapped_column(Integer)
    payout_total: Mapped[int | None] = mapped_column(Integer)
    pnl: Mapped[int | None] = mapped_column(Integer)
    roi: Mapped[float | None] = mapped_column(Float)
    category: Mapped[str | None] = mapped_column(String(16))
    odds_final_ratio: Mapped[float | None] = mapped_column(Float)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str | None] = mapped_column(String(16))
    params: Mapped[dict | None] = mapped_column(JSON)
    period_from: Mapped[date | None] = mapped_column(Date)
    period_to: Mapped[date | None] = mapped_column(Date)
    split: Mapped[str | None] = mapped_column(String(16))
    metrics: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now_jst)
    note: Mapped[str | None] = mapped_column(Text)


class BacktestPrediction(Base):
    __tablename__ = "backtest_predictions"
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("backtest_runs.id"), primary_key=True)
    race_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)


# ---------------------------------------------------------------- ops
class RawFile(Base):
    __tablename__ = "raw_files"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32))
    key: Mapped[str] = mapped_column(String(128))
    sha256: Mapped[str | None] = mapped_column(String(64))
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime)
    path: Mapped[str | None] = mapped_column(String(256))
    __table_args__ = (UniqueConstraint("source", "key"),)


class FetchLog(Base):
    __tablename__ = "fetch_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32))
    key: Mapped[str] = mapped_column(String(128))
    started_at: Mapped[datetime] = mapped_column(DateTime)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    http_status: Mapped[int | None] = mapped_column(Integer)
    ok: Mapped[bool] = mapped_column(Boolean)
    retry_no: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)


class JobRun(Base):
    __tablename__ = "job_run"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job: Mapped[str] = mapped_column(String(64))
    started_at: Mapped[datetime] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    ok: Mapped[bool | None] = mapped_column(Boolean)
    summary: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
