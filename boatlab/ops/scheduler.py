"""日次スケジューラ（docs/07 §1）。依存を増やさない単純な分刻みループ。

常駐プロセス1つ（compose の scheduler サービス）で動かす。ジョブは冪等なので
再起動・二重実行に強い。すべて JST 基準。

  07:30       朝取込（出走表）→ 暫定予想（stage=program）
  08:00-21:45 5分ごと: today.json 再取込（直前情報・結果）→ 採点
              締切 6〜12分前のレース: 公式オッズ（3連単）と単勝・複勝を1回ずつ取得
                                      → 単勝・複勝プールの歪みの候補を記録（stage=pre、購入はしない）
              締切 4〜10分前のレース: 確定予想（stage=final、1回のみ）
              締切 2〜4分前のレース: 単勝・複勝をもう1回取得 → 候補を記録（stage=late）
  06:10       前日の turnmark（最終オッズ・返還）取込 → 再採点
  01:15（毎月1日） active モデルをパラメータ据え置きで再学習（docs/04 §13-5）
  02:30       SQLite バックアップ（7世代）
  02:45       買い方の自動研究（固定候補の評価。設定は変えない）

環境変数: BOATLAB_DISABLE_OFFICIAL_ODDS=1 で公式オッズ取得を止める（推定オッズで運用）。
"""
from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import time
import traceback
from datetime import date, datetime, timedelta

from sqlalchemy import select

from boatlab.config import DATA_DIR, DATABASE_URL
from boatlab.features.history import HistoryFrames, load_history
from boatlab.ingest.base import FetchLimitExceeded, Fetcher
from boatlab.ingest.history import ingest_turnmark_day, make_fetcher
from boatlab.ingest.official_web import fetch_odds3t, fetch_oddstf
from boatlab.model.pipeline import Predictor
from boatlab.model.selection import SelectionParams
from boatlab.ops.daily import ingest_today, predict_pending, record_late_modes, record_pool_gap, score_pending, train_and_register
from boatlab.store.db import init_db, session_scope
from boatlab.store.models import JobRun, ModelVersion, OddsSnapshot, Race
from boatlab.store.writer import write_bundle
from boatlab.util import now_jst

log = logging.getLogger(__name__)


class Scheduler:
    def __init__(self):
        self.fetcher: Fetcher = make_fetcher()
        self.predictor: Predictor | None = None
        self.predictor_version: str | None = None
        self.hist_cache: HistoryFrames | None = None
        self.hist_cache_day: date | None = None
        self.last_run: dict[str, datetime] = {}
        self.odds_done: set[int] = set()
        self.tf_done: set[int] = set()
        self.tf_late_done: set[int] = set()
        self._shadow_cache: dict[str, Predictor] = {}

    # ---------------- helpers
    def _load_active(self) -> Predictor | None:
        with session_scope() as s:
            mv = s.execute(select(ModelVersion).where(ModelVersion.status == "active")).scalars().first()
        if mv is None:
            return None
        if self.predictor is None or self.predictor_version != mv.version:
            self.predictor = Predictor.load(mv.version)
            self.predictor_version = mv.version
            log.info("loaded active model %s", mv.version)
        return self.predictor

    def _load_shadows(self) -> list[Predictor]:
        """status='shadow' のモデル版（採用前の並走比較。role='shadow' で予想を保存）。"""
        with session_scope() as s:
            versions = [mv.version for mv in s.execute(select(ModelVersion).where(ModelVersion.status == "shadow")).scalars()]
        out = []
        for v in versions:
            if v not in self._shadow_cache:
                try:
                    self._shadow_cache[v] = Predictor.load(v)
                except Exception as e:
                    log.warning("shadow model %s load failed: %r", v, e)
                    continue
            out.append(self._shadow_cache[v])
        return out

    def _hist(self, d: date) -> HistoryFrames:
        if self.hist_cache is None or self.hist_cache_day != d:
            log.info("loading 3y history cache for %s", d)
            self.hist_cache = load_history(d - timedelta(days=3 * 365), d - timedelta(days=1), slim=True)
            self.hist_cache_day = d
        return self.hist_cache

    def _due(self, name: str, when: bool, min_interval_sec: int = 55) -> bool:
        if not when:
            return False
        last = self.last_run.get(name)
        if last and (now_jst() - last).total_seconds() < min_interval_sec:
            return False
        self.last_run[name] = now_jst()
        return True

    def _job(self, name: str, fn):
        started = now_jst()
        # 開始時点で記録を残す（途中で固まる・コンテナが落ちる場合も「開始したまま未完了」が画面で分かる）
        with session_scope() as s:
            jr = JobRun(job=name, started_at=started, ok=None)
            s.add(jr)
            s.flush()
            jid = jr.id
        try:
            out = fn()
            with session_scope() as s:
                jr = s.get(JobRun, jid)
                jr.finished_at = now_jst(); jr.ok = True
                jr.summary = out if isinstance(out, dict) else {"result": str(out)[:200]}
            if out:
                log.info("%s: %s", name, out)
        except Exception as e:
            log.error("%s failed: %s\n%s", name, e, traceback.format_exc(limit=4))
            with session_scope() as s:
                jr = s.get(JobRun, jid)
                jr.finished_at = now_jst(); jr.ok = False; jr.error = repr(e)[:500]

    # ---------------- jobs
    def job_morning(self):
        d = now_jst().date()
        out = ingest_today(self.fetcher, d)
        pr = self._load_active()
        if pr:
            out["program"] = predict_pending(pr, "program", d=d, min_minutes_before_close=10,
                                             hist_cache=self._hist(d))
            for sh in self._load_shadows():
                predict_pending(sh, "program", role="shadow", d=d, min_minutes_before_close=10, hist_cache=self._hist(d))
        return out

    def job_intraday(self):
        d = now_jst().date()
        out = {"ingest": ingest_today(self.fetcher, d)}
        out["score"] = score_pending()
        pr = self._load_active()
        if pr is None:
            return out
        now = now_jst()
        pg_on = self._poolgap_enabled()
        # 締切 6〜12分前: 公式オッズ（1レース1回。失敗・無効時は推定オッズで確定される）
        if os.environ.get("BOATLAB_DISABLE_OFFICIAL_ODDS", "") != "1":
            with session_scope() as s:
                races = s.execute(select(Race).where(Race.race_date == d, Race.status != "cancelled")).scalars().all()
            for r in races:
                if r.id in self.odds_done or r.closed_at is None:
                    continue
                mins = (r.closed_at - now).total_seconds() / 60
                if 6 <= mins <= 12:
                    self.odds_done.add(r.id)
                    rec = None
                    try:
                        rec = fetch_odds3t(self.fetcher, d, r.stadium_code, r.race_no)
                        if rec is not None:
                            from boatlab.ingest.records import DayBundle
                            with session_scope() as s:
                                write_bundle(s, DayBundle(odds=[rec]))
                            out.setdefault("odds", 0)
                            out["odds"] += 1
                    except FetchLimitExceeded:
                        log.warning("official odds daily limit reached")
                    except Exception as e:
                        log.warning("odds fetch failed %s: %r", r.id, e)
                    # 単勝・複勝（1回の取得で両方）→ プール差の候補を記録（購入はしない）
                    self.tf_done.add(r.id)
                    if not pg_on:
                        continue
                    try:
                        tf = fetch_oddstf(self.fetcher, d, r.stadium_code, r.race_no)
                        self._store_tf(tf, out)
                        if rec is not None and tf:
                            out.setdefault("poolgap", 0)
                            out["poolgap"] += record_pool_gap(rec.odds, *self._tf_dicts(tf), r.id, "pre", round(mins, 1))
                    except FetchLimitExceeded:
                        log.warning("official odds daily limit reached")
                    except Exception as e:
                        log.warning("oddstf fetch failed %s: %r", r.id, e)
            # 締切 2〜4分前: 単勝・複勝をもう一度だけ取り、締切直前の歪みを記録する
            for r in races:
                if r.id in self.tf_late_done or r.closed_at is None:
                    continue
                mins = (r.closed_at - now).total_seconds() / 60
                if 2 <= mins <= 4:
                    self.tf_late_done.add(r.id)
                    # 3連単を取り直して、穴・複勝単勝の「直前版」を記録する（表示には使わない。2026-09-13〜）
                    try:
                        rec3 = fetch_odds3t(self.fetcher, d, r.stadium_code, r.race_no)
                        if rec3 is not None:
                            from boatlab.ingest.records import DayBundle
                            with session_scope() as s:
                                write_bundle(s, DayBundle(odds=[rec3]))
                            out.setdefault("odds_late", 0)
                            out["odds_late"] += 1
                            out.setdefault("late_modes", 0)
                            out["late_modes"] += record_late_modes(r.id, rec3.odds, now, mins)
                    except FetchLimitExceeded:
                        log.warning("official odds daily limit reached")
                    except Exception as e:
                        log.warning("odds3t(late) failed %s: %r", r.id, e)
                    if not pg_on:
                        continue
                    try:
                        tf = fetch_oddstf(self.fetcher, d, r.stadium_code, r.race_no, tag="_late")
                        self._store_tf(tf, out)
                        o3 = self._latest_3t(r.id)
                        if o3 and tf:
                            out.setdefault("poolgap_late", 0)
                            out["poolgap_late"] += record_pool_gap(o3, *self._tf_dicts(tf), r.id, "late", round(mins, 1))
                    except FetchLimitExceeded:
                        log.warning("official odds daily limit reached")
                    except Exception as e:
                        log.warning("oddstf(late) fetch failed %s: %r", r.id, e)
        # 暫定予想の取りこぼし救済（morning 時に出走表が未公開だったレース）
        out["program"] = predict_pending(pr, "program", d=d, min_minutes_before_close=15, hist_cache=self._hist(d))
        # 締切 4〜10分前: 確定予想（済みのレースは predict_pending 側でスキップ）
        out["final"] = predict_pending(pr, "final", d=d, min_minutes_before_close=4,
                                       max_minutes_before_close=10, hist_cache=self._hist(d))
        for sh in self._load_shadows():
            predict_pending(sh, "program", role="shadow", d=d, min_minutes_before_close=15, hist_cache=self._hist(d))
            r = predict_pending(sh, "final", role="shadow", d=d, min_minutes_before_close=4,
                                max_minutes_before_close=10, hist_cache=self._hist(d))
            out.setdefault("shadow_final", 0)
            out["shadow_final"] += r.get("predicted", 0)
        return out

    # ---------------- 単勝・複勝プール差のヘルパ
    @staticmethod
    def _poolgap_enabled() -> bool:
        """設定タブで停止されていたら公式サイトへの取得ごと行わない（1日1,000回の枠を無駄にしない）。"""
        try:
            from boatlab.ops.daily import _settings_row, poolgap_from_settings
            with session_scope() as s:
                return bool(poolgap_from_settings(_settings_row(s)).enabled)
        except Exception:
            return True

    @staticmethod
    def _tf_dicts(tf: list) -> tuple[dict | None, dict | None]:
        w = next((x.odds for x in tf if x.bet_type == "win"), None)
        p = next((x.odds for x in tf if x.bet_type == "place"), None)
        return w, p

    @staticmethod
    def _store_tf(tf: list, out: dict) -> None:
        if not tf:
            return
        from boatlab.ingest.records import DayBundle
        with session_scope() as s:
            write_bundle(s, DayBundle(odds=tf))
        out.setdefault("oddstf", 0)
        out["oddstf"] += len(tf)

    @staticmethod
    def _latest_3t(race_id: int) -> dict | None:
        with session_scope() as s:
            row = s.execute(select(OddsSnapshot.odds).where(OddsSnapshot.race_id == race_id, OddsSnapshot.bet_type == "3t")
                            .order_by(OddsSnapshot.captured_at.desc()).limit(1)).first()
        return row[0] if row else None

    def job_yesterday_final(self):
        y = now_jst().date() - timedelta(days=1)
        out = {"turnmark": ingest_turnmark_day(self.fetcher, y)}
        out["score"] = score_pending()
        return out

    def job_monthly_retrain(self):
        with session_scope() as s:
            mv = s.execute(select(ModelVersion).where(ModelVersion.status == "active")).scalars().first()
        if mv is None:
            return {"skipped": "no active model"}
        if mv.trained_until and (now_jst().date() - mv.trained_until).days < 20:
            return {"skipped": f"recently trained ({mv.trained_until})"}
        p = mv.params or {}
        sel = p.get("selection", {})
        until = now_jst().date() - timedelta(days=1)
        train_and_register(mv.version, until,
                           SelectionParams(**{k: v for k, v in sel.items() if k in SelectionParams().__dict__}),
                           description=mv.description or "", half_life=p.get("half_life"),
                           num_rounds=int(p.get("num_rounds", 400)), train_max_rows=1_200_000,
                           status=mv.status, years=9, seeds=tuple(p.get("seeds", (7,))))
        self.predictor = None  # 次回予想時に再読込
        return {"retrained": mv.version, "until": str(until)}

    def job_research(self):
        """買い方の自動研究（固定候補の評価・順送り検証）。結果は data/research/strategy_research.json。設定は変えない。"""
        from boatlab.research.strategy import run_research
        rep = run_research()
        return {"n_bt": rep["n_bt"], "n_live": rep["n_live"], "rolling_roi": rep["rolling"]["summary"].get("roi"),
                "flagged": sum(1 for r in rep["rows"] if r.get("flag"))}

    def job_backup(self):
        if not DATABASE_URL.startswith("sqlite"):
            return {"skipped": "not sqlite"}
        src = DATABASE_URL.replace("sqlite:///", "")
        bdir = DATA_DIR / "backups"
        bdir.mkdir(parents=True, exist_ok=True)
        dst = bdir / f"lab_{now_jst():%Y%m%d}.db"
        con = sqlite3.connect(src)
        bck = sqlite3.connect(dst)
        with bck:
            con.backup(bck)
        bck.close()
        con.close()
        backups = sorted(bdir.glob("lab_*.db"))
        for old in backups[:-7]:
            old.unlink()
        return {"backup": str(dst), "kept": min(len(backups), 7)}

    # ---------------- loop
    def tick(self):
        t = now_jst()
        hm = t.strftime("%H:%M")
        racing = "07:55" <= hm <= "21:50"
        if self._due("morning", hm >= "07:30" and self.last_run.get("morning", t - timedelta(days=1)).date() < t.date(), 0):
            self._job("morning", self.job_morning)
            self.odds_done.clear()
            self.tf_done.clear()
            self.tf_late_done.clear()
        if self._due("intraday", racing, 300 if not self._urgent() else 60):
            self._job("intraday", self.job_intraday)
        if self._due("yesterday", hm >= "06:10" and self.last_run.get("yesterday", t - timedelta(days=1)).date() < t.date(), 0):
            self._job("yesterday_final", self.job_yesterday_final)
        if t.day == 1 and self._due("retrain", hm >= "01:15" and self.last_run.get("retrain", t - timedelta(days=40)).month != t.month, 0):
            self._job("monthly_retrain", self.job_monthly_retrain)
        if self._due("backup", hm >= "02:30" and self.last_run.get("backup", t - timedelta(days=1)).date() < t.date(), 0):
            self._job("backup", self.job_backup)
        if self._due("research", hm >= "02:45" and self.last_run.get("research", t - timedelta(days=1)).date() < t.date(), 0):
            self._job("research", self.job_research)

    def _urgent(self) -> bool:
        """締切12分前以内のレースがあるときは1分間隔で回す。"""
        now = now_jst()
        with session_scope() as s:
            races = s.execute(select(Race.closed_at).where(Race.race_date == now.date(), Race.status != "cancelled")).all()
        for (ca,) in races:
            if ca and 0 < (ca - now).total_seconds() / 60 <= 12:
                return True
        return False

    def run_forever(self):
        init_db()
        log.info("scheduler started (JST now=%s, db=%s)", now_jst(), DATABASE_URL)
        while True:
            try:
                self.tick()
            except Exception:
                log.exception("tick failed")
            time.sleep(30)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    Scheduler().run_forever()
