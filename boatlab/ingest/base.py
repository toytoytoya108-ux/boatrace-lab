"""取得層の共通基盤：レート制限・リトライ・原本キャッシュ・取得ログ。

すべての外部アクセスはこのモジュールの Fetcher を経由する（docs/07 §2）。
- ソースごとの最小間隔を守る（単一スレッド前提）
- 一時的エラー（接続失敗・5xx・タイムアウト）は指数バックオフで最大3回
- 4xx はリトライしない（404 = データなし）
- 取得した原本は data/raw/<source>/<key> に保存（再取得抑止・監査）
- 1 日あたりの上限（official_web）を超えたら FetchLimitExceeded
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import httpx

from boatlab.config import RAW_DIR, SOURCES, SourceSpec

log = logging.getLogger(__name__)


class FetchLimitExceeded(RuntimeError):
    pass


class NotFound(RuntimeError):
    pass


@dataclass
class FetchResult:
    source: str
    key: str
    url: str
    status: int
    content: bytes
    fetched_at: datetime
    from_cache: bool
    sha256: str


class _RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


class Fetcher:
    """HTTP 取得の唯一の入口。"""

    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        log_hook: Callable[[dict], None] | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.raw_dir = Path(raw_dir)
        self.log_hook = log_hook  # fetch_log テーブルへの書き込みなど
        self.timeout = timeout
        self.max_retries = max_retries
        self._limiters = {name: _RateLimiter(spec.min_interval_sec) for name, spec in SOURCES.items()}
        self._daily_count: dict[tuple[str, date], int] = {}
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

    # ---- 原本キャッシュ ----
    def raw_path(self, source: str, key: str) -> Path:
        return self.raw_dir / source / key

    def cached(self, source: str, key: str) -> FetchResult | None:
        p = self.raw_path(source, key)
        if p.exists():
            content = p.read_bytes()
            return FetchResult(
                source, key, "", 200, content,
                datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc), True,
                hashlib.sha256(content).hexdigest(),
            )
        return None

    # ---- 取得 ----
    def fetch(self, source: str, url: str, key: str, use_cache: bool = True) -> FetchResult:
        spec: SourceSpec = SOURCES[source]
        if use_cache:
            c = self.cached(source, key)
            if c is not None:
                return c
        today = datetime.now(timezone.utc).date()
        if spec.max_per_day is not None:
            n = self._daily_count.get((source, today), 0)
            if n >= spec.max_per_day:
                raise FetchLimitExceeded(f"{source}: daily limit {spec.max_per_day} reached")
            self._daily_count[(source, today)] = n + 1

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._limiters[source].wait()
            started = time.monotonic()
            fetched_at = datetime.now(timezone.utc)
            status = 0
            err = ""
            try:
                r = self._client.get(url, headers=spec.headers)
                status = r.status_code
                if status == 200:
                    content = r.content
                    p = self.raw_path(source, key)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(content)
                    self._log(source, key, started, status, True, attempt, "")
                    return FetchResult(source, key, url, status, content, fetched_at, False,
                                       hashlib.sha256(content).hexdigest())
                if 400 <= status < 500:
                    self._log(source, key, started, status, False, attempt, f"http {status}")
                    if status == 404:
                        raise NotFound(url)
                    raise httpx.HTTPStatusError(f"http {status}", request=r.request, response=r)
                err = f"http {status}"
            except NotFound:
                raise
            except httpx.HTTPStatusError:
                raise
            except Exception as e:  # 接続失敗・タイムアウト・5xx
                err = repr(e)
                last_err = e
            self._log(source, key, started, status, False, attempt, err)
            backoff = min(2 ** attempt * 2.0, 30.0)
            log.warning("fetch retry %s %s (%s) in %.0fs", source, key, err, backoff)
            time.sleep(backoff)
        raise RuntimeError(f"fetch failed after retries: {source} {key}: {last_err}")

    def fetch_json(self, source: str, url: str, key: str, use_cache: bool = True):
        res = self.fetch(source, url, key, use_cache=use_cache)
        return json.loads(res.content.decode("utf-8")), res

    def _log(self, source, key, started, status, ok, attempt, err):
        rec = {
            "source": source, "key": key, "started_at": datetime.now(timezone.utc),
            "duration_ms": int((time.monotonic() - started) * 1000), "http_status": status,
            "ok": ok, "retry_no": attempt, "error": err or None,
        }
        if self.log_hook:
            try:
                self.log_hook(rec)
            except Exception:  # ログ失敗で取得を止めない
                log.exception("fetch log hook failed")
