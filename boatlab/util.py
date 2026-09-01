"""共通ユーティリティ。システム内の時刻はすべて JST naive。"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")


def now_jst() -> datetime:
    """サーバーのタイムゾーンに依らず JST の naive datetime を返す。"""
    return datetime.now(JST).replace(tzinfo=None)
