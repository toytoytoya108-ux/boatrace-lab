"""スケジューラの発火ロジック（時刻シミュレーション）。実ジョブは呼ばない。"""
from datetime import datetime, timedelta

import boatlab.ops.scheduler as sch


class FakeSched(sch.Scheduler):
    def __init__(self):
        super().__init__()
        self.calls = []

    def _job(self, name, fn):
        self.calls.append((name, sch.now_jst()))

    def _urgent(self):
        return False


def test_tick_schedule(monkeypatch):
    s = FakeSched()
    t = [datetime(2026, 9, 2, 1, 0)]
    monkeypatch.setattr(sch, "now_jst", lambda: t[0])
    s.tick()
    assert s.calls == []                       # 深夜（バックアップ時刻前）は何もしない
    t[0] = datetime(2026, 9, 2, 2, 31); s.tick()
    assert [c[0] for c in s.calls] == ["backup"]
    t[0] = datetime(2026, 9, 2, 6, 11); s.tick()
    assert [c[0] for c in s.calls][-1] == "yesterday_final"
    t[0] = datetime(2026, 9, 2, 7, 31); s.tick()
    assert [c[0] for c in s.calls][-1] == "morning"
    n0 = len(s.calls)
    s.tick()                                   # 同時刻の再tickで morning は再発火しない
    assert len(s.calls) == n0
    t[0] = datetime(2026, 9, 2, 8, 0); s.tick()
    assert [c[0] for c in s.calls][-1] == "intraday"
    t[0] = datetime(2026, 9, 2, 8, 2); s.tick() # 5分間隔内は発火しない
    assert [c[0] for c in s.calls][-1] == "intraday" and len(s.calls) == n0 + 1
    t[0] = datetime(2026, 9, 2, 8, 6); s.tick()
    assert len(s.calls) == n0 + 2
    t[0] = datetime(2026, 9, 2, 22, 30); s.tick()  # レース時間外
    assert len(s.calls) == n0 + 2
    # 翌日: morning が再発火、月初1日のみ retrain
    t[0] = datetime(2026, 10, 1, 1, 20); s.tick()
    assert "monthly_retrain" in [c[0] for c in s.calls]
    t[0] = datetime(2026, 10, 1, 7, 30); s.tick()  # 新しい日: morning（backup・yesterday も同tickで発火）
    fired = [c[0] for c in s.calls if c[1].date() == t[0].date() and c[1].hour == 7]
    assert "morning" in fired


def test_urgent_shortens_interval(monkeypatch):
    s = FakeSched()
    monkeypatch.setattr(FakeSched, "_urgent", lambda self: True)
    t = [datetime(2026, 9, 1, 12, 0)]
    monkeypatch.setattr(sch, "now_jst", lambda: t[0])
    s.tick()
    t[0] += timedelta(seconds=70)
    s.tick()
    assert [c[0] for c in s.calls].count("intraday") == 2  # 締切直前は毎分
