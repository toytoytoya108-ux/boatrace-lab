"""実戦投入判定：買い方ごとの閾値（focused は hit_rate を持たない）で落ちないこと。"""
import pandas as pd
from boatlab.analytics import performance as perf


def _df(n=30):
    rows = []
    for i in range(n):
        rows.append(dict(decision="buy", hit=(i % 3 == 0), hit_kind="main" if i % 3 == 0 else None, stake_total=3000,
                         payout_total=9000 if i % 3 == 0 else 0, pnl=6000 if i % 3 == 0 else -3000, confidence=0.7,
                         expected_return=1.05, stadium="多摩川" if i % 2 else "戸田", stadium_code=5 if i % 2 else 2,
                         flags={}, race_date=pd.Timestamp("2026-09-01") + pd.Timedelta(days=i), race_id=i, model_version="1.0",
                         grade=None, race_no=1, hole_stake=0, main_stake=3000))
    return pd.DataFrame(rows)


def test_focused_thresholds_do_not_require_hit_rate():
    out = perf.readiness(_df(), perf.FOCUSED_READINESS_DEFAULT)
    names = [c["name"] for c in out["checks"]]
    assert "累計的中率" not in names and "回収率の95%区間下限" in names
    assert isinstance(out["passed"], bool)


def test_std_thresholds_still_work():
    th = {"min_races": 10, "hit_rate": 0.3, "roi": 1.0, "recent_n": 10, "recent_hit_rate": 0.3, "target_roi": 1.1}
    out = perf.readiness(_df(), th)
    assert any(c["name"] == "累計的中率" for c in out["checks"])


def test_empty_df_both_modes():
    empty = _df(0)
    for th in (perf.FOCUSED_READINESS_DEFAULT, {"min_races": 10, "hit_rate": 0.3, "roi": 1.0, "recent_n": 10, "recent_hit_rate": 0.3, "target_roi": 1.1}):
        out = perf.readiness(empty, th)
        assert out["passed"] is False
