"""単勝・複勝プールの歪み拾い: パーサ・候補抽出・追記専用の確認。"""
from datetime import date, datetime

import pytest

from boatlab.ingest.official_web import parse_oddstf
from boatlab.model.trifecta import PERM_LABELS, PERMS
from boatlab.research.poolgap import PoolGapParams, find_picks, pool_probs, ref_probs

WIN_HTML = """<h3>単勝</h3><table>{}</table><h3>複勝</h3><table>{}</table>"""
_ROW = '<tr><td><span class="table1_boatImage1Number is-type1_3 is-boatColor{b}">{b}</span></td><td>選手{b}</td>{o}</tr>'


def _page(win, place):
    w = "".join(_ROW.format(b=i + 1, o=f"<td>{v}</td>") for i, v in enumerate(win))
    p = "".join(_ROW.format(b=i + 1, o=f"<td>{lo}</td><td>{hi}</td>") for i, (lo, hi) in enumerate(place))
    return WIN_HTML.format(w, p)


def test_parse_oddstf():
    win = [1.5, 12.3, 4.0, 9.9, 25.0, 60.0]
    place = [(1.0, 1.2), (2.5, 3.1), (1.8, 2.2), (2.0, 2.4), (4.0, 5.0), (8.0, 9.0)]
    got = parse_oddstf(_page(win, place))
    assert [got["win"][str(i + 1)] for i in range(6)] == win
    assert got["place"]["2"] == {"lo": 2.5, "hi": 3.1}
    assert parse_oddstf("<html>まったく別のページ</html>") == {}       # 壊れたら空 dict（NULL扱い）


def _odds3t_from_probs(p120, takeout=0.25):
    return {PERM_LABELS[i]: round((1 - takeout) / max(p120[i], 1e-6), 1) for i in range(120)}


def test_find_picks_flags_underpriced_boat():
    # 1号艇が強いレース（3連単プールでは1着確率 ≈ 0.6）
    p = [0.0] * 120
    for i, perm in enumerate(PERMS):
        p[i] = 0.02 if perm[0] == 0 else 0.004
    s = sum(p)
    p = [x / s for x in p]
    o3 = _odds3t_from_probs(p)
    ref_win, _ = ref_probs(o3)
    assert ref_win[0] > 0.5
    # 単勝プールは1号艇を 1着確率 0.2 相当（オッズ3.75倍）でしか評価していない → 候補になるはず
    win_odds = {"1": 3.7, "2": 4.0, "3": 5.0, "4": 6.0, "5": 8.0, "6": 12.0}
    picks = find_picks(o3, win_odds, None)
    assert [x["lane"] for x in picks if x["bet_type"] == "win"] == [1]
    assert picks[0]["ratio"] > 2.0 and picks[0]["odds"] == 3.7
    # 単勝プールが3連単と同じ見立てなら候補は出ない
    fair = {str(i + 1): round(0.75 / max(ref_win[i], 1e-6), 1) for i in range(6)}
    assert find_picks(o3, fair, None) == []
    # 上限オッズ超え・元返しは除外
    assert 1 not in [x["lane"] for x in find_picks(o3, {**win_odds, "1": 25.0}, None)]
    assert 1 not in [x["lane"] for x in find_picks(o3, {**win_odds, "1": 1.0}, None)]
    # enabled=False で何も出さない
    assert find_picks(o3, win_odds, None, PoolGapParams(enabled=False)) == []


def test_pool_probs_place_sums_to_two():
    place = {str(i + 1): {"lo": v, "hi": v + 0.3} for i, v in enumerate([1.1, 2.0, 2.5, 3.0, 5.0, 9.0])}
    _, pl = pool_probs(None, place)
    assert pl is not None and abs(pl.sum() - 2.0) < 1e-9
    assert find_picks({}, None, place) == []          # 3連単オッズが無ければ候補なし


def test_incomplete_odds_is_ignored():
    o3 = _odds3t_from_probs([1 / 120] * 120)
    assert find_picks(o3, {"1": 2.0}, None) == []      # 6艇そろわない単勝オッズは使わない
    assert ref_probs({PERM_LABELS[i]: 10.0 for i in range(50)}) is None


def test_picks_are_append_only(tmp_path, monkeypatch):
    monkeypatch.setenv("BOATLAB_DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    import importlib

    from boatlab import config as cfg
    importlib.reload(cfg)
    from boatlab.store import db as dbm
    importlib.reload(dbm)
    from boatlab.store import models as mm
    dbm.init_db()
    with dbm.session_scope() as s:
        s.add(mm.Race(id=202609090101, race_date=date(2026, 9, 9), stadium_code=1, race_no=1, source="t"))
    with dbm.session_scope() as s:
        s.add(mm.PoolGapPick(race_id=202609090101, bet_type="win", lane=3, stage="pre",
                             created_at=datetime(2026, 9, 9, 12, 0), minutes_before=8.2, odds_seen=7.4,
                             p_pool=0.099, p_ref=0.23, ratio=2.32, stake=100, params_version="pg1", params={}))
    with dbm.session_scope() as s:
        assert s.query(mm.PoolGapPick).count() == 1
    with pytest.raises(Exception):
        with dbm.session_scope() as s:
            s.query(mm.PoolGapPick).update({"stake": 200})
