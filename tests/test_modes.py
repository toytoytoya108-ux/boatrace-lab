"""3モードの選定ロジック。"""
import numpy as np

from boatlab.model.modes import (ModeParams, guaranteed_stakes, market_probs, market_summary,
                                 select_ana, select_katai, select_place)
from boatlab.model.trifecta import PERMS


def _odds_from_probs(p, takeout=0.25):
    p = np.asarray(p, float); p = p / p.sum()
    return (1 - takeout) / np.clip(p, 1e-9, None)


def _lane1_heavy(w1=0.6):
    p = np.array([w1 if perm[0] == 0 else (1 - w1) / 100 for perm in PERMS])
    return _odds_from_probs(p)


def _flat():
    return _odds_from_probs(np.ones(120))


def _rough():
    """荒れる分布: 上位20通りで60%、残り100通りが各0.4%（万舟圏）→ 市場が示す万舟確率 0.40。
    実データの上位10%（0.25〜0.40）に相当。幾何減衰では上位に集中しすぎて万舟圏が薄くなる。"""
    p = np.concatenate([np.full(20, 0.6 / 20), np.full(100, 0.4 / 100)])
    return _odds_from_probs(p)


def test_market_probs_and_summary():
    o = _lane1_heavy()
    q = market_probs(o)
    assert abs(q.sum() - 1) < 1e-9
    ms = market_summary(q)
    assert ms["q1_arg"] == 0 and ms["q1_max"] > 0.5 and abs(sum(ms["q2"]) - 2.0) < 1e-9
    assert market_probs(np.full(120, np.nan)) is None


def test_ana_fires_only_on_rough_race():
    prm = ModeParams()
    rough = select_ana(_rough(), prm)               # 幾何減衰＝万舟圏の買い目が多い
    assert rough["fired"] and len(rough["points"]) == 21 and sum(rough["stakes"]) == 2100
    # 人気20〜40番目の並び: 市場確率が単調非増加
    q = market_probs(_rough())
    order = np.argsort(-q)
    assert rough["points"] == [int(i) for i in order[19:40]]
    safe = select_ana(_lane1_heavy(0.8), prm)       # 1号艇が堅い＝万舟確率が低い
    assert not safe["fired"] and safe["reason"] == "q_man_low"
    assert len(safe["points"]) == 21                # 見送りでも「買うならこれ」は返す（仮想採点用）
    assert not select_ana(np.full(120, np.nan), prm)["fired"]
    assert not select_ana(_flat(), ModeParams(ana_enabled=False))["fired"]


def test_guaranteed_stakes_property():
    # 成立するなら、どの点が当たっても払戻 >= budget >= Σstake
    for odds in ([12.0, 15.0, 20.0, 30.0, 50.0, 80.0], [8.0, 9.0, 11.0], [3.0, 4.0, 200.0]):
        k, st = guaranteed_stakes(np.array(odds), 3000)
        if k:
            assert st.sum() <= 3000
            assert all(st[i] * odds[i] >= 3000 for i in range(k))
            assert all(int(x) % 100 == 0 for x in st)
    # 本命が安すぎると成立しない（1点で予算超え）
    assert guaranteed_stakes(np.array([1.5]), 3000) == (1, np.array([2000]))   # 1点2,000円→払戻3,000円
    k, st = guaranteed_stakes(np.array([1.2, 1.3]), 3000)
    assert k == 1                                                  # 2点は無理、1点なら 2500円で成立
    assert guaranteed_stakes(np.array([np.nan, 5.0]), 3000)[0] == 0


def test_katai_trims_from_tail_and_respects_confidence():
    prm = ModeParams()
    odds = np.full(120, 500.0)
    main = list(range(10))
    odds[:10] = [6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0, 60.0, 90.0]
    r = select_katai(main, odds, 0.75, prm)
    assert r["fired"] and r["points"] == main[: len(r["points"])] and r["stake_total"] <= 3000
    assert r["min_payout"] >= 3000
    low = select_katai(main, odds, 0.5, prm)
    assert not low["fired"] and low["reason"] == "confidence_low" and len(low["points"]) == len(r["points"])
    odds[:10] = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]  # 本命が安すぎ、3点も成立しない
    r = select_katai(main, odds, 0.9, prm)
    assert not r["fired"] and r["reason"] == "too_few_points"
    assert select_katai(main, np.full(120, np.nan), 0.9, prm)["reason"] == "odds_missing"


def test_place_thresholds():
    prm = ModeParams()
    r = select_place(_lane1_heavy(0.9), prm)
    assert r["fukusho"]["fired"] and r["fukusho"]["lane"] == 1
    assert r["tansho"]["fired"] and r["tansho"]["lane"] == 1
    r = select_place(_rough(), prm)
    assert not r["fukusho"]["fired"] and not r["tansho"]["fired"]
    r = select_place(np.full(120, np.nan), prm)
    assert r["fukusho"]["reason"] == "odds_missing"


def test_params_roundtrip():
    p = ModeParams(ana_qman_min=0.3, katai_budget=5000)
    assert ModeParams.from_dict(p.to_dict()) == p
    assert ModeParams.from_dict({"unknown": 1}) == ModeParams()


def test_race_tags():
    from boatlab.model.modes import race_tags
    be = {"1": {"klass": "B1", "nat_win_rate": 4.5, "avg_st": 0.18, "exhibition_rank": 5, "course_pred": 1},
          "2": {"klass": "A1", "nat_win_rate": 6.8, "avg_st": 0.14, "exhibition_rank": 1, "course_pred": 2},
          "3": {"klass": "B1", "nat_win_rate": 4.0, "avg_st": 0.20, "exhibition_rank": 3, "course_pred": 3},
          "4": {"klass": "A2", "nat_win_rate": 5.5, "avg_st": 0.15, "exhibition_rank": 2, "course_pred": 4},
          "5": {"klass": "B1", "nat_win_rate": 4.2, "avg_st": 0.17, "exhibition_rank": 4, "course_pred": 5},
          "6": {"klass": "B2", "nat_win_rate": 3.1, "avg_st": 0.19, "exhibition_rank": 6, "course_pred": 5}}
    q = np.full(120, 1 / 120)
    t = race_tags(be, 4, "第1戦 ルーキーシリーズ", "予選", q)
    assert set(t["rough"]) == {"l1_b", "kado", "l1_ext4", "rough_stadium", "maezuke", "top1_12"}
    assert t["solid"] == ["women_rookie"]
    t = race_tags({"1": {"klass": "A1", "exhibition_rank": 1}}, 24, "一般", "モーニング一般", None)
    assert t["rough"] == [] and t["solid"] == ["kikaku"]
    assert race_tags(None, None, None, None, None) == {"rough": [], "solid": []}
