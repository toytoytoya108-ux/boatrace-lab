import numpy as np
from boatlab.model.staking import StakingParams, allocate


def _race(seed=0):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(120) * 0.3)
    odds = 0.75 / p
    sel = [int(i) for i in np.argsort(-p)[:15]]
    return p, odds, sel, sel[:10]


def test_all_methods_sum_to_total_and_respect_bounds():
    for seed in range(20):
        p, odds, sel, main = _race(seed)
        for m in ("uniform", "payout_equal", "prob", "group"):
            for power in (1.0, 2.0):
                st = allocate(sel, main, p, odds, StakingParams(method=m, prob_power=power))
                assert len(st) == 15 and sum(st) == 3000
                assert all(x % 100 == 0 and x >= 100 for x in st)
                if m != "group":
                    assert max(st) <= 900


def test_prob_gives_more_to_higher_probability():
    p, odds, sel, main = _race(3)
    st = allocate(sel, main, p, odds, StakingParams(method="prob", prob_power=2))
    assert st[0] >= st[-1] and st[0] > 200


def test_uniform_matches_legacy():
    p, odds, sel, main = _race(1)
    assert allocate(sel, main, p, odds, StakingParams()) == [200] * 15


def test_nan_odds_handled_in_payout_equal():
    p, odds, sel, main = _race(2)
    odds = odds.copy(); odds[sel[0]] = np.nan; odds[sel[3]] = np.inf
    st = allocate(sel, main, p, odds, StakingParams(method="payout_equal"))
    assert sum(st) == 3000


def test_score_race_uses_per_point_stakes():
    from boatlab.backtest.metrics import score_race
    sel = list(range(15)); stakes = [500] + [100] * 13 + [1200]
    sc = score_race(sel, sel[:10], 0, 1000, [], stakes=stakes)
    assert sc["stake_total"] == 3000 and sc["payout_total"] == 5000 and sc["pnl"] == 2000
    sc2 = score_race(sel, sel[:10], 14, 1000, [], stakes=stakes)
    assert sc2["payout_total"] == 12000 and sc2["hit_kind"] == "hole"
