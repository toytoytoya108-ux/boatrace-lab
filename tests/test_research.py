"""買い方の自動研究: 合成データで一通り動くこと・設定を変えないこと・順送りの縛りを確認する。"""
import json

import numpy as np

from boatlab.research import strategy as S


def _synthetic(n=900, seed=0, months=("2026-01", "2026-02", "2026-03", "2026-06", "2026-07")):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.full(120, 0.15), size=n).astype(np.float32)
    O = (0.75 / np.clip(P, 1e-4, None) * rng.uniform(0.7, 1.4, size=P.shape)).astype(np.float32)
    tri = np.array([rng.choice(120, p=p / p.sum()) for p in P])
    payout = O[np.arange(n), tri] * 100
    date = np.array([f"{months[i % len(months)]}-{(i % 27) + 1:02d}" for i in range(n)])
    return dict(kind="bt", date=date, stadium=np.ones(n, int), P=P, O=O, tri=tri, payout=payout.astype(float), model="test")


def test_run_research_synthetic(tmp_path):
    bt = _synthetic()
    live = dict(_synthetic(120, seed=1, months=("2026-09",)), kind="live")
    out = tmp_path / "r.json"
    rep = S.run_research(bt=bt, live=live, current={"ev_min": 1.0, "odds_hi": 50, "s15_min": 0.73, "max_points": 5}, out=out)
    assert out.exists() and json.loads(out.read_text(encoding="utf-8"))["grid_size"] == len(S.GRID)
    assert len(rep["rows"]) == len(S.GRID)                      # 現在設定はグリッド内 → 行は増えない
    cur = [r for r in rep["rows"] if r["is_current"]]
    assert len(cur) == 1 and cur[0]["label"] == rep["current"]
    for r in rep["rows"]:
        assert set(r["params"]) == {"ev_min", "odds_hi", "s15_min", "max_points"}
        for k in ("bt_explore", "bt_confirm", "live"):
            assert k in r and "n" in r[k]
        assert r["flag"] in ("", "候補", "仮候補")
    # 順送り: 選択は必ず「その月より前」のデータだけで行われる（picks の月は昇順、最初の月は選ばれない）
    months = [p["month"] for p in rep["rolling"]["picks"]]
    assert months == sorted(months) and "2026-01" not in months
    # 候補の印は現在設定より確認期間で高いものだけ
    for r in rep["rows"]:
        if r["flag"]:
            assert r["bt_confirm"]["roi"] > cur[0]["bt_confirm"]["roi"]


def test_current_outside_grid_is_appended(tmp_path):
    bt = _synthetic(300)
    rep = S.run_research(bt=bt, live=None, current={"ev_min": 0.85, "odds_hi": 25, "s15_min": 0.73, "max_points": 4}, out=tmp_path / "r.json")
    assert len(rep["rows"]) == len(S.GRID) + 1
    assert sum(r["is_current"] for r in rep["rows"]) == 1
