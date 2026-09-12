"""favorite-check: 実データ形状で落ちないこと（メモリ事故と payouts NULL の再発防止）。

2026-09-12: 初版は全期間の3連単オッズ（2026年だけで3.7万レース×120通り）を pandas に
読み込んでおり、2GB の本番サーバーでは落ちた。また結果未確定の行では payouts が NULL で
渡ってきて .get() に失敗した（poolgap でも同じ踏み方をしている）。
"""
from datetime import date, datetime

import numpy as np
import pytest
from typer.testing import CliRunner

from boatlab.model.trifecta import PERM_LABELS, PERMS


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("BOATLAB_DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    monkeypatch.setenv("BOATLAB_DATA_DIR", str(tmp_path))
    import importlib

    from boatlab import config as cfg
    importlib.reload(cfg)
    from boatlab.store import db as dbm
    importlib.reload(dbm)
    dbm.init_db()
    return dbm


def _seed(dbm, n=60, strength=(9.0, 3, 2, 1.2, 0.8, 0.5), noise=0.12, seed=0):
    from boatlab.store import models as mm
    rng = np.random.default_rng(seed)
    day = date(2026, 9, 11)
    base = int(day.strftime("%Y%m%d")) * 10000
    rid_of = lambda i: base + (i // 12 + 1) * 100 + (i % 12) + 1
    with dbm.session_scope() as s:
        for i in range(n):
            s.add(mm.Race(id=rid_of(i), race_date=day, stadium_code=i // 12 + 1,
                          race_no=(i % 12) + 1, source="t"))
    with dbm.session_scope() as s:
        for i in range(n):
            st = rng.dirichlet(np.array(strength))
            p = np.array([st[a] * st[b] * st[c] for a, b, c in PERMS])
            p /= p.sum()
            q = p * rng.lognormal(0, noise, 120)
            q /= q.sum()
            mk = lambda v: {PERM_LABELS[j]: round(float(0.75 / max(v[j], 1e-5)), 1) for j in range(120)}
            s.add(mm.OddsSnapshot(race_id=rid_of(i), bet_type="3t", captured_at=datetime(2026, 9, 11, 12, i % 50),
                                  source="official_web", odds=mk(p)))
            s.add(mm.OddsSnapshot(race_id=rid_of(i), bet_type="3t", captured_at=datetime(2026, 9, 12, 6, 10),
                                  source="turnmark_final", odds=mk(q)))
            # 半分は結果未確定（payouts が NULL）＝以前クラッシュした形
            s.add(mm.Result(race_id=rid_of(i), trifecta="1-2-3", trifecta_payout=2500.0, source="t",
                            payouts=({"place": [{"combination": "1", "amount": 110},
                                                {"combination": "2", "amount": 180}]} if i % 2 == 0 else None)))


def _run(args=()):
    from boatlab.cli import app
    return CliRunner().invoke(app, ["favorite-check", *args])


def test_runs_on_realistic_shape(env):
    _seed(env)
    r = _run()
    assert r.exit_code == 0, r.exception
    assert "いちばん堅い艇が一致" in r.output and "確率の差" in r.output


def test_only_reads_races_that_have_pre_deadline_odds(env):
    """締切前オッズの無い過去レースまで読み込まない（本番DBは3.7万レース分ある）。"""
    from boatlab.store import models as mm
    _seed(env, n=24)
    old = [202601010000 + (i // 12 + 1) * 100 + (i % 12) + 1 for i in range(288)]
    with env.session_scope() as s:                       # 確定オッズだけの古いレースを大量に足す
        for i, rid in enumerate(old):
            s.add(mm.Race(id=rid, race_date=date(2026, 1, 1), stadium_code=i // 12 + 1, race_no=(i % 12) + 1, source="t"))
    with env.session_scope() as s:
        for rid in old:
            s.add(mm.OddsSnapshot(race_id=rid, bet_type="3t", captured_at=datetime(2026, 1, 2, 6, 10),
                                  source="turnmark_final", odds={l: 10.0 for l in PERM_LABELS}))
    r = _run()
    assert r.exit_code == 0, r.exception
    assert "締切前オッズのあるレース: 24R" in r.output     # 324R ではない


def test_reports_selection_agreement_when_confident(env):
    """1号艇が圧倒的なレースを与えると、閾値を満たす対象レースが出て一致率も高い。"""
    _seed(env, n=60, strength=(60.0, 3, 2, 1.2, 0.8, 0.5), noise=0.05, seed=3)
    r = _run(["--threshold", "0.80"])
    assert r.exit_code == 0, r.exception
    assert "買い対象レース" in r.output
    agree = float(r.output.split("いちばん堅い艇が一致: ")[1].split("%")[0])
    assert agree >= 90.0


def test_exits_cleanly_without_pre_deadline_odds(env):
    r = _run()
    assert r.exit_code == 1 and "まだありません" in r.output
