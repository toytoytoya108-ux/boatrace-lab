"""学習→予想保存→採点 の一連の流れ（小データ）。過去日をシミュレートするため採点は invalid になる（リーク検査の動作確認）。"""
import os
import shutil
from datetime import date, datetime
from pathlib import Path

import pytest
from sqlalchemy import select, text


@pytest.mark.skipif(not Path("data/lab.db").exists(), reason="requires ingested DB")
def test_train_predict_score(tmp_path, monkeypatch):
    # 実DBの 2018 年分をコピーして使う（書き込みはコピー側）
    src = Path("data/lab.db")
    dst = tmp_path / "t.db"
    shutil.copy(src, dst)
    for suf in ("-wal", "-shm"):
        if Path(str(src) + suf).exists():
            shutil.copy(str(src) + suf, str(dst) + suf)
    monkeypatch.setenv("BOATLAB_DATABASE_URL", f"sqlite:///{dst}")
    monkeypatch.setenv("BOATLAB_DATA_DIR", str(tmp_path))
    import importlib
    import boatlab.config as cfgm
    importlib.reload(cfgm)
    import boatlab.store.db as dbm
    importlib.reload(dbm)
    from boatlab.store import db as dbmod
    dbmod.init_db(f"sqlite:///{dst}")
    from boatlab.ops import daily
    from boatlab.model.selection import SelectionParams
    importlib.reload(daily)
    pr = daily.train_and_register("test-0.1", date(2018, 6, 30), SelectionParams(hole_min_odds=20), num_rounds=30, years=1)
    assert pr.lam[0] > 0
    from boatlab.model.pipeline import Predictor
    pr2 = Predictor.load("test-0.1")
    d = date(2018, 7, 5)
    # 設定：資金配分＝確率比例2乗（extra.staking）
    from boatlab.store.models import SettingsVersion
    with dbmod.session_scope() as s:
        s.add(SettingsVersion(extra={"staking": {"method": "prob", "prob_power": 2.0}}))
    out = daily.predict_pending(pr2, "final", d=d, now=datetime(2018, 7, 5, 8, 0))
    assert out["predicted"] > 100
    from boatlab.store.models import Prediction, PredictionSelection, Scoring
    with dbmod.session_scope() as s:
        p = s.execute(select(Prediction).where(Prediction.model_version == "test-0.1")).scalars().first()
        sels = s.execute(select(PredictionSelection).where(PredictionSelection.prediction_id == p.id)).scalars().all()
        assert len(sels) == 15 and sum(1 for x in sels if x.kind == "hole") == 5
        stakes = [x.stake for x in sorted(sels, key=lambda x: x.rank)]
        assert sum(stakes) == 3000 and all(x % 100 == 0 and x >= 100 for x in stakes) and stakes[0] >= stakes[-1]
        assert p.flags.get("staking") == "prob"
        assert abs(sum(p.probs.values()) - 1.0) < 1e-6
        assert 0 <= p.confidence <= 1 and p.decision in ("buy", "skip")
    # 絞り込み型（role='focused'）も同時に保存される
    with dbmod.session_scope() as s:
        fps = s.execute(select(Prediction).where(Prediction.model_version == "test-0.1", Prediction.role == "focused")).scalars().all()
        assert len(fps) == out["predicted"]
        n_buy = 0
        for fp in fps:
            fsel = s.execute(select(PredictionSelection).where(PredictionSelection.prediction_id == fp.id)).scalars().all()
            assert len(fsel) <= 5 and all(100 <= x.stake <= 1000 and x.stake % 100 == 0 for x in fsel)
            assert sum(x.stake for x in fsel) <= 3000
            assert fp.flags.get("mode") == "focused" and fp.decision in ("buy", "skip")
            n_buy += fp.decision == "buy"
        assert n_buy < len(fps)  # 全部買いにはならない（絞り込み）
    sc = daily.score_pending()
    assert sc["scored"] == out["predicted"] * 2  # 本体＋絞り込み型
    with dbmod.session_scope() as s:
        rows = s.execute(select(Scoring)).scalars().all()
        # 過去日シミュレーション → created_at > 締切 → 全件 invalid（リーク検査が働いている）
        assert all(r.valid is False and r.invalid_reason == "created_after_close" for r in rows)
    # 採点は点ごとの賭け金で行われる（有効行は無いので score_race を直接確認）
    from boatlab.backtest.metrics import score_race
    from boatlab.model.trifecta import combo_index
    with dbmod.session_scope() as s:
        p = s.execute(select(Prediction).where(Prediction.model_version == "test-0.1")).scalars().first()
        sels = sorted(s.execute(select(PredictionSelection).where(PredictionSelection.prediction_id == p.id)).scalars().all(), key=lambda x: x.rank)
        idx = [combo_index(x.combo) for x in sels]
        sc = score_race(idx, idx[:10], idx[0], 1000, [], stakes=[x.stake for x in sels])
        assert sc["payout_total"] == sels[0].stake * 10 and sc["stake_total"] == 3000
