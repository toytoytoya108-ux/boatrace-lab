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
    out = daily.predict_pending(pr2, "final", d=d, now=datetime(2018, 7, 5, 8, 0))
    assert out["predicted"] > 100
    from boatlab.store.models import Prediction, PredictionSelection, Scoring
    with dbmod.session_scope() as s:
        p = s.execute(select(Prediction)).scalars().first()
        sels = s.execute(select(PredictionSelection).where(PredictionSelection.prediction_id == p.id)).scalars().all()
        assert len(sels) == 15 and sum(1 for x in sels if x.kind == "hole") == 5
        assert abs(sum(p.probs.values()) - 1.0) < 1e-6
        assert 0 <= p.confidence <= 1 and p.decision in ("buy", "skip")
    sc = daily.score_pending()
    assert sc["scored"] == out["predicted"]
    with dbmod.session_scope() as s:
        rows = s.execute(select(Scoring)).scalars().all()
        # 過去日シミュレーション → created_at > 締切 → 全件 invalid（リーク検査が働いている）
        assert all(r.valid is False and r.invalid_reason == "created_after_close" for r in rows)
