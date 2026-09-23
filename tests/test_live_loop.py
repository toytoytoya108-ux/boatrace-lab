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
    # 3モード（穴・堅い・複勝単勝）も確定予想と同時に記録される。市場ベースの2つは
    # 実オッズが無いレースでは skip（odds_estimated）として残る＝黙って欠ける記録は無い
    with dbmod.session_scope() as s:
        for role in ("katai_t", "honmei", "place", "ev1"):
            mps = s.execute(select(Prediction).where(Prediction.model_version == "test-0.1", Prediction.role == role)).scalars().all()
            assert len(mps) == out["predicted"], role
            for mp in mps:
                assert mp.flags.get("mode") == role and mp.decision in ("buy", "skip")
                msel = s.execute(select(PredictionSelection).where(PredictionSelection.prediction_id == mp.id)).scalars().all()
                if mp.skip_reason == "odds_estimated":
                    assert msel == []                       # 実オッズが無ければ市場ベースの買い目は出せない
                elif mp.decision == "skip" and role == "katai" and mp.skip_reason in ("too_few_points", "no_guarantee", "odds_missing"):
                    assert msel == []
                elif role == "ana":
                    assert len(msel) == 21 and all(x.stake == 100 and x.kind == "ana" for x in msel)
                elif role == "katai":
                    assert 3 <= len(msel) <= 10 and sum(x.stake for x in msel) <= 3000
                    assert all(x.stake % 100 == 0 and x.kind == "katai" for x in msel)
                elif role == "katai_t":
                    assert 1 <= len(msel) <= 10 and sum(x.stake for x in msel) == 3000 and all(x.kind == "katai_t" for x in msel)
                elif role == "honmei":
                    # 保証が成立しない／点数が足りないレースは買い目なし。成立していれば10点で、
                    # どの点が当たっても払戻 ≥ 投資×1.5（min_payout ≥ 1.5 × Σ賭け金）
                    if mp.skip_reason in ("no_guarantee", "too_few_points", "odds_missing"):
                        assert msel == []
                    else:
                        assert len(msel) == 10 and all(x.kind == "honmei" and x.stake % 100 == 0 for x in msel)
                        total = sum(x.stake for x in msel)
                        f = mp.flags
                        assert total <= 10000 and f["min_payout"] >= total * 1.5 - 1e-6
                        assert all(x.stake * x.odds_at_pred >= f["min_payout"] - 1e-6 for x in msel)
                elif role == "ev1":
                    # 規則が成立すれば 複1（オッズ不要）。較正は実オッズのときだけ。見送り行は較正の参考1点（実オッズ時）か空
                    f = mp.flags
                    assert (mp.decision == "buy") == bool(f["rule"]["fired"] or f["cal"]["fired"])
                    if mp.decision == "buy":
                        assert all(x.kind in ("fukusho", "ev3t") for x in msel) and (f["rule"]["fired"] or f["cal"]["fired"])
                    else:
                        assert len(msel) <= 1 and all(x.kind == "ev3t" for x in msel)
                        assert (msel == []) == (f["cal"]["reason"] != "ev_below_min")
                else:
                    assert 1 <= len(msel) <= 2 and all(x.kind in ("fukusho", "tansho") and x.combo[0] in "複単" and x.combo[1:] in "123456" for x in msel)
    sc = daily.score_pending()
    assert sc["scored"] == out["predicted"] * 6  # 本体＋絞り込み型＋4モード（◎×は入れていないので加味版は無い）
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
