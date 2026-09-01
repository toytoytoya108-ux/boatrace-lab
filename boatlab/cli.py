"""CLI: `python -m boatlab.cli <command>` または `lab <command>`。"""
from __future__ import annotations

import logging
from datetime import date

import typer

app = typer.Typer(help="boatlab — ボートレース研究・予想システム", no_args_is_help=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@app.command()
def init_db():
    """テーブルとトリガを作成する。"""
    from boatlab.store.db import init_db as _init
    eng = _init()
    typer.echo(f"initialized: {eng.url}")


@app.command()
def ingest_history(
    start: str = typer.Option(..., "--from", help="YYYY-MM-DD"),
    end: str = typer.Option(..., "--to", help="YYYY-MM-DD"),
    turnmark: bool = typer.Option(True, help="2026-01-01 以降は turnmark（最終オッズ）も取り込む"),
    v3: bool = typer.Option(True, help="Open API v3（programs/previews/results）を取り込む"),
):
    """過去データを一括取込する（冪等・原本キャッシュ付き）。"""
    from boatlab.ingest.history import ingest_range
    from boatlab.store.db import init_db as _init
    _init()
    sources = tuple(s for s, on in (("openapi_v3", v3), ("turnmark", turnmark)) if on)
    out = ingest_range(date.fromisoformat(start), date.fromisoformat(end), sources=sources)
    typer.echo(out)


@app.command()
def quality_report(out: str = typer.Option("reports/data_quality.md", help="出力先 Markdown")):
    """取込済みデータの件数・欠損率・整合性レポートを出力する。"""
    from boatlab.analytics.data_quality import write_report
    path = write_report(out)
    typer.echo(f"written: {path}")


@app.command()
def stadium_stats(out_dir: str = typer.Option("reports/stats")):
    """場別・コース別・決まり手・配当・気象条件の統計を CSV に出力する。"""
    from boatlab.analytics.stadium_stats import write_all
    for p in write_all(out_dir):
        typer.echo(f"written: {p}")


@app.command()
def export_parquet(out_dir: str = typer.Option("data/parquet")):
    """モデリング用に主要テーブルを parquet へ書き出す。"""
    from boatlab.analytics.export import export_all
    for name, n in export_all(out_dir).items():
        typer.echo(f"{name}: {n} rows")


@app.command()
def ingest_today(day: str = typer.Option(None, help="YYYY-MM-DD（省略時は今日）")):
    """当日の出走表・直前情報・結果を取り込む（BoatraceOpenAPI/api）。"""
    from datetime import date as _d
    from boatlab.ingest.history import make_fetcher
    from boatlab.ops.daily import ingest_today as _run
    from boatlab.store.db import init_db as _init
    _init()
    typer.echo(_run(make_fetcher(), _d.fromisoformat(day) if day else None))


@app.command()
def predict(version: str = typer.Option(..., help="モデル版"), stage: str = typer.Option("final"),
            role: str = typer.Option("active"), day: str = typer.Option(None),
            min_minutes: int = typer.Option(4), max_minutes: int = typer.Option(None)):
    """締切前のレースに予想を保存する（追記専用）。"""
    from datetime import date as _d
    from boatlab.model.pipeline import Predictor
    from boatlab.ops.daily import predict_pending
    pr = Predictor.load(version)
    typer.echo(predict_pending(pr, stage, role, _d.fromisoformat(day) if day else None, min_minutes, max_minutes))


@app.command()
def score():
    """結果が出た予想を採点する。"""
    from boatlab.ops.daily import score_pending
    typer.echo(score_pending())


@app.command()
def train(version: str = typer.Option(...), until: str = typer.Option(..., help="学習に使う最終日 YYYY-MM-DD"),
          hole_min_odds: float = typer.Option(20.0), beta: float = typer.Option(0.3),
          half_life: float = typer.Option(0.0, help="半減期（年）。0 で均等"), rounds: int = typer.Option(400),
          seeds: str = typer.Option("7", help="seed アンサンブル（カンマ区切り。例 7,17,27）"),
          max_rows: int = typer.Option(1_200_000), description: str = typer.Option("")):
    """モデルを学習して data/models/<version> に保存し、model_versions に登録（candidate）。"""
    from datetime import date as _d
    from boatlab.model.selection import SelectionParams
    from boatlab.ops.daily import train_and_register
    from boatlab.store.db import init_db as _init
    _init()
    pr = train_and_register(version, _d.fromisoformat(until), SelectionParams(hole_min_odds=hole_min_odds, beta=beta),
                            description, (half_life or None), rounds, train_max_rows=max_rows, years=9,
                            seeds=tuple(int(x) for x in seeds.split(",")))
    typer.echo(f"trained {pr.version} until {pr.trained_until} lam={pr.lam} market={pr.market.fit_report}")


@app.command()
def activate(version: str = typer.Option(...)):
    """モデル版を active にする（他の active は retired）。ユーザーの明示操作。"""
    from sqlalchemy import select
    from boatlab.store.db import session_scope
    from boatlab.store.models import ModelVersion
    with session_scope() as s:
        for mv in s.execute(select(ModelVersion)).scalars():
            if mv.version == version:
                mv.status = "active"
            elif mv.status == "active":
                mv.status = "retired"
    typer.echo(f"active: {version}")


if __name__ == "__main__":
    app()
