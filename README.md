# boatrace-lab（boatlab）

ボートレースの過去データを統計的に分析し、3連単15点（本線10＋穴5、1点200円）の予想・バックテスト・ペーパートレード・成績検証を行う研究システム。設計書は `docs/` を参照。

## 最重要ルール（コードで担保）

* 予想は `predictions` に追記専用で保存（UPDATE/DELETE はトリガで拒否、`created_at` は現在時刻以外を拒否）。
* 特徴量は「対象レースの前日まで」の実績のみから作る（`boatlab/features/asof.py`、`tests/test_asof_leak.py`）。
* バックテストは本番 `predictions` に書かない。学習／校正ホールドアウト／予測期間を時系列で分離。
* 欠損は NULL のまま。推測値で埋めない。

## セットアップ

```bash
pip install -e ".[dev]"
lab init-db                                   # data/lab.db（SQLite）。Postgres は BOATLAB_DATABASE_URL で切替
lab ingest-history --from 2018-01-01 --to 2026-08-30   # Open API v3（2018〜）＋ turnmark 最終オッズ（2026〜）
lab quality-report                            # reports/data_quality.md
lab stadium-stats                             # reports/stats/*.csv
python scripts/run_backtest_campaign.py       # 検証→探索→封印テスト（数時間）
pytest
```

## 本番運用（VPS）

`docs/10_deploy_kagoya.md` を参照。要点は 1 行:

```bash
curl -fsSL https://raw.githubusercontent.com/<USER>/<REPO>/main/deploy/bootstrap.sh | sudo REPO_URL=https://github.com/<USER>/<REPO>.git bash
```

Docker Compose 3 サービス（`app`=API+PWA, `scheduler`=日次ループ `boatlab/ops/scheduler.py`, `caddy`=HTTPS）。
更新は `sudo bash /opt/boatlab/deploy/update.sh`。

## 構成

```
boatlab/
  config.py        取得元 URL・コード表・レート制限
  ingest/          base(取得基盤) / parsers(JSON→正規化) / history(一括取込)
  store/           models(SQLAlchemy) / db(初期化・トリガ) / writer(冪等書込)
  features/        history(読込) / asof(時点固定集計) / build(特徴量セット fs1)
  model/           strength(LightGBM) / trifecta(PL-λ) / calibration / market(オッズ推定) / selection(15点・判定)
  backtest/        dataset / walkforward(2段階) / metrics
  analytics/       data_quality / stadium_stats / export
scripts/           バックテスト・キャンペーン / Model 1.1 実験 / server_init（初回構築）
deploy/            Dockerfile / Caddyfile / bootstrap.sh / update.sh（docker-compose.yml はリポジトリ直下）
tests/             パーサ・追記専用・as-of リーク検査
docs/              設計書 00〜08
```

## データソース

| 用途 | ソース |
|---|---|
| 過去 出走表・直前情報・結果（2018〜） | Boatrace Open API v3（非公式・MIT） |
| 過去 最終オッズ（2026-01〜） | turnmark/api v1（非公式・MIT） |
| 当日 出走表・直前情報・結果 | BoatraceOpenAPI/api v1（約3分更新） |
| 当日 締切前オッズ | 公式サイト odds3t（1レース1〜2回・3秒間隔・1日上限1,000） |
| 照合 | 公式配布 番組表B／競走成績K／期別成績 |

時刻はすべて JST（naive）。`boatlab.util.now_jst()` を使う。
