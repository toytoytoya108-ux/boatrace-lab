# 09 実装メモ（Phase 2〜7 の実装状況と設計からの変更点）

更新：2026-08-31

## 実装済み

| Phase | 内容 | 実装 | 検証 |
|---|---|---|---|
| 2 | 取得層（レート制限・リトライ・原本キャッシュ・取得ログ） | `boatlab/ingest/base.py` | 実データ取込で使用 |
| 2 | JSON パーサ（v3 flat / v1 nested、boats の list/dict 両対応） | `ingest/parsers.py` | `tests/test_parsers.py` |
| 2 | DB（SQLite/Postgres 両対応、追記専用トリガ、created_at 検査） | `store/models.py`, `store/db.py` | `tests/test_store.py` |
| 2 | 冪等な書込（初回値固定・追記・UPSERT） | `store/writer.py` | 同上 |
| 2 | 過去一括取込 2018-01〜2026-08（Open API v3 ＋ turnmark 最終オッズ） | `ingest/history.py`, `lab ingest-history` | 品質レポート `reports/data_quality.md` |
| 3 | as-of 特徴量（前日まで／期間ウィンドウ／縮約／相対値） | `features/asof.py`, `features/build.py` | `tests/test_asof_leak.py`（未来・同日行を注入しても不変） |
| 3 | 場別・コース別・決まり手・配当・風波の統計 | `analytics/stadium_stats.py` | `reports/stats/*.csv` |
| 4 | 艇別強さ（LightGBM win/top2/top3、時間減衰重み）、ベースライン M0/M0b | `model/strength.py` | 検証期間で比較 |
| 4 | 120通り確率（PL-λ、λ は holdout で探索） | `model/trifecta.py` | |
| 4 | 校正（combo isotonic → 再正規化、セット isotonic → 信頼度） | `model/calibration.py` | 信頼度帯別の実績表 |
| 4 | 市場オッズ推定（公開情報モデル＋単勝/2連単/3連単払戻で回帰） | `model/market.py` | 2026 実オッズと比較 `market_odds_eval.json` |
| 5 | 期待値・信頼度・15点選定（本線10＋穴5、分散制約、relaxed 記録）・判定 | `model/selection.py` | |
| 6 | ウォークフォワード（確率生成と選定評価の2段階）・指標（Wilson/ブートストラップ/連敗/DD） | `backtest/*` | `scripts/run_backtest_campaign.py` |
| 7 | 学習成果物の保存/読込、当日取込、予想保存（追記専用）、採点（有効性検査） | `model/pipeline.py`, `ops/daily.py` | `tests/test_live_loop.py` |
| 7 | JSON API（認証・当日・レース詳細・成績・内訳・校正・判定・モデル・設定） | `api/app.py` | 手動＋スクリーンショット |
| 7 | PWA（ホーム／レース一覧／詳細／成績／モデル／設定、SW: API は Network First） | `web/` | `docs/screenshots/` |

## 設計からの変更点

1. **過去データの主経路**：公式 B/K ファイルではなく Boatrace Open API v3 JSON（2018〜）。公式ファイルのパーサは未実装（照合用として将来）。人気順位・レースタイムは未取込（`results.trifecta_popularity`, `result_entries.race_time` は NULL）。
2. **DB は SQLite 既定**（Postgres は接続文字列で切替）。時刻は JST naive。
3. **信頼度**：レビュー指摘に従い C = Cal(S) のみ。完全性・不一致は別ゲート（`flags`）。
4. **自前集計の as-of は前日まで**（同日先行レースは使わない）。
5. **フロントは React ではなく単一 HTML の PWA**（ビルド不要でスマホからのデプロイが容易）。React 化は将来可能（API は共通）。
6. `evaluate()` は実オッズ（turnmark 最終）があればそれを使い、無ければ推定オッズ。予想レコードに `odds_source` を記録。

## 追記（2026-09-01）

* スケジューラ実装（`ops/scheduler.py`、依存追加なしの分刻みループ、`tests/test_scheduler.py`）。履歴3年分を日単位でキャッシュし 2GB RAM でも動くようにした。
* デプロイ一式（`deploy/`、`docker-compose.yml`、`scripts/server_init.sh`、`docs/10_deploy_kagoya.md`）。Docker Hub がサンドボックスから遮断されているためイメージのビルドは未検証（compose 構文・スクリプト構文・ネイティブ実行は検証済み）。
* 実地テストで検出・修正したバグ：(1) `today.json` が前日分のまま未更新のとき前日の結果を当日 race_id に書いてしまう → race_id をファイル内日付から採番し、当日以外を全リストから除外。(2) ライブの `result` ブロックが払戻空で存在するため全レースを「中止」と誤判定 → ライブ取込では着順が無ければ結果として扱わない（中止判定は翌日の turnmark 照合）。
* Model 1.1 実験（`scripts/run_model11_experiments.py`）：ハイパーパラメータ・学習データ量・seed アンサンブルを検証期間で比較中。

## 未実装（今後）

* 公式 B/K/期別成績の取込（照合・人気順位・コース別期別成績）
* 公式 `beforeinfo` 取得（Open API 3分更新で代替中）・`odds3t` の実ページ検証（サンドボックスから到達不可。VPS で初回稼働時に確認）
* shadow 運用の自動比較、週次レポート、通知
* 資金配分最適化、場別専用モデル、荒れ度予測、LLM 根拠文

## 実行手順（要約）

```bash
lab init-db
lab ingest-history --from 2018-01-01 --to 2026-08-30
lab quality-report && lab stadium-stats
python scripts/run_backtest_campaign.py dataset compare sweep   # 検証・探索
# reports/backtest/chosen_selection.json を書いてから
python scripts/run_backtest_campaign.py dataset test market      # 封印テスト（1回）
lab train --version 1.0 --until 2026-08-30 --hole-min-odds 20 --beta 0.3
lab activate --version 1.0
lab ingest-today && lab predict --version 1.0 --stage final && lab score
uvicorn boatlab.api.app:app --host 0.0.0.0 --port 8080   # BOATLAB_PASSWORD を設定
```
