# 02 システム構成設計

## 1. 全体像

```
┌──────────────────────── 取得層（差し替え可能なアダプタ群） ────────────────────────┐
│ OfficialDownload(B/K/期別)  OpenAPI(programs/previews/results)  OfficialWebOdds  CsvImport │
│    └ レート制限・キャッシュ・リトライ・取得ログ（fetch_log）を共通ミドルウェアで強制        │
└───────────────────────────────────┬────────────────────────────────────────────┘
                                    ▼  正規化レコード（Pydantic dataclass）
┌──────────────────────────── 保存層（PostgreSQL） ────────────────────────────┐
│ raw_*（取得原本）  →  正規化テーブル（races / entries / results / odds_snapshots …）│
│ predictions（追記専用・改変不可）   scoring（照合結果）   model_registry / metrics     │
└───────────────────────────────────┬────────────────────────────────────────────┘
                                    ▼
┌──────────── 特徴量層（as-of 時点固定） ────────────┐  ┌───────── モデル層 ─────────┐
│ FeatureBuilder(asof=レース締切時刻)                │→ │ 艇別強さ → 120通り確率      │
│  過去レースのみ集計・欠損はNULL・完全性スコア      │  │ 校正 → 市場オッズ推定       │
└──────────────────────────────────────────────────┘  │ 期待値・信頼度 → 15点選定    │
                                                      └────────────┬───────────────┘
        ┌────────────── バックテスト層 ─────────────┐               │
        │ ウォークフォワード／パラメータ探索／指標   │◄──────────────┘（同一コードを共用）
        └──────────────────────────────────────────┘
┌──────────── 運用層 ────────────┐   ┌──────────── API/UI層 ────────────┐
│ スケジューラ（日次・レース前後）│   │ FastAPI(JSON API) + PWA(React)     │
│ 予想確定 → 保存 → 結果 → 採点  │   │ 認証・ダッシュボード・分析画面       │
└────────────────────────────────┘   └──────────────────────────────────┘
```

**設計原則**

1. 取得層と予想ロジックの分離：モデルは正規化テーブルのみを見る。取得元の変更はアダプタ差し替えで吸収。
2. as-of（時点固定）原則：特徴量は「そのレースの締切時刻より前に確定していた情報」だけから計算する関数 `build_features(race_id, asof_ts)` に一本化し、当日予想もバックテストも**同じ関数**を使う。
3. 予想は追記専用：`predictions` は DB トリガで UPDATE/DELETE を禁止。結果は別テーブル。採点は結合ビューで導出。
4. すべてに版を付ける：モデル版・特徴量定義版・選定ロジック版・パラメータ版を予想レコードに刻む。
5. 欠損は欠損のまま：推測値で埋めない。完全性スコアを予想に添付し、UIで警告表示。

## 2. 技術スタック（提案）

| 層 | 採用 | 理由 |
|---|---|---|
| 言語 | Python 3.12 | 統計・機械学習ライブラリ、ユーザーのPython経験 |
| DB | **SQLite（既定・WAL）**、PostgreSQL 16 へは接続文字列の切替のみ | 単一ユーザー・単一ホストで十分、無料枠の小型VMで運用しやすい。追記専用トリガは両方に実装済み（`store/db.py`）。同時アクセスが増えたら Postgres へ |
| ORM/移行 | SQLAlchemy 2 + Alembic | スキーマ版管理 |
| モデル | numpy / pandas / scikit-learn / LightGBM | 校正・GBDT・実績豊富 |
| API | FastAPI + Uvicorn | 軽量・型付き・自動ドキュメント |
| スケジューラ | APScheduler（専用ワーカープロセス1つ。API は別プロセスで複数ワーカー可）＋ 手動実行 CLI | 二重実行を防ぐ。将来 Celery 等へ拡張可 |
| フロント | React + TypeScript + Vite、PWA（vite-plugin-pwa / Workbox）、グラフは Chart.js | モバイルファースト、ホーム画面追加、キャッシュ戦略を制御しやすい |
| 配信/TLS | Caddy（自動HTTPS）で API と静的PWA を同一オリジン配信 | 設定が最小 |
| コンテナ | Docker Compose（app / db / caddy） | どの稼働環境でも同じ構成 |
| 通知（将来） | `notifier` インターフェース（LINE Messaging API / Web Push） | 差し替え可能 |

## 3. 稼働環境の選択肢（ユーザー判断事項 G1）

| 案 | 構成 | 月額目安 | 長所 | 短所 |
|---|---|---|---|---|
| A（推奨） | 小型VPS 1台（例：さくらVPS/ConoHa/Lightsail 1〜2GB）＋Docker Compose | 約700〜1,500円 | 常時稼働・スケジュール確実・スマホからHTTPSで常時アクセス・構成が最も単純 | 少額の固定費、初回のサーバー設定 |
| B | GitHub Actions（定期バッチ）＋ Supabase/Neon（Postgres無料枠）＋ Netlify（PWA） | 0円〜 | 固定費ゼロ、Netlify経験あり | Actions の cron は数分〜数十分遅延し「締切5分前に確定」が不安定。無料枠の実行時間（2,000分/月）を超えやすい。構成が分散し障害切り分けが難しい |
| C | 購入予定の低予算PC（常時稼働）＋ Cloudflare Tunnel/Tailscale でスマホ公開 | 電気代のみ | 固定費ほぼゼロ、資源に余裕 | 停電・再起動・回線に依存、自宅外からのアクセス設定が必要 |

| D（無料・推奨） | Oracle Cloud Always Free の VM 1台（Ampere A1：2 OCPU/12GB、または AMD micro 1GB×2）＋Docker Compose | 0円（本人確認のためクレジットカード登録が必要。無料枠内は課金なし） | 常時稼働の本物のサーバーが無料。Aと同じ構成・同じコード。スマホのブラウザだけで作成〜運用可能（cloud-init で自動構築、更新は GitHub 経由で自動デプロイ） | 2026-06 に無料枠が 4→2 OCPU に予告なく半減した実績があり、将来の縮小リスク。ARM の空き容量が無い時期がある。7日間 CPU・ネットワーク・メモリすべて 20% 未満だと回収されうる（日次バッチで回避、または PAYG 切替で回収対象外） |

**ユーザー回答（2026-08-31）：無料・スマホ運用を希望** → 案D を第一候補、確保できなければ案A（月700〜1,500円）へ。
「スマホだけで運用」の実現方法：(1) サーバー作成は Oracle の Web コンソール（スマホブラウザ）で、用意した cloud-init スクリプトを貼り付けるだけで Docker・アプリ・HTTPS（sslip.io ホスト名＋Caddy）まで自動構築。(2) コード更新は GitHub Actions が VM に SSH デプロイ（ユーザーは何もしない）。(3) 日常操作・設定変更・モデル採用はすべて PWA から。(4) モデルの重い再学習は VM 上で夜間に実行（2 OCPU/12GB で十分。1GB VM しか取れない場合は Codespaces で月次学習し成果物をアップロード）。

無料代替として検討し**不採用**としたもの：GitHub Actions を常駐バッチに使う（利用規約上「ソフトウェアの生産・テスト・配布に無関係な活動」に該当しうる、cron が数十分遅れる）、Cloudflare Workers（Python/LightGBM が動かない）、Render/Koyeb 無料枠（スリープ・性能不足）、Android Termux（iPhone不可、バックグラウンド停止、端末依存で一元管理に反する）。

いずれもコンテナ構成は同一。本文書は案A/Dを前提に書くが、B/Cでもコード変更は不要（スケジューラの起動方法のみ差分）。

## 4. モジュール構成（Pythonパッケージ）

```
boatrace-lab/
├── app/
│   ├── ingest/          # 取得層
│   │   ├── base.py          # Source インターフェース、RateLimiter、Retry、FetchLog
│   │   ├── official_files.py# B/K/期別 LZH → 正規化（Shift_JIS固定書式パーサ＋検証）
│   │   ├── openapi.py       # programs / previews / results JSON
│   │   ├── official_web.py  # odds3t（オッズ）、beforeinfo/raceresult（フォールバック）
│   │   └── csv_import.py    # 手動CSV（過去オッズ・展示など）
│   ├── store/           # SQLAlchemy モデル、リポジトリ、Alembic
│   ├── features/        # as-of 特徴量（選手・モーター・場・コース・気象・展示）
│   ├── model/
│   │   ├── strength.py      # 艇別・コース別 1着/2着/3着 強さモデル
│   │   ├── trifecta.py      # 120通り確率（条件付き順位モデル）
│   │   ├── calibration.py   # 確率校正（isotonic / Platt）と校正評価
│   │   ├── market.py        # 市場オッズ推定モデル（過去オッズ欠如の補完）
│   │   ├── confidence.py    # 信頼度
│   │   ├── ev.py            # 期待値・期待回収率
│   │   ├── selection.py     # 本線10点＋穴5点の選定
│   │   └── registry.py      # モデル版・パラメータ・採用状態
│   ├── backtest/        # ウォークフォワード、パラメータ探索、指標
│   ├── scoring/         # 予想と結果の照合、的中区分、損益
│   ├── analytics/       # 成績集計（期間別・場別・オッズ帯別…）、外れ分析
│   ├── ops/             # 日次ジョブ、スケジューラ、アラート、通知インターフェース
│   ├── api/             # FastAPI ルータ、認証
│   └── cli.py           # `lab ingest`, `lab backtest`, `lab predict`, `lab score` 等
├── web/                 # React PWA
├── docs/
├── tests/               # パーサ・as-of・リーク検査・採点の単体テスト
└── docker-compose.yml
```

## 5. データフローと時刻の定義

時刻はすべて **JST の naive datetime** で保存する（`boatlab.util.now_jst()`）。サーバーのタイムゾーンに依存しない。

| 時刻 | 定義 | 用途 |
|---|---|---|
| `post_time` | 締切予定時刻（B/programs） | as-of 基準、予想確定期限 |
| `preview_snapshots.fetched_at` | 直前情報を取得した時刻（追記専用） | `<= asof_ts` の最新行のみ特徴量に使う |
| `predictions.created_at` | 予想レコード保存時刻（DBサーバー時刻、改変不可） | リーク検査：`created_at < post_time` を採点時に必ず検証 |
| `results.fetched_at` | 結果取得時刻 | `> post_time` を検証 |

**当日フロー**

1. 朝（開催日 07:30）：`BoatraceOpenAPI/api` の `today.json`（3分更新）から出走表 → `races/entries` 登録（今節成績は初回取得値を固定）→ 出走表ベースの「暫定予想（stage=program）」を生成・保存。
2. 各レースの締切約10分前：`today.json` を再取得して直前情報（3分遅れ）を追記。取れていなければ公式 `beforeinfo` を1回取得 → 締切約6分前に公式 `odds3t` を1回取得 → 取得完了イベントで「確定予想（stage=final）」を生成・保存（イベント駆動。締切5分前を過ぎても未取得なら、その時点の情報で欠損警告付きで確定）。
3. レース後（締切+約15分）：`today.json` から結果 → `results/payouts` 保存 → 採点 → 成績集計更新。
4. 翌朝（06:00）：`turnmark/api` の前日ファイルで最終オッズ（120通り）・返還・結果を照合・補完（`odds_snapshots.source='turnmark_final'`）。公式 K/B ファイルは週次で照合（人気順位・レースタイム補完）。事後取得の値は当日予想には使わない。
5. 週次：モデル再学習候補の作成とバックテスト比較（採用は手動承認）。

**リーク防止の技術的担保**

* `predictions` の INSERT 時トリガで `created_at` を `now()` に強制、UPDATE/DELETE を拒否。
* `scoring` ジョブは `prediction.created_at < post_time_at_pred` かつ `< races.post_time` かつ `results.fetched_at > post_time` の場合のみ算入。違反は `invalid_reason` を付けて集計から除外し、アラート。
* 特徴量ビルダーは「対象レースの `race_date` より前のレース」の `results/entries` しか読まないよう、リポジトリ層で強制フィルタ（未来行を注入しても出力が変わらないことを単体テストで検証）。
* 直前情報・気象・オッズは `fetched_at <= asof_ts` のスナップショットだけを使い、使用した行IDと特徴量値そのものを `predictions.features_used` に保存（再現性・監査）。
* バックテストは `predictions` に書かず `backtest_runs / backtest_predictions` に書く。レースごとに同じビルダーを `asof_ts = post_time` で呼び、学習データは `train_end < 検証開始` を強制。
* 進入コースは学習・推論とも「スタート展示の進入（無ければ艇番）」で統一し、実進入（結果）は特徴量に使わない。

## 6. セキュリティ・運用

* 単一ユーザー認証（パスワード＋長期セッションCookie、HTTPSのみ）。API はすべて認証必須。
* シークレット（DB・LLM API キー等）は環境変数。
* バックアップ：日次 `pg_dump` を保存（VPS内＋任意でオブジェクトストレージ）。
* ログ：取得ログ（fetch_log）、ジョブログ（job_run）、アプリログ（JSON）。
* 監視：ジョブ失敗・取得失敗率・予想未確定レース数をダッシュボードに表示（通知は将来）。
