# boatlab — 引き継ぎメモ（Claude Code 用）

競艇3連単の「研究・予想システム」。当てる予想を出すことではなく、過去データで検証しながら
予想モデルを作り・改善し続けることが目的。開発の経緯と設計は `docs/00`〜`docs/10` を参照。

## 絶対に守るルール（リーク防止）
- レース結果を予想生成に一切使わない。予想は結果取得より前に保存する。
- 過去の予想は書き換えない（`predictions` テーブルは追記専用。トリガで UPDATE/DELETE を拒否）。
- 特徴量は対象レースの前日までの実績だけを使う（`boatlab/features/asof.py` の as-of 原則）。
- バックテストの封印テスト期間はモデル版ごとに1回だけ評価する。
- 見送りレースも仮想採点して記録する。モデル・設定はバージョン管理する。
- 正直さを最優先。「当たるように見せる」ことはしない。効かなければ効かないと報告する。

## いまの状態（2026-09-04 時点）
- 本番: KAGOYA VPS で稼働（PWA + スケジューラ）。`deploy/`、手順は `docs/10`。更新は VPS で
  `sudo bash /opt/boatlab/deploy/update.sh`。
- 採用モデル: **Model 1.0**（active）。Model 1.1（3seedアンサンブル）は shadow 並走候補で未採用。
- 買い方: **絞り込み型**（role='focused'、`FocusedParams`、docs/04 §16）を15点固定と並行して毎日保存。PWA上部トグルで切替。
- 資金配分（15点固定側）: **確率比例1乗**（`settings_versions.extra.staking`）。均等と回収率は同等だが「的中しても
  赤字」の割合が下がるため採用。`boatlab/model/staking.py`。
- これまでの正直な結論: 現行モデルに市場優位はない（3連単 log-loss 市場3.70 vs モデル3.78）。
  控除率は実測25%。回収率100%超えの買い方は見つかっていない。詳細は `reports/backtest/`。

## 進行中の作業: fs2 特徴量の効果検証
- 新特徴量42個を7グループに分けて追加済み（`boatlab/features/build.py` の `FEATURE_GROUPS`）:
  entry(進入) / st(実ST) / kimarite(決まり手) / weather(場×コース×風) / form(直近フォーム) /
  exh_trust(展示信頼度) / parts(部品交換)。リーク検査済み。
- 検証: `scripts/run_fs2_experiments.py`。基準値→各グループ単独→効いたものの組合せ、を四半期WFで比較。
  1実験=1サブプロセス（6GBメモリ対策）、期ごとに `.ckpt.pkl` へチェックポイント（途中落ち再開可）。
- 結果は `reports/backtest/fs2_experiments.{csv,md}` に追記。判定は 3連単 log-loss が基準比 −0.001 以上。
- 結果（run_fs2_fast.py、単一分割）: form −0.0058 / exh_trust −0.0014 が有効、form+exh_trust=3.8090（base 3.8171）。他は誤差。
- 次: form+exh_trust で Model 1.2 を全期間WF＋ROI＋封印テストで本検証→shadow並走→PWAでユーザーが採用判断。

## データの置き場所（重要）
- `data/`（lab.db 2.2GB、features 1.3GB、probstores 3.2GB、models）は **.gitignore でリポジトリに入らない**。
- 別マシンで動かすときは lab.db を作り直す必要がある: `lab init-db` → `lab ingest-history --from 2018-01-01
  --to <今日>`（数時間）。特徴量キャッシュは `build_entry_dataset` が初回に自動生成する（1〜2時間）。
- VPS には lab.db がある（構築時に生成済み）。fs2 実験用の probstores はクラウドのサンドボックスにのみ存在。

## よく使うコマンド
```
pip install -e .                         # 開発インストール
python -m pytest tests -q                # テスト（DBが要るものは data/lab.db が必要）
lab init-db                              # テーブル+トリガ作成
lab ingest-history --from 2018-01-01 --to 2026-08-30
lab train --version 1.2 --seeds 7,17,27 # モデル学習→candidate登録
lab predict --version 1.0 --stage final # 締切前レースに予想保存（追記専用）
lab score                               # 結果が出た予想を採点
python scripts/run_backtest_campaign.py # 検証キャンペーン一式
python scripts/run_fs2_experiments.py   # fs2 グループ効果検証（進行中）
```

## コード地図
- `boatlab/ingest/` 取得層（Open API v3 / turnmark / 公式odds3t、原本キャッシュ、レート制限）
- `boatlab/store/` SQLite スキーマ・追記専用トリガ・書き込み
- `boatlab/features/` as-of 特徴量（asof.py が要）、fs2 拡張、履歴読込
- `boatlab/model/` strength(LightGBM) / trifecta(PL) / calibration / market / selection(15点) / staking / pipeline
- `boatlab/backtest/` dataset / walkforward(ProbStore) / metrics
- `boatlab/ops/` daily(ingest/predict/score/train) / scheduler(日次ループ)
- `boatlab/api/app.py` FastAPI + PWA(`web/`)
- `scripts/` バックテスト・実験の実行スクリプト

## 優先指標の順（設計方針）
回収率 ＞ 損益 ＞ 的中率 ＞ 期待値 ＞ 最大DD ＞ 最大連敗 ＞ 穴回収率 ＞ 直近成績。
実戦投入条件の具体値は後で決める（設定で変更可）。自動購入はしない。最終判断はユーザー。
