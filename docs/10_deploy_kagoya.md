# 10 サーバー構築手順（KAGOYA CLOUD VPS・スマホのみで完結）

所要：手作業 20〜30分 ＋ 自動構築 2〜4時間（放置）。

## 0. 事前準備（1回だけ）

| 準備 | 目的 | やり方（スマホ） |
|---|---|---|
| GitHub リポジトリ | サーバーがコードを取りに行く場所 | GitHub アプリ or ブラウザで新規リポジトリ作成（Public 推奨。秘密情報は含まれない）→ このタスクに接続 → Claude が push |
| Termius（無料） | サーバーに1行コマンドを貼るため | App Store / Google Play |
| KAGOYA CLOUD VPS 契約 | サーバー本体 | ブラウザ。プラン **2GB**、OS **Ubuntu 24.04** |

## 1. KAGOYA でサーバーを作る

1. コントロールパネル →「インスタンス作成」→ OS: Ubuntu 24.04、スペック: 2GB。
2. 「ログイン用認証キー」を新規作成し、**秘密鍵（.pem）をダウンロード**（iPhone は「ファイル」に保存）。
3. 作成完了後、**IPアドレス**をメモ。

## 2. Termius で接続

1. Termius → Keychain → 「+」→ 先ほどの .pem を読み込み（Import key / ファイルから）。
2. Hosts → 「+」→ Address: サーバーのIP、Username: `root`（KAGOYA Ubuntu の既定）、Key: 読み込んだ鍵 → 保存 → 接続。
3. 初回は「接続を信頼しますか」→ はい。

## 3. 1行コマンドを貼る

```bash
curl -fsSL https://raw.githubusercontent.com/<USER>/<REPO>/main/deploy/bootstrap.sh | sudo REPO_URL=https://github.com/<USER>/<REPO>.git bash
```

`<USER>/<REPO>` は Claude が実際の値に置き換えた行をチャットで渡します。数分で以下が表示されます。

```
 URL       : https://xxx-xxx-xxx-xxx.sslip.io
 パスワード : ************
 初回のデータ取込と学習が裏で進行中です（2〜4時間）。
```

この時点で Termius は閉じて構いません（処理はサーバー側で続きます）。

## 4. 待つ → PWA を開く

* 2〜4時間後、URL をスマホで開いてパスワードでログイン。ホーム画面に追加（iPhone: 共有 →「ホーム画面に追加」）。
* 「モデル」タブに Model 1.0（active）が出ていれば構築完了。翌朝 07:30（JST）から自動運用が始まります。
* 進捗を見たい場合は Termius で `tail -f /opt/boatlab/data/init.log`。

## 5. 日常運用（すべて PWA）

* ホーム：今日の推奨レース・仮想収支・累計・実戦投入判定
* レース：一覧 → 詳細（15点・根拠・結果）
* 成績：期間別・場別・オッズ帯別・校正
* モデル：候補版の採用（ボタン）／バックテスト結果
* 設定：購入候補条件・実戦投入条件

## 6. 更新（新モデル・機能追加のとき）

Claude が GitHub に push → Termius で 1 行：

```bash
sudo bash /opt/boatlab/deploy/update.sh
```

## 7. 構成と安全対策

* Docker Compose 3 サービス：`app`（API+PWA）、`scheduler`（日次ループ）、`caddy`（自動HTTPS、`<IP>.sslip.io`）。
* SQLite（`/opt/boatlab/data/lab.db`）。毎日 02:30 に7世代バックアップ（`data/backups/`）。
* スワップ 4GB（月次再学習用）。メモリ上限：app 700MB / scheduler 1.4GB。
* パスワードは `/opt/boatlab/.env` と `/root/boatlab-info.txt`。変更するときは `.env` を編集して `docker compose up -d app`。
* 公式サイトへのアクセスはオッズ取得のみ（1レース1回・3秒間隔・1日上限1,000）。`BOATLAB_DISABLE_OFFICIAL_ODDS=1` を `.env` に書くと停止（推定オッズで運用）。

## 8. トラブル時

| 症状 | 対処 |
|---|---|
| URL が開かない | 数分待つ（証明書取得中）。`docker compose ps` で caddy が Up か確認 |
| ログインできない | `/root/boatlab-info.txt` のパスワードを確認 |
| 予想が出ない | ホームの「システム状態」でジョブ失敗を確認 → `docker compose logs --tail 200 scheduler` |
| 取込が止まった | `docker compose --profile init run --rm init` を再実行（冪等・続きから） |
