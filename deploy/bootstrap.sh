#!/usr/bin/env bash
# Boatrace Lab — VPS 1行セットアップ（Ubuntu 24.04, root で実行）
#
#   curl -fsSL https://raw.githubusercontent.com/<USER>/<REPO>/main/deploy/bootstrap.sh | sudo REPO_URL=https://github.com/<USER>/<REPO>.git bash
#
# やること: スワップ4GB / Docker / コード取得 / .env生成（パスワード自動発行）/
#           HTTPS(Caddy + <IP>.sslip.io) / API・スケジューラ起動 / 初回データ取込と学習（バックグラウンド）
set -euo pipefail

REPO_URL="${REPO_URL:?REPO_URL を指定してください（例: REPO_URL=https://github.com/user/boatrace-lab.git）}"
INSTALL_DIR="${INSTALL_DIR:-/opt/boatlab}"

echo "== [1/7] スワップ 4GB（2GB RAM で月次再学習を行うため）"
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "== [2/7] Docker"
if ! command -v docker >/dev/null; then
  apt-get update -y
  apt-get install -y ca-certificates curl git
  curl -fsSL https://get.docker.com | sh
fi
systemctl enable --now docker

echo "== [3/7] コード取得 → ${INSTALL_DIR}"
if [ -d "${INSTALL_DIR}/.git" ]; then
  git -C "${INSTALL_DIR}" pull --ff-only
else
  git clone --depth 1 "${REPO_URL}" "${INSTALL_DIR}"
fi
cd "${INSTALL_DIR}"
mkdir -p data

echo "== [4/7] .env 生成（初回のみ）"
IP=$(curl -fsS -4 https://ifconfig.me || hostname -I | awk '{print $1}')
DOMAIN="${DOMAIN:-$(echo "${IP}" | tr '.' '-').sslip.io}"
if [ ! -f .env ]; then
  PASS=$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 16)
  SECRET=$(tr -dc 'A-Za-z0-9' </dev/urandom | head -c 48)
  cat > .env <<ENV
DOMAIN=${DOMAIN}
BOATLAB_PASSWORD=${PASS}
BOATLAB_SECRET=${SECRET}
ENV
  chmod 600 .env
fi

echo "== [5/7] ビルドと起動（app / scheduler / caddy）"
docker compose build app
docker compose up -d app scheduler caddy

echo "== [6/7] 初回データ取込＋学習をバックグラウンドで開始（2〜4時間）"
nohup docker compose --profile init run --rm init \
  > "${INSTALL_DIR}/data/init.log" 2>&1 &

echo "== [7/7] 完了情報"
PASS=$(grep BOATLAB_PASSWORD .env | cut -d= -f2)
cat <<DONE | tee /root/boatlab-info.txt

========================================================
 Boatrace Lab セットアップ起動完了
   URL       : https://${DOMAIN}
   パスワード : ${PASS}
   （この情報は /root/boatlab-info.txt にも保存済み）

 初回のデータ取込と学習が裏で進行中です（2〜4時間）。
   進捗確認 : tail -f ${INSTALL_DIR}/data/init.log
 完了後、翌朝 07:30(JST) から自動運用が始まります。
 URL をスマホで開き、ログインしてホーム画面に追加してください。
========================================================
DONE
