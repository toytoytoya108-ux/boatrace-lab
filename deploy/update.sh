#!/usr/bin/env bash
# コード更新の反映。
#
#   cd /opt/boatlab
#   sudo setsid nohup bash deploy/update.sh > /tmp/update.log 2>&1 < /dev/null &
#   sleep 3; tail -f /tmp/update.log      # Ctrl-C で抜けてもビルドは続く
#
# 2026-09-14 の事故（7:37 で更新が止まり予想が欠けた）を受けて2点を直した。
#
# 罠1: **SSH が切れるとビルドが死ぬ。** フォアグラウンドだと SIGHUP で落ちる。
#   上の setsid nohup が対策。スクリプト側では何もできないので使う側で外す。
# 罠2: **2GB VPS では scheduler(mem_limit 1400m) を止めないと build が OOM する。**
#   sshd ごと詰まって SSH が切れるので罠1と同時に起きる。下で stop してから build する。
#   build が失敗しても scheduler を上げ直す（ERR トラップ）。止まったままが最悪。
# 罠3: **git pull はこのスクリプト自身を書き換える。** bash は実行中のファイルを逐次読むので、
#   長さが変わると次の read がズレた位置に落ちる。pull 直後に自分を exec し直して読み直す。
set -euo pipefail
cd "${INSTALL_DIR:-/opt/boatlab}"

# --- 段階1: pull して自分を読み直す（罠3）
if [ "${BOATLAB_UPDATE_STAGE:-0}" = "0" ]; then
  git pull --ff-only
  export BOATLAB_UPDATE_STAGE=1
  exec bash "$0" "$@"
fi

# --- 段階2: 入れ替え
restore() {
  echo "!! 更新に失敗した。scheduler を上げ直す（予想を止めないため）"
  docker compose up -d app scheduler caddy || true
}
trap restore ERR

echo "== scheduler を止めてメモリを空ける（罠2）"
docker compose stop scheduler
free -h || true

echo "== build"
docker compose build app

echo "== up"
docker compose up -d app scheduler caddy
trap - ERR

docker image prune -f
echo "updated: $(git rev-parse --short HEAD)"
docker compose ps
