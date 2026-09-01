#!/usr/bin/env bash
# コード更新の反映: sudo bash /opt/boatlab/deploy/update.sh
set -euo pipefail
cd "${INSTALL_DIR:-/opt/boatlab}"
git pull --ff-only
docker compose build app
docker compose up -d app scheduler caddy
docker image prune -f
echo "updated: $(git rev-parse --short HEAD)"
