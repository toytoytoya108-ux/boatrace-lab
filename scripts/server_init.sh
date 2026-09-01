#!/usr/bin/env bash
# 初回構築（init コンテナ内で1回だけ実行）:
#   DB初期化 → 2018年〜昨日の過去データ一括取込（2〜3時間）→ Model 1.0 学習 → 採用
# 冪等: 途中で落ちても再実行すれば続きから進む（取込は原本キャッシュ・冪等書込）。
set -euo pipefail
cd /app

YESTERDAY=$(python -c "from datetime import date, timedelta; print(date.today() - timedelta(days=1))")
echo "== init-db"
lab init-db
echo "== ingest 2018-01-01 .. ${YESTERDAY}（数時間かかります。ログはこのまま流れます）"
lab ingest-history --from 2018-01-01 --to "${YESTERDAY}"
echo "== quality report"
lab quality-report --out data/reports/data_quality.md || true
echo "== train Model 1.0 (hole=20, beta=0.3, decay none)"
python - <<'EOF'
from datetime import date, timedelta
from boatlab.ops.daily import train_and_register
from boatlab.model.selection import SelectionParams
from boatlab.store.db import session_scope
from boatlab.store.models import ModelVersion
from sqlalchemy import select

until = date.today() - timedelta(days=1)
pr = train_and_register(
    "1.0", until, SelectionParams(hole_min_odds=20, beta=0.3),
    description="初版。LightGBM(win/top2/top3)+PL-λ+isotonic校正。検証2024-01〜2025-06で選定。封印テスト: 全レース55.9%/78.9%、購入候補72.5%/76.6%",
    half_life=None, num_rounds=400, train_max_rows=1_200_000, years=9)
with session_scope() as s:
    if not s.execute(select(ModelVersion).where(ModelVersion.status == "active")).scalars().first():
        s.get(ModelVersion, "1.0").status = "active"
print("trained & activated:", pr.version, "until", pr.trained_until)
EOF
echo "== DONE. スケジューラが翌朝から自動運用を開始します。PWA を開いて確認してください。"
