"""取込データの品質レポート（件数・欠損率・整合性）。Markdown を出力。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import text

from boatlab.store.db import get_engine


def _df(sql: str) -> pd.DataFrame:
    return pd.read_sql_query(text(sql), get_engine())


def build_report() -> str:
    parts: list[str] = ["# データ品質レポート\n"]

    yearly = _df("""
        SELECT substr(race_date,1,4) AS year,
               COUNT(*) AS races,
               SUM(CASE WHEN status='finished' THEN 1 ELSE 0 END) AS finished,
               SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END) AS cancelled,
               SUM(CASE WHEN closed_at IS NULL THEN 1 ELSE 0 END) AS no_closed_at,
               COUNT(DISTINCT race_date) AS days
        FROM races GROUP BY 1 ORDER BY 1""")
    parts.append("## 年別レース数\n\n" + yearly.to_markdown(index=False) + "\n")

    cov = _df("""
        SELECT substr(r.race_date,1,4) AS year,
               COUNT(*) AS races,
               SUM(CASE WHEN res.race_id IS NULL THEN 1 ELSE 0 END) AS no_result,
               SUM(CASE WHEN res.trifecta IS NULL AND r.status='finished' THEN 1 ELSE 0 END) AS finished_no_trifecta,
               SUM(CASE WHEN res.is_irregular=1 THEN 1 ELSE 0 END) AS irregular,
               ROUND(AVG(res.trifecta_payout),0) AS avg_trifecta_payout,
               SUM(CASE WHEN res.trifecta_payout>=10000 THEN 1 ELSE 0 END)*1.0/COUNT(*) AS manshu_rate
        FROM races r LEFT JOIN results res ON res.race_id=r.id GROUP BY 1 ORDER BY 1""")
    parts.append("## 結果カバレッジ\n\n" + cov.to_markdown(index=False) + "\n")

    ent = _df("""
        SELECT substr(r.race_date,1,4) AS year,
               COUNT(*) AS entries,
               ROUND(AVG(CASE WHEN e.regno IS NULL THEN 1.0 ELSE 0 END),4) AS regno_null,
               ROUND(AVG(CASE WHEN e.klass IS NULL THEN 1.0 ELSE 0 END),4) AS klass_null,
               ROUND(AVG(CASE WHEN e.nat_win_rate IS NULL THEN 1.0 ELSE 0 END),4) AS nat_win_null,
               ROUND(AVG(CASE WHEN e.loc_win_rate IS NULL THEN 1.0 ELSE 0 END),4) AS loc_win_null,
               ROUND(AVG(CASE WHEN e.avg_st IS NULL THEN 1.0 ELSE 0 END),4) AS avg_st_null,
               ROUND(AVG(CASE WHEN e.motor_rate2 IS NULL THEN 1.0 ELSE 0 END),4) AS motor2_null,
               ROUND(AVG(CASE WHEN e.nat_rate3 IS NULL THEN 1.0 ELSE 0 END),4) AS nat3_null
        FROM entries e JOIN races r ON r.id=e.race_id GROUP BY 1 ORDER BY 1""")
    parts.append("## 出走表の欠損率（年別）\n\n" + ent.to_markdown(index=False) + "\n")

    pv = _df("""
        SELECT substr(r.race_date,1,4) AS year,
               COUNT(*) AS rows_,
               ROUND(AVG(CASE WHEN p.exhibition_time IS NULL THEN 1.0 ELSE 0 END),4) AS exh_time_null,
               ROUND(AVG(CASE WHEN p.st_exh IS NULL THEN 1.0 ELSE 0 END),4) AS st_exh_null,
               ROUND(AVG(CASE WHEN p.course IS NULL THEN 1.0 ELSE 0 END),4) AS course_null,
               ROUND(AVG(CASE WHEN p.tilt IS NULL THEN 1.0 ELSE 0 END),4) AS tilt_null
        FROM preview_snapshots p JOIN races r ON r.id=p.race_id
        WHERE p.source='openapi_v3_hist' GROUP BY 1 ORDER BY 1""")
    parts.append("## 直前情報の欠損率（openapi_v3_hist, 年別）\n\n" + pv.to_markdown(index=False) + "\n")

    pv_cov = _df("""
        SELECT substr(r.race_date,1,4) AS year, COUNT(*) AS races,
               SUM(CASE WHEN EXISTS(SELECT 1 FROM preview_snapshots p WHERE p.race_id=r.id) THEN 1 ELSE 0 END) AS with_preview
        FROM races r GROUP BY 1 ORDER BY 1""")
    parts.append("## 直前情報の有無（レース単位）\n\n" + pv_cov.to_markdown(index=False) + "\n")

    cond = _df("""
        SELECT substr(r.race_date,1,4) AS year, c.phase, COUNT(*) AS rows_,
               ROUND(AVG(CASE WHEN c.wind_speed_m IS NULL THEN 1.0 ELSE 0 END),4) AS wind_null,
               ROUND(AVG(CASE WHEN c.wave_cm IS NULL THEN 1.0 ELSE 0 END),4) AS wave_null,
               ROUND(AVG(CASE WHEN c.water_temp_c IS NULL THEN 1.0 ELSE 0 END),4) AS water_temp_null
        FROM race_conditions c JOIN races r ON r.id=c.race_id WHERE c.source='openapi_v3_hist'
        GROUP BY 1,2 ORDER BY 1,2""")
    parts.append("## 気象の欠損率\n\n" + cond.to_markdown(index=False) + "\n")

    odds = _df("""
        SELECT substr(r.race_date,1,7) AS month, COUNT(DISTINCT r.id) AS races,
               COUNT(DISTINCT o.race_id) AS races_with_3t_odds
        FROM races r LEFT JOIN odds_snapshots o ON o.race_id=r.id AND o.bet_type='3t'
        WHERE r.race_date >= '2026-01-01' GROUP BY 1 ORDER BY 1""")
    parts.append("## 3連単オッズのカバレッジ（2026〜, turnmark）\n\n" + odds.to_markdown(index=False) + "\n")

    chk = _df("""
        SELECT
          (SELECT COUNT(*) FROM races r WHERE (SELECT COUNT(*) FROM entries e WHERE e.race_id=r.id) <> 6) AS races_not_6_entries,
          (SELECT COUNT(*) FROM races r WHERE r.status='finished' AND (SELECT COUNT(*) FROM result_entries x WHERE x.race_id=r.id) <> 6) AS finished_not_6_result_entries,
          (SELECT COUNT(*) FROM results res WHERE res.trifecta IS NOT NULL AND res.trifecta <>
             (SELECT CAST(a.lane AS TEXT)||'-'||CAST(b.lane AS TEXT)||'-'||CAST(c.lane AS TEXT)
              FROM result_entries a, result_entries b, result_entries c
              WHERE a.race_id=res.race_id AND b.race_id=res.race_id AND c.race_id=res.race_id
                AND a.finish_pos=1 AND b.finish_pos=2 AND c.finish_pos=3)) AS trifecta_mismatch,
          (SELECT COUNT(*) FROM racers) AS racers,
          (SELECT COUNT(*) FROM fetch_log WHERE ok=0) AS fetch_failures,
          (SELECT COUNT(*) FROM job_run WHERE ok=0) AS failed_jobs""")
    parts.append("## 整合性チェック\n\n" + chk.T.rename(columns={0: "value"}).to_markdown() + "\n")

    kim = _df("""
        SELECT kimarite, COUNT(*) AS n, ROUND(COUNT(*)*1.0/(SELECT COUNT(*) FROM results WHERE kimarite IS NOT NULL),4) AS share
        FROM results WHERE kimarite IS NOT NULL GROUP BY 1 ORDER BY 2 DESC""")
    parts.append("## 決まり手分布（全期間）\n\n" + kim.to_markdown(index=False) + "\n")

    c1 = _df("""
        SELECT r.stadium_code, s.name, COUNT(*) AS n,
               ROUND(AVG(CASE WHEN x.finish_pos=1 THEN 1.0 ELSE 0 END),4) AS course1_win_rate
        FROM result_entries x JOIN races r ON r.id=x.race_id JOIN stadiums s ON s.code=r.stadium_code
        WHERE x.course=1 GROUP BY 1,2 ORDER BY 1""")
    parts.append("## 場別 1コース1着率（整合性の目安：概ね 0.45〜0.65）\n\n" + c1.to_markdown(index=False) + "\n")
    return "\n".join(parts)


def write_report(out: str) -> Path:
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(build_report(), encoding="utf-8")
    return p
