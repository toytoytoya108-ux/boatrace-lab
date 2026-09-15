"""CLI: `python -m boatlab.cli <command>` または `lab <command>`。"""
from __future__ import annotations

import logging
from datetime import date

import typer

app = typer.Typer(help="boatlab — ボートレース研究・予想システム", no_args_is_help=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@app.command()
def init_db():
    """テーブルとトリガを作成する。"""
    from boatlab.store.db import init_db as _init
    eng = _init()
    typer.echo(f"initialized: {eng.url}")


@app.command()
def ingest_history(
    start: str = typer.Option(..., "--from", help="YYYY-MM-DD"),
    end: str = typer.Option(..., "--to", help="YYYY-MM-DD"),
    turnmark: bool = typer.Option(True, help="2026-01-01 以降は turnmark（最終オッズ）も取り込む"),
    v3: bool = typer.Option(True, help="Open API v3（programs/previews/results）を取り込む"),
):
    """過去データを一括取込する（冪等・原本キャッシュ付き）。"""
    from boatlab.ingest.history import ingest_range
    from boatlab.store.db import init_db as _init
    _init()
    sources = tuple(s for s, on in (("openapi_v3", v3), ("turnmark", turnmark)) if on)
    out = ingest_range(date.fromisoformat(start), date.fromisoformat(end), sources=sources)
    typer.echo(out)


@app.command()
def quality_report(out: str = typer.Option("reports/data_quality.md", help="出力先 Markdown")):
    """取込済みデータの件数・欠損率・整合性レポートを出力する。"""
    from boatlab.analytics.data_quality import write_report
    path = write_report(out)
    typer.echo(f"written: {path}")


@app.command()
def stadium_stats(out_dir: str = typer.Option("reports/stats")):
    """場別・コース別・決まり手・配当・気象条件の統計を CSV に出力する。"""
    from boatlab.analytics.stadium_stats import write_all
    for p in write_all(out_dir):
        typer.echo(f"written: {p}")


@app.command()
def export_parquet(out_dir: str = typer.Option("data/parquet")):
    """モデリング用に主要テーブルを parquet へ書き出す。"""
    from boatlab.analytics.export import export_all
    for name, n in export_all(out_dir).items():
        typer.echo(f"{name}: {n} rows")


@app.command()
def ingest_today(day: str = typer.Option(None, help="YYYY-MM-DD（省略時は今日）")):
    """当日の出走表・直前情報・結果を取り込む（BoatraceOpenAPI/api）。"""
    from datetime import date as _d
    from boatlab.ingest.history import make_fetcher
    from boatlab.ops.daily import ingest_today as _run
    from boatlab.store.db import init_db as _init
    _init()
    typer.echo(_run(make_fetcher(), _d.fromisoformat(day) if day else None))


@app.command()
def predict(version: str = typer.Option(..., help="モデル版"), stage: str = typer.Option("final"),
            role: str = typer.Option("active"), day: str = typer.Option(None),
            min_minutes: int = typer.Option(4), max_minutes: int = typer.Option(None)):
    """締切前のレースに予想を保存する（追記専用）。"""
    from datetime import date as _d
    from boatlab.model.pipeline import Predictor
    from boatlab.ops.daily import predict_pending
    pr = Predictor.load(version)
    typer.echo(predict_pending(pr, stage, role, _d.fromisoformat(day) if day else None, min_minutes, max_minutes))


@app.command()
def score():
    """結果が出た予想を採点する。"""
    from boatlab.ops.daily import score_pending
    typer.echo(score_pending())


@app.command()
def train(version: str = typer.Option(...), until: str = typer.Option(..., help="学習に使う最終日 YYYY-MM-DD"),
          hole_min_odds: float = typer.Option(20.0), beta: float = typer.Option(0.3),
          half_life: float = typer.Option(0.0, help="半減期（年）。0 で均等"), rounds: int = typer.Option(400),
          seeds: str = typer.Option("7", help="seed アンサンブル（カンマ区切り。例 7,17,27）"),
          max_rows: int = typer.Option(1_200_000), description: str = typer.Option("")):
    """モデルを学習して data/models/<version> に保存し、model_versions に登録（candidate）。"""
    from datetime import date as _d
    from boatlab.model.selection import SelectionParams
    from boatlab.ops.daily import train_and_register
    from boatlab.store.db import init_db as _init
    _init()
    pr = train_and_register(version, _d.fromisoformat(until), SelectionParams(hole_min_odds=hole_min_odds, beta=beta),
                            description, (half_life or None), rounds, train_max_rows=max_rows, years=9,
                            seeds=tuple(int(x) for x in seeds.split(",")))
    typer.echo(f"trained {pr.version} until {pr.trained_until} lam={pr.lam} market={pr.market.fit_report}")


@app.command()
def activate(version: str = typer.Option(...)):
    """モデル版を active にする（他の active は retired）。ユーザーの明示操作。"""
    from sqlalchemy import select
    from boatlab.store.db import session_scope
    from boatlab.store.models import ModelVersion
    with session_scope() as s:
        for mv in s.execute(select(ModelVersion)).scalars():
            if mv.version == version:
                mv.status = "active"
            elif mv.status == "active":
                mv.status = "retired"
    typer.echo(f"active: {version}")


@app.command()
def research():
    """買い方の自動研究を今すぐ実行（夜間ジョブと同じ処理）。"""
    from boatlab.research.strategy import OUT_FILE, run_research
    rep = run_research()
    typer.echo(f"written: {OUT_FILE}  bt={rep['n_bt']} live={rep['n_live']} rolling_roi={rep['rolling']['summary'].get('roi')}")


@app.command("check-oddstf")
def check_oddstf(stadium: int = typer.Option(..., "--stadium"), race: int = typer.Option(..., "--race"),
                 day: str = typer.Option("", "--day", help="YYYY-MM-DD（既定=今日）")):
    """公式の単勝・複勝ページを1回だけ取得して、読めているか確認する（本番サーバーで実行）。

    パーサはサンドボックスから公式サイトに接続できないため実ページで未検証。
    開催中の場・レースを指定して、6艇分のオッズが出れば正常。
    """
    from datetime import date as _date

    from boatlab.config import OFFICIAL_ODDSTF
    from boatlab.ingest.history import make_fetcher
    from boatlab.ingest.official_web import digest_oddstf, fetch_oddstf
    from boatlab.util import now_jst
    d = _date.fromisoformat(day) if day else now_jst().date()
    f = make_fetcher()
    recs = fetch_oddstf(f, d, stadium, race)
    for r in recs:
        typer.echo(f"{r.bet_type}: {r.odds}")
    if recs and any(r.bet_type == "place" for r in recs):
        typer.echo("OK: 単勝・複勝とも6艇そろって読めています。")
        return
    # 失敗（または複勝だけ取れない）ときは構造の要約を出す。これをそのまま貼れば直せる。
    url = OFFICIAL_ODDSTF.format(rno=race, jcd=stadium, yyyymmdd=d.strftime("%Y%m%d"))
    res = f.fetch("official_web", url, f"oddstf/{d:%Y%m%d}/{stadium:02d}_{race:02d}_debug.html", use_cache=False)
    typer.echo("---- 構造の要約（このままチャットに貼ってください）----")
    typer.echo(digest_oddstf(res.content.decode("utf-8", errors="replace")))
    if not recs:
        raise typer.Exit(1)


@app.command("poolgap")
def poolgap_report():
    """単勝・複勝プールの歪みの観測状況を表示する。"""
    from boatlab.research.poolgap import report
    rep = report()
    typer.echo(f"記録 {rep['n']} 件（採点済み {rep.get('n_scored', 0)}）")
    for x in rep.get("summary", []):
        typer.echo(f"  {x['bet_type']}/{x['stage']}: n={x['n']} 的中={x['hit']} 回収={x['roi']} 損益={x['pnl']}円")
    if rep.get("drift"):
        d = rep["drift"]
        typer.echo(f"  締切前→確定のオッズ変化: 中央値{d['median']*100:.1f}% / ±10%以内 {d['within10']*100:.0f}%（{d['n']}件）")


@app.command("check-odds3t")
def check_odds3t(stadium: int = typer.Option(..., "--stadium"), race: int = typer.Option(..., "--race"),
                 day: str = typer.Option("", "--day")):
    """3連単オッズが公式ページから読めているかを確認する（本番サーバーで実行）。

    直近に公式オッズが何件保存されているかも表示する。0 が続いていれば取得できていない。
    """
    from datetime import date as _date

    from sqlalchemy import text as _text

    from boatlab.ingest.history import make_fetcher
    from boatlab.ingest.official_web import digest_odds3t, fetch_odds3t
    from boatlab.store.db import get_engine
    from boatlab.util import now_jst
    try:
        import pandas as pd
        df = pd.read_sql_query(_text("""
            SELECT substr(CAST(race_id AS TEXT),1,8) AS d, bet_type, COUNT(*) AS n
            FROM odds_snapshots WHERE source='official_web' GROUP BY d, bet_type ORDER BY d DESC LIMIT 12"""), get_engine())
        typer.echo("公式サイトから保存できたオッズ（日別）:")
        typer.echo(df.to_string(index=False) if len(df) else "  0件（一度も保存できていません）")
    except Exception as e:
        typer.echo(f"  DB確認に失敗: {e!r}")
    d = _date.fromisoformat(day) if day else now_jst().date()
    f = make_fetcher()
    rec = fetch_odds3t(f, d, stadium, race)
    if rec is not None:
        fin = sum(1 for v in rec.odds.values() if isinstance(v, (int, float)))
        none_keys = [k for k, v in rec.odds.items() if v is None]
        # 2026-09-12: キー数だけ見て「読めている」と誤判定した。予想側は「数値が100通り以上」を条件にするので
        # ここでも同じ基準で判定する
        if fin >= 100:
            typer.echo(f"OK: 3連単 {len(rec.odds)} 通り中 {fin} 通りが数値。例 1-2-3={rec.odds.get('1-2-3')}"
                       + (f"（数値でない組 {len(none_keys)}: {none_keys[:8]}）" if none_keys else ""))
            return
        typer.echo(f"NG: キーは {len(rec.odds)} 通りあるが数値は {fin} 通りだけ。予想は推定オッズに落ちる。")
        typer.echo(f"    数値でない組: {none_keys[:20]}  → `lab dump-odds3t` でセルの文字列を確認")
        raise typer.Exit(1)
    from boatlab.config import OFFICIAL_ODDS3T
    url = OFFICIAL_ODDS3T.format(rno=race, jcd=stadium, yyyymmdd=d.strftime("%Y%m%d"))
    res = f.fetch("official_web", url, f"odds3t/{d:%Y%m%d}/{stadium:02d}_{race:02d}_debug.html", use_cache=False)
    typer.echo("読めませんでした。---- 構造の要約（このままチャットに貼ってください）----")
    typer.echo(digest_odds3t(res.content.decode("utf-8", errors="replace")))
    raise typer.Exit(1)


@app.command("today-status")
def today_status():
    """当日データの取り込み状況を、上流(today.json)とDBの両方で突き合わせる。

    「結果が来ていない」ときに、上流が遅れているのか取り込みが止まっているのかを切り分ける。
    """
    import httpx
    import pandas as pd
    from sqlalchemy import text as _text

    from boatlab.config import OPENAPI_API_TODAY, STADIUMS
    from boatlab.store.db import get_engine
    from boatlab.util import now_jst
    now = now_jst()
    typer.echo(f"現在 {now:%Y-%m-%d %H:%M} JST")

    def _races(st):
        r = st.get("races") or []
        return list(r.values()) if isinstance(r, dict) else list(r)

    up = {}
    up_day = None
    try:
        doc = httpx.get(OPENAPI_API_TODAY, timeout=60).json()
        progs = doc["programs"]
        # 上流は朝のうち前日の番組表を返し続けることがある。**日付を確かめずに突き合わせると、
        # 前日の「結果あり168」と当日の「DB 0件」を並べて故障のように見えてしまう。**
        key = str(now.date())
        up_day = key if key in progs else (list(progs.keys())[0] if progs else None)
        day = progs.get(up_day) or {}
        for code, st in day.items():
            rs = _races(st)
            up[int(code)] = (sum(1 for x in rs if x.get("result")), len(rs))
        tot = sum(v[1] for v in up.values())
        if up_day is None:
            typer.echo("上流 today.json: 番組表が空です")
        else:
            mark = "" if up_day == key else "  ← 今日ではありません（上流はまだ前日の番組表を出しています）"
            typer.echo(f"上流 today.json[{up_day}]: {len(up)}場 {tot}レース / "
                       f"結果あり {sum(v[0] for v in up.values())}{mark}")
    except Exception as e:
        typer.echo(f"上流の取得に失敗: {e!r}")

    df = pd.read_sql_query(_text("""
        SELECT r.stadium_code AS c, COUNT(*) AS n,
               SUM(CASE WHEN res.race_id IS NOT NULL THEN 1 ELSE 0 END) AS done,
               MAX(CASE WHEN res.race_id IS NOT NULL THEN r.closed_at END) AS last_done,
               MAX(res.fetched_at) AS last_fetch
        FROM races r LEFT JOIN results res ON res.race_id = r.id
        WHERE r.race_date = :d GROUP BY r.stadium_code ORDER BY r.stadium_code"""),
        get_engine(), params={"d": str(now.date())})
    typer.echo(f"DB[{now.date()}]: {len(df)}場 {int(df['n'].sum())}レース / 結果あり {int(df['done'].sum())}")
    if len(df) == 0:
        typer.echo("  本日の出走表がまだDBにありません。"
                   + ("取り込みは08:00から5分ごとです（それより前なら正常）。" if f"{now:%H:%M}" < "08:00"
                      else "08:00を過ぎているので scheduler のジョブ状態を確認してください。"))
        if up_day and up_day != str(now.date()):
            typer.echo(f"  上流も {up_day} の番組表のままなので、上流待ちです（異常ではありません）。")
        return
    typer.echo("場      DB結果  上流結果  DBで結果のある最後の締切")
    same_day = (up_day == str(now.date()))
    for _, x in df.iterrows():
        u = up.get(int(x["c"])) if same_day else None
        typer.echo(f"{STADIUMS.get(int(x['c']), x['c']):<6} {int(x['done'])}/{int(x['n'])}"
                   f"     {(str(u[0]) + '/' + str(u[1])) if u else '-':<7}"
                   f"  {str(x['last_done'])[-8:-3] if x['last_done'] else '-'}")
    last = df["last_fetch"].dropna()
    if len(last):
        typer.echo(f"結果の最終取り込み時刻: {max(last)}")


@app.command("odds-drift")
def odds_drift(day: str = typer.Option("", "--day", help="YYYY-MM-DD（既定=前日）")):
    """3連単オッズが締切前から確定までどれだけ動くかを実測する。

    これまでの実オッズ検証は全て「確定オッズ」で行っている。買えるのは締切前なので、
    大きく動くなら過去の回収率はすべて割り引いて読む必要がある（単勝プールはこれで全滅した）。
    """
    import json as _json
    from datetime import date as _date, timedelta as _td

    import numpy as np
    import pandas as pd
    from sqlalchemy import text as _text

    from boatlab.store.db import get_engine
    from boatlab.util import now_jst
    d = _date.fromisoformat(day) if day else (now_jst().date() - _td(days=1))
    lo, hi = int(d.strftime("%Y%m%d")) * 10000, (int(d.strftime("%Y%m%d")) + 1) * 10000
    df = pd.read_sql_query(_text("""
        SELECT race_id, source, captured_at, odds FROM odds_snapshots
        WHERE bet_type='3t' AND race_id >= :lo AND race_id < :hi AND source IN ('official_web','turnmark_final')
        ORDER BY race_id, captured_at"""), get_engine(), params={"lo": lo, "hi": hi})
    if not len(df):
        typer.echo(f"{d}: 3連単オッズのデータがありません")
        raise typer.Exit(1)
    load = lambda v: _json.loads(v) if isinstance(v, str) else v
    pre, fin = {}, {}
    for _, r in df.iterrows():
        (fin if r["source"] == "turnmark_final" else pre)[r["race_id"]] = load(r["odds"])
    both = sorted(set(pre) & set(fin))
    typer.echo(f"{d}: 締切前 {len(pre)}R / 確定 {len(fin)}R / 突合できた {len(both)}R")
    if not both:
        typer.echo("確定オッズがまだ取り込まれていません（翌朝06:10のジョブ後に再実行してください）")
        raise typer.Exit(1)
    rows = []
    for rid in both:
        a, b = pre[rid], fin[rid]
        for k, v in a.items():
            w = b.get(k)
            if v and w and float(v) > 0 and float(w) > 0:
                rows.append((float(v), float(w)))
    x = np.array(rows)
    drift = x[:, 1] / x[:, 0] - 1.0
    typer.echo(f"買い目 {len(x):,} 通りで比較")
    typer.echo(f"  全体: 中央値 {np.median(drift)*100:+.1f}%  10〜90%点 {np.percentile(drift,10)*100:+.0f}〜{np.percentile(drift,90)*100:+.0f}%"
               f"  ±10%以内 {np.mean(np.abs(drift) <= .10)*100:.1f}%")
    typer.echo("  締切前オッズ帯ごと:")
    for a_, b_ in ((1, 5), (5, 10), (10, 20), (20, 50), (50, 200), (200, 1e9)):
        m = (x[:, 0] >= a_) & (x[:, 0] < b_)
        if m.sum() < 30:
            continue
        dd = drift[m]
        lab = f"{a_}〜{b_}倍" if b_ < 1e8 else f"{a_}倍〜"
        typer.echo(f"    {lab:>10}  n={int(m.sum()):6d}  中央値 {np.median(dd)*100:+6.1f}%  ±10%以内 {np.mean(np.abs(dd)<=.10)*100:5.1f}%"
                   f"  10〜90%点 {np.percentile(dd,10)*100:+.0f}〜{np.percentile(dd,90)*100:+.0f}%")
    inv_pre = np.array([sum(1/float(v) for v in pre[r].values() if v and float(v) > 0) for r in both])
    inv_fin = np.array([sum(1/float(v) for v in fin[r].values() if v and float(v) > 0) for r in both])
    typer.echo(f"  Σ(1/オッズ) 中央値: 締切前 {np.median(inv_pre):.3f} / 確定 {np.median(inv_fin):.3f}（1.33前後なら健全）")
    m = (x[:, 0] >= 5) & (x[:, 0] <= 20)
    if m.sum() > 30:
        typer.echo(f"  ※絞り込み型が買う帯（5〜20倍）: 中央値 {np.median(drift[m])*100:+.1f}%、"
                   f"期待値はおおむね {(1+np.median(drift[m]))*100:.0f}% 倍に補正される")


@app.command("favorite-check")
def favorite_check(threshold: float = typer.Option(0.8878, "--threshold",
                   help="市場が見る『2着以内』確率の下限（2026年探索期間の上位10%）")):
    """「堅いレースの複勝」の選定が、締切前オッズでも同じになるかを実測する。

    バックテスト（回収率99.1%）は確定オッズで選んでいる。実際に選ぶのは締切前なので、
    同じレース・同じ艇が選ばれなければ意味がない。単勝プールはここで全滅した。
    ただし今回は個別の買い目ではなく「1着・2着確率の合計」という集約量なので、安定するはず。
    """
    import json as _json

    import numpy as np
    from sqlalchemy import text as _text

    from boatlab.model.trifecta import PERM_LABELS, PERMS
    from boatlab.store.db import get_engine
    FIRST = np.array([p[0] for p in PERMS])
    SECOND = np.array([p[1] for p in PERMS])
    lab2 = {l: i for i, l in enumerate(PERM_LABELS)}

    def place_prob(js):
        d = _json.loads(js) if isinstance(js, str) else js
        if not isinstance(d, dict):
            return None
        inv = np.zeros(120)
        for k, v in d.items():
            j = lab2.get(k)
            if j is not None and v and float(v) > 0:
                inv[j] = 1.0 / float(v)
        if (inv > 0).sum() < 100:
            return None
        q = inv / inv.sum()
        return np.array([q[(FIRST == a) | (SECOND == a)].sum() for a in range(6)])

    eng = get_engine()
    with eng.connect() as con:
        # 締切前オッズを取れているレースだけが対象（全期間を読むとメモリが足りない）
        ids = [r[0] for r in con.execute(_text(
            "SELECT DISTINCT race_id FROM odds_snapshots WHERE bet_type='3t' AND source='official_web'"))]
        if not ids:
            typer.echo("締切前の3連単オッズがまだありません（official_web）。")
            raise typer.Exit(1)
        typer.echo(f"締切前オッズのあるレース: {len(ids)}R。1件ずつ突合します…")
        pre, fin, pay = {}, {}, {}
        for k in range(0, len(ids), 200):                   # 200件ずつ（メモリ対策）
            chunk = ids[k:k + 200]
            ph = ",".join(str(int(x)) for x in chunk)
            for rid, src, js in con.execute(_text(
                    f"SELECT race_id, source, odds FROM odds_snapshots WHERE bet_type='3t' "
                    f"AND race_id IN ({ph}) AND source IN ('official_web','turnmark_final') ORDER BY captured_at")):
                v = place_prob(js)
                if v is not None:
                    (fin if src == "turnmark_final" else pre)[rid] = v
            for rid, js in con.execute(_text(f"SELECT race_id, payouts FROM results WHERE race_id IN ({ph})")):
                d = _json.loads(js) if isinstance(js, str) else js
                if not isinstance(d, dict):
                    continue
                pay[rid] = {int(str(e["combination"]).strip()): float(e["amount"] or 0)
                            for e in (d.get("place") or []) if str(e.get("combination", "")).strip().isdigit()}
    both = sorted(set(pre) & set(fin))
    typer.echo(f"締切前 {len(pre)}R / うち確定も揃った {len(both)}R（閾値 {threshold}）")
    if len(both) < 20:
        typer.echo("突合できたレースが少なすぎます。確定オッズは翌朝06:10に入るので、数日ぶん貯めてから再実行してください。")
        raise typer.Exit(1)
    a = np.array([pre[r] for r in both])
    b = np.array([fin[r] for r in both])
    sa, sb = a.argmax(1), b.argmax(1)
    ca, cb = a.max(1), b.max(1)
    typer.echo(f"  いちばん堅い艇が一致: {(sa == sb).mean()*100:.1f}%")
    typer.echo(f"  確率の差: 中央値 {np.median(cb - ca)*100:+.2f}pt / 絶対差の中央値 {np.median(np.abs(cb - ca))*100:.2f}pt")
    # 締切前の確率は系統的に低く出る（締切間際に本命へ資金が寄るため）。同じ閾値を当てると
    # 対象が減るだけで「選定が一致するか」を測れない。**同じ選定率になるよう閾値を較正**して比べる。
    pb = cb >= threshold
    rate = float(pb.mean())
    if rate <= 0:
        typer.echo(f"  確定基準で閾値 {threshold} を超えたレースが0件。日数を貯めてから再実行してください。")
        raise typer.Exit(0)
    th_pre = float(np.quantile(ca, 1 - rate))
    pa = ca >= th_pre
    inter = int((pa & pb).sum())
    typer.echo(f"  選定率 {rate*100:.1f}% に合わせた締切前の閾値: {th_pre:.4f}（確定は {threshold}）")
    typer.echo(f"  買い対象レース: 締切前 {int(pa.sum())}R / 確定 {int(pb.sum())}R / 共通 {inter}R"
               f" → 重なり {inter / max(int(pb.sum()), 1)*100:.1f}%")
    if int(pb.sum()) >= 5:
        typer.echo(f"  選ばれたレースに限れば艇の一致: {(sa[pb] == sb[pb]).mean()*100:.1f}%")
    need = 40
    if int(pb.sum()) < need:
        typer.echo(f"  ※対象が {int(pb.sum())}R しかありません。重なりの判定には {need}R 以上ほしいので、"
                   f"あと数日ぶん貯めてから再実行してください。")
    ret = [pay[r].get(int(sa[k]) + 1, 0.0) for k, r in enumerate(both) if pa[k] and pay.get(r)]  # 締切前の選定で買った場合
    if ret:
        ret = np.array(ret)
        typer.echo(f"  締切前の選定で複勝1点100円: n={len(ret)} 的中{(ret > 0).mean()*100:.1f}% "
                   f"回収{ret.sum()/(100*len(ret))*100:.1f}% 元返し{(ret == 100).mean()*100:.1f}%")
        typer.echo("  ※本数が少ないうちは回収率は大きく振れます。見るべきは上の一致率の方です。")




@app.command()
def modes(day: str = typer.Option("", "--day", help="YYYY-MM-DD（既定=今日）"),
          stadium: int = typer.Option(0, "--stadium"), race: int = typer.Option(0, "--race")):
    """3モード（穴・堅い・複勝単勝）の記録を表示する。--stadium/--race を付けると、その1レースについて
    最新の公式3連単オッズから今この瞬間の選定を計算して見せる（記録はしない・動作確認用）。"""
    import json as _json
    from datetime import date as _date

    import numpy as np
    from sqlalchemy import text as _text

    from boatlab.config import STADIUMS
    from boatlab.model.modes import MEASURED, ModeParams, market_probs, market_summary, select_ana, select_place
    from boatlab.model.trifecta import PERM_LABELS as _PL
    from boatlab.store.db import get_engine
    from boatlab.util import now_jst
    eng = get_engine()
    d = _date.fromisoformat(day) if day else now_jst().date()
    with eng.connect() as c:
        row = c.execute(_text("SELECT extra FROM settings_versions ORDER BY id DESC LIMIT 1")).fetchone()
    extra = row[0] if row else None
    if isinstance(extra, str):
        try:
            extra = _json.loads(extra)
        except Exception:
            extra = None
    prm = ModeParams.from_dict((extra or {}).get("modes"))
    if stadium and race:
        rid = int(f"{d:%Y%m%d}{stadium:02d}{race:02d}")
        with eng.connect() as c:
            o = c.execute(_text("SELECT odds, captured_at FROM odds_snapshots WHERE race_id=:r AND bet_type='3t' "
                                "AND source='official_web' ORDER BY captured_at DESC LIMIT 1"), {"r": rid}).fetchone()
        if not o:
            typer.echo(f"{STADIUMS.get(stadium)} {race}R: 公式3連単オッズの取得記録がありません（締切6〜12分前に取得されます）")
            raise typer.Exit()
        od = _json.loads(o[0]) if isinstance(o[0], str) else o[0]
        arr = np.array([np.nan if od.get(k) is None else float(od[k]) for k in _PL])
        q = market_probs(arr)
        ms = market_summary(q) if q is not None else None
        typer.echo(f"{STADIUMS.get(stadium)} {race}R  オッズ取得 {o[1]}  有効 {int(np.isfinite(arr).sum())}/120")
        if ms is None:
            typer.echo("  オッズが100通り未満で市場確率を作れません")
            raise typer.Exit()
        typer.echo(f"  市場: 万舟確率 {ms['q_man']:.3f}（穴の条件 ≥{prm.ana_qman_min}）"
                   f"  1着最大 {ms['q1_max']:.3f}（{ms['q1_arg']+1}号艇、単勝の条件 ≥{prm.tansho_q_min}）"
                   f"  2着以内最大 {ms['q2_max']:.3f}（{ms['q2_arg']+1}号艇、複勝の条件 ≥{prm.fukusho_q_min}）")
        a = select_ana(arr, prm)
        typer.echo(f"  穴狙い: {'発火 ' + str(len(a['points'])) + '点 ' + ' '.join(_PL[i] for i in a['points'][:6]) + ' …' if a['fired'] else '見送り（' + str(a['reason']) + '）'}")
        pl = select_place(arr, prm)
        typer.echo(f"  複勝: {'発火 ' + str(pl['fukusho']['lane']) + '号艇' if pl['fukusho']['fired'] else '見送り'}"
                   f"  単勝: {'発火 ' + str(pl['tansho']['lane']) + '号艇' if pl['tansho']['fired'] else '見送り'}")
        typer.echo("  堅い予想はモデルの本線が要るので、記録（下の一覧）で確認してください")
        raise typer.Exit()
    typer.echo(f"{d} の各モード記録（確定予想・仮想）")
    with eng.connect() as c:
        rows = c.execute(_text("""
            SELECT p.role, r.stadium_code, r.race_no, r.closed_at, p.decision, p.skip_reason, p.rationale_text,
                   (SELECT SUM(ps.stake) FROM prediction_selections ps WHERE ps.prediction_id=p.id) stake,
                   sc.valid, sc.hit, sc.pnl
            FROM predictions p JOIN races r ON r.id=p.race_id LEFT JOIN scoring sc ON sc.prediction_id=p.id
            WHERE r.race_date=:d AND p.stage='final' AND p.role IN ('ana','katai','katai_t','honmei','place')
            ORDER BY p.role, r.closed_at""" ), {"d": str(d)}).mappings().all()
    if not rows:
        typer.echo("  記録なし（確定予想は各レースの締切4〜10分前に保存されます）")
        raise typer.Exit()
    est = sum(1 for x in rows if x["role"] == "ana" and x["skip_reason"] == "odds_estimated")
    n_ana = sum(1 for x in rows if x["role"] == "ana")
    typer.echo(f"  実オッズで作られたレース {n_ana - est} / 推定オッズ（市場ベースの2モードは見送り） {est}")
    for role, nm in (("ana", "3連単（穴狙い）"), ("katai", "3連単（堅い予想）"), ("katai_t", "3連単（堅い・上位厚め）"),
                     ("honmei", f"3連単（本命10点・{prm.honmei_multiple:g}倍保証）"), ("place", "複勝・単勝")):
        rs = [x for x in rows if x["role"] == role]
        fired = [x for x in rs if x["decision"] == "buy"]
        scored = [x for x in fired if x["valid"]]
        stake = sum(int(x["stake"] or 0) for x in fired)
        pnl = sum(int(x["pnl"] or 0) for x in scored)
        m = MEASURED.get(role) or MEASURED["fukusho"]
        typer.echo(f"\n[{nm}] 記録 {len(rs)}R / 発火 {len(fired)}R / 投資予定 {stake:,}円"
                   f" / 採点済 {len(scored)}R 的中 {sum(1 for x in scored if x['hit'])} 損益 {pnl:+,}円"
                   f"   （実測の目安: 回収率 {m['roi']*100:.1f}%）")
        for x in fired[:20]:
            res = ("🎯" if x["hit"] else "外れ") + f" {int(x['pnl'] or 0):+,}円" if x["valid"] else "結果待ち"
            typer.echo(f"  {STADIUMS.get(x['stadium_code'])} {x['race_no']:>2}R {str(x['closed_at'])[11:16]}  {int(x['stake'] or 0):>5,}円  {res}")
        if len(fired) > 20:
            typer.echo(f"  … 他 {len(fired)-20}R")
        if role == "ana":
            # タグ別（事前登録した3条件の前向き検証。発火300R以上で タグ付き > タグ無し かつ差5pt以上 が合格）
            from boatlab.model.modes import PREREGISTERED, TAG_NAMES
            import json as _j
            with eng.connect() as c:
                trs = c.execute(_text("""SELECT p.flags, sc.hit, sc.stake_total, sc.payout_total FROM predictions p
                    JOIN scoring sc ON sc.prediction_id=p.id WHERE p.stage='final' AND p.role='ana' AND p.decision='buy' AND sc.valid=1""")).fetchall()
            acc = {}
            for fl, hit, stk, pay in trs:
                fl = _j.loads(fl) if isinstance(fl, str) else (fl or {})
                for k in (fl.get("tags") or ["none"]):
                    a = acc.setdefault(k, [0, 0, 0]); a[0] += 1; a[1] += int(stk or 0); a[2] += int(pay or 0)
            if acc:
                typer.echo("  タグ別（累計・発火のみ）:")
                for k, a in sorted(acc.items(), key=lambda x: -x[1][0]):
                    typer.echo(f"    {'★' if k in PREREGISTERED else ' '} {TAG_NAMES.get(k, 'タグなし'):<28} {a[0]:>4}R  回収率 {a[2]/a[1]*100 if a[1] else 0:.1f}%")
        # 見送りは理由ごとに件数、直近5件は個別に（「候補なし」が正しい見送りか、オッズ未取得かを見分ける）
        skipped = [x for x in rs if x["decision"] != "buy"]
        if skipped:
            from collections import Counter
            cnt = Counter(x["skip_reason"] or "?" for x in skipped)
            typer.echo("  見送り: " + "、".join(f"{k} {v}R" for k, v in cnt.most_common()))
            for x in skipped[-5:]:
                typer.echo(f"    {STADIUMS.get(x['stadium_code'])} {x['race_no']:>2}R {str(x['closed_at'])[11:16]}  "
                           f"{x['skip_reason']}  参考{int(x['stake'] or 0):,}円  {x['rationale_text'] or ''}")


@app.command()
def why_estimated(stadium: int = typer.Option(..., "--stadium"), race: int = typer.Option(..., "--race"),
                  day: str = typer.Option("", "--day", help="YYYY-MM-DD（既定=今日）")):
    """あるレースの確定予想が「実オッズ未取得」になった理由を、DBの中身から切り分ける。

    見るもの: 公式3連単オッズの記録（あるか・いつか・何通りか）、確定予想の時刻と odds_snapshot_id、
    そして predict_pending と同じ照合を再実行して、どの条件で落ちたか。"""
    import json as _json
    from datetime import date as _date

    import numpy as np
    from sqlalchemy import text as _text

    from boatlab.config import STADIUMS
    from boatlab.model.trifecta import PERM_LABELS as _PL
    from boatlab.store.db import get_engine
    from boatlab.util import now_jst
    d = _date.fromisoformat(day) if day else now_jst().date()
    rid = int(f"{d:%Y%m%d}{stadium:02d}{race:02d}")
    eng = get_engine()
    typer.echo(f"{STADIUMS.get(stadium)} {race}R  race_id={rid}  現在 {now_jst():%H:%M} JST")
    with eng.connect() as c:
        r = c.execute(_text("SELECT id, closed_at FROM races WHERE id=:r"), {"r": rid}).fetchone()
        if not r:
            typer.echo("  races に無い（race_id の作り方が違う？）")
            same = c.execute(_text("SELECT id, race_no FROM races WHERE race_date=:d AND stadium_code=:s ORDER BY race_no"),
                             {"d": str(d), "s": stadium}).fetchall()
            typer.echo(f"  同じ場のID: {[x[0] for x in same][:3]}")
            raise typer.Exit()
        typer.echo(f"  締切 {r[1]}")
        snaps = c.execute(_text("SELECT id, source, bet_type, captured_at, odds FROM odds_snapshots WHERE race_id=:r "
                                "ORDER BY captured_at"), {"r": rid}).fetchall()
        typer.echo(f"  オッズ記録 {len(snaps)}件:")
        for sid, src, bt, cap, od in snaps:
            od = _json.loads(od) if isinstance(od, str) else od
            keys = list((od or {}).keys())
            fin = sum(1 for k in _PL if (od or {}).get(k) not in (None, 0, "") and _finite((od or {}).get(k)))
            typer.echo(f"    id={sid} {src} {bt} {cap}  キー数={len(keys)} 例={keys[:2]}  PERM_LABELS と一致する有限値={fin}/120")
        preds = c.execute(_text("SELECT id, role, stage, created_at, odds_snapshot_id, flags, decision, skip_reason "
                                "FROM predictions WHERE race_id=:r ORDER BY created_at"), {"r": rid}).fetchall()
        typer.echo(f"  予想 {len(preds)}件:")
        for pid, role, stage, cat, osid, flags, dec, why in preds:
            fl = _json.loads(flags) if isinstance(flags, str) else (flags or {})
            typer.echo(f"    id={pid} {role}/{stage} {cat}  odds_snapshot_id={osid}  odds_estimated={fl.get('odds_estimated')}  {dec} {why or ''}")
        # predict_pending と同じ照合を、確定予想の時刻を now として再実行
        fin_preds = [p for p in preds if p[2] == "final" and p[1] == "active"]
        if fin_preds:
            from datetime import datetime as _dt
            now = fin_preds[0][3]
            now = _dt.fromisoformat(str(now)) if not isinstance(now, _dt) else now
            typer.echo(f"  照合の再実行（now=確定予想の時刻 {now}）:")
            ok = False
            for sid, src, bt, cap, od in snaps:
                if src != "official_web" or bt != "3t":
                    continue
                cap_dt = _dt.fromisoformat(str(cap)) if not isinstance(cap, _dt) else cap
                age = (now - cap_dt).total_seconds()
                od = _json.loads(od) if isinstance(od, str) else od
                arr = np.array([np.nan if (od or {}).get(k) is None else float(od[k]) for k in _PL])
                fin = int(np.isfinite(arr).sum())
                verdict = "採用" if (age <= 900 and fin >= 100) else ("15分より古い" if age > 900 else "有限値が100未満")
                typer.echo(f"    id={sid} 取得から {age/60:.1f}分  有限値 {fin}/120 → {verdict}")
                ok = ok or verdict == "採用"
            if not ok:
                typer.echo("  → この予想の時点で採用できる official_web の3連単オッズが無かった")
        else:
            typer.echo("  確定予想（active/final）が無い")


def _finite(v) -> bool:
    try:
        import math
        return math.isfinite(float(v))
    except Exception:
        return False


if __name__ == "__main__":
    app()


@app.command()
def dump_odds3t(stadium: int = typer.Option(..., "--stadium"), race: int = typer.Option(..., "--race"),
                day: str = typer.Option("", "--day", help="YYYY-MM-DD（既定=今日）")):
    """保存済みの生HTMLから3連単オッズを再パースし、読めなかった組とそのセルの文字列を見せる。

    「120通りのキーはあるのに値が数値でない」ときに、公式ページのどんな表記をパーサが取りこぼしているかを
    特定する（例: 4桁オッズのカンマ、欠場表記、記号）。"""
    import re as _re
    from datetime import date as _date
    from pathlib import Path as _P

    from sqlalchemy import text as _text

    from boatlab.config import STADIUMS
    from boatlab.ingest.official_web import _TAG, parse_odds3t
    from boatlab.model.trifecta import PERM_LABELS as _PL
    from boatlab.store.db import get_engine
    from boatlab.util import now_jst
    d = _date.fromisoformat(day) if day else now_jst().date()
    # 生HTMLは DB ではなく data/raw/official_web/odds3t/YYYYMMDD/ に保存される（Fetcher.fetch）
    from boatlab.config import RAW_DIR
    folder = _P(RAW_DIR) / "official_web" / "odds3t" / f"{d:%Y%m%d}"
    files = sorted(folder.glob(f"{stadium:02d}_{race:02d}_*.html"))
    if not files:
        typer.echo(f"{STADIUMS.get(stadium)} {race}R: 生HTMLがありません（{folder}/{stadium:02d}_{race:02d}_*.html）")
        have = sorted(x.name for x in folder.glob("*.html"))[:8] if folder.exists() else []
        typer.echo(f"  同じ日にあるファイル例: {have}")
        raise typer.Exit(1)
    p = files[-1]
    key, fa = p.name, p.stat().st_mtime
    html = p.read_bytes().decode("utf-8", errors="replace")
    typer.echo(f"{STADIUMS.get(stadium)} {race}R  {p}  {len(html):,} bytes")
    odds = parse_odds3t(html)
    fin = {k: v for k, v in odds.items() if isinstance(v, (int, float)) and v == v}
    missing = [k for k in _PL if odds.get(k) is None]
    typer.echo(f"  パース: キー {len(odds)} / 数値 {len(fin)} / 数値でない {len(missing)}")
    typer.echo(f"  数値でない組: {missing[:40]}{' …' if len(missing) > 40 else ''}")
    # 表の全セルのうち、「1桁の数字」でも「数値」でもないものを、空セルも含めて生HTMLごと集計する
    text = _re.sub(r"<!--.*?-->", "", html, flags=_re.S)
    odd_cells = {}
    for tb in _re.findall(r"<table[^>]*>(.*?)</table>", text, flags=_re.S):
        for r in _re.findall(r"<tr[^>]*>(.*?)</tr>", tb, flags=_re.S):
            for cell in _re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, flags=_re.S):
                v = _TAG.sub("", cell).replace("\n", "").strip().replace(",", "")
                if not _re.fullmatch(r"\d", v) and not _re.fullmatch(r"\d+\.\d+", v) and not _re.fullmatch(r"\d{2,}", v):
                    k = f"text={v[:20]!r} html={cell.strip()[:70]!r}"
                    odd_cells[k] = odd_cells.get(k, 0) + 1
    top = sorted(odd_cells.items(), key=lambda x: -x[1])[:25]
    typer.echo("  艇番でも数値でもないセル（出現回数）:")
    for k, n in top:
        typer.echo(f"    ×{n}  {k}")


@app.command()
def late_drift(days: int = typer.Option(14, "--days", help="直近何日分を見るか")):
    """締切8分前の帯 → 3分前の帯 → 確定の帯 の入れ替わりと、8分前版／直前版の仮想回収率を比べる。

    予想の時刻を2〜4分前に動かすべきかを数字で決めるための材料（2026-09-13〜記録）。"""
    import json as _json

    import numpy as np
    from sqlalchemy import text as _text

    from boatlab.model.trifecta import PERM_LABELS as _PL
    from boatlab.store.db import get_engine
    from boatlab.util import now_jst
    eng = get_engine()
    since = str((now_jst() - __import__("datetime").timedelta(days=days)).date())
    with eng.connect() as c:
        rows = c.execute(_text("""
            SELECT p.race_id, p.role, p.decision, p.flags, sc.valid, sc.stake_total, sc.payout_total, sc.hit,
                   (SELECT o.odds FROM odds_snapshots o WHERE o.race_id=p.race_id AND o.bet_type='3t' AND o.source='turnmark_final' LIMIT 1) fin
            FROM predictions p JOIN races r ON r.id=p.race_id LEFT JOIN scoring sc ON sc.prediction_id=p.id
            WHERE r.race_date >= :d AND p.stage='final' AND p.role IN ('ana','ana_late')""" ), {"d": since}).mappings().all()
        sels = {}
        for pid, rid, role, combo in c.execute(_text("""
            SELECT p.id, p.race_id, p.role, ps.combo FROM predictions p JOIN prediction_selections ps ON ps.prediction_id=p.id
            JOIN races r ON r.id=p.race_id WHERE r.race_date >= :d AND p.role IN ('ana','ana_late')"""), {"d": since}):
            sels.setdefault((rid, role), set()).add(combo)
    by = {}
    for x in rows:
        by.setdefault(x["race_id"], {})[x["role"]] = dict(x)
    pairs = [(rid, v["ana"], v["ana_late"]) for rid, v in by.items() if "ana" in v and "ana_late" in v]
    typer.echo(f"直近{days}日: 8分前版と直前版の両方があるレース {len(pairs)}")
    # 本命10点: 締切直前でも ×1.5 が保たれているか（`min15.md` の但し書きの実測）
    with eng.connect() as c:
        hl = c.execute(_text("""SELECT p.flags FROM predictions p JOIN races r ON r.id=p.race_id
            WHERE r.race_date >= :d AND p.stage='final' AND p.role='hon_late' AND p.decision='buy'"""),
            {"d": since}).fetchall()
    ms = []
    for (fl,) in hl:
        fl = _json.loads(fl) if isinstance(fl, str) else (fl or {})
        if fl.get("mult") and fl.get("early_mult"):
            ms.append((float(fl["early_mult"]), float(fl["mult"])))
    if ms:
        em = np.array([a for a, _ in ms]); lm = np.array([b for _, b in ms])
        typer.echo(f"\n[本命10点] 8分前と直前の倍率を比べられたレース {len(ms)}")
        typer.echo(f"  8分前の倍率 中央値 {np.median(em):.3f} → 直前 {np.median(lm):.3f}"
                   f"（差の中央値 {np.median(lm - em):+.3f}）")
        typer.echo(f"  直前に倍率が1.5を割った割合 {(lm < 1.5).mean()*100:.1f}%"
                   f" → 割るなら設定の倍率を上げる（例 1.6〜1.7）")
    if not pairs:
        raise typer.Exit()
    ov_el, ov_ef, ov_lf, fired_agree = [], [], [], 0
    roi = {"ana": [0, 0, 0], "ana_late": [0, 0, 0]}
    for rid, e, l in pairs:
        se, sl = sels.get((rid, "ana"), set()), sels.get((rid, "ana_late"), set())
        if se and sl:
            ov_el.append(len(se & sl))
        fin = e["fin"]
        if fin:
            d0 = _json.loads(fin) if isinstance(fin, str) else fin
            inv = np.array([1.0 / float(d0[k]) if d0.get(k) else 0.0 for k in _PL])
            if (inv > 0).sum() >= 100:
                q = inv / inv.sum()
                sf = {_PL[int(i)] for i in np.argsort(-q)[19:40]}
                if se:
                    ov_ef.append(len(se & sf))
                if sl:
                    ov_lf.append(len(sl & sf))
        fired_agree += (e["decision"] == l["decision"])
        for k, x in (("ana", e), ("ana_late", l)):
            if x["decision"] == "buy" and x["valid"]:
                roi[k][0] += 1; roi[k][1] += int(x["stake_total"] or 0); roi[k][2] += int(x["payout_total"] or 0)
    def m(v): return f"{np.mean(v):.1f}/21" if v else "—"
    typer.echo(f"  発火の一致（両方発火 or 両方見送り）: {fired_agree}/{len(pairs)}")
    typer.echo(f"  人気20〜40の帯の重なり: 8分前↔直前 {m(ov_el)}  8分前↔確定 {m(ov_ef)}  直前↔確定 {m(ov_lf)}")
    for k, nm in (("ana", "8分前版（表示している予想）"), ("ana_late", "直前版（記録のみ）")):
        n, st, pay = roi[k]
        typer.echo(f"  {nm}: 発火・採点済 {n}R  回収率 {pay/st*100 if st else 0:.1f}%")
    typer.echo("  ※ 確定との重なりは翌朝06:10の確定オッズ取込後に埋まります")


@app.command("guarantee-check")
def guarantee_check(days: int = typer.Option(7, "--days", help="直近何日分を見るか"),
                    role: str = typer.Option("honmei", "--role", help="honmei / katai")):
    """「当たったのに投資を下回った」を分解する。保証つき配分の検算。

    保証は組み立て時に `賭け金_i × オッズ_i ≥ 倍率 × Σ賭け金` を全点で満たすことで成り立つ。
    破れる経路は3つしかないので、どれなのかを1レースずつ切り分ける。

      A 組み立て不良 … 保存したオッズで計算し直しても倍率に届かない＝こちらの不具合
      B 配当の下振れ … 公式配当 < 予想オッズ×100。締切までの下落、または同着・返還
      C 返還         … 返還点が出て投資が減る（倍率は上がる側なので破れる原因にはならない）
    """
    import json as _j

    from sqlalchemy import text as _text

    from boatlab.config import STADIUMS
    from boatlab.store.db import get_engine
    from boatlab.util import now_jst
    eng = get_engine()
    since = str((now_jst() - __import__("datetime").timedelta(days=days)).date())
    with eng.connect() as c:
        rows = c.execute(_text("""
            SELECT p.id, r.stadium_code AS st, r.race_no AS rno, r.closed_at AS ca,
                   r.race_date AS rd, p.flags,
                   sc.actual_trifecta AS tri, sc.actual_payout AS pay, sc.hit,
                   sc.stake_total AS stk, sc.payout_total AS pot,
                   sc.refunded_points AS rfp, sc.refunded_stake AS rfs
            FROM predictions p
            JOIN races r ON r.id = p.race_id
            JOIN scoring sc ON sc.prediction_id = p.id
            WHERE p.stage='final' AND p.role=:role AND p.decision='buy'
              AND sc.valid=1 AND r.race_date >= :since
            ORDER BY r.closed_at"""), {"role": role, "since": since}).mappings().all()
    if not rows:
        typer.echo(f"直近{days}日に role={role} の発火・採点済レースがありません。")
        return
    with eng.connect() as c:
        sels = {}
        for pid in [x["id"] for x in rows]:
            sels[pid] = c.execute(_text(
                "SELECT combo, stake, odds_at_pred FROM prediction_selections "
                "WHERE prediction_id=:p ORDER BY rank"), {"p": pid}).mappings().all()

    dmin = min(str(x["rd"]) for x in rows); dmax = max(str(x["rd"]) for x in rows)
    per_day = {}
    for x in rows:
        per_day[str(x["rd"])] = per_day.get(str(x["rd"]), 0) + 1
    typer.echo(f"role={role} の保証の検算　対象 {since} 以降（--days {days}）")
    typer.echo(f"発火・採点済 {len(rows)}R　{dmin}〜{dmax}　内訳: "
               + "、".join(f"{k} {v}R" for k, v in sorted(per_day.items())))
    typer.echo("※ --days N は「今日からN日前**以降**」なので N=1 でも2日分が入る\n")
    hits = broke = 0
    causes = {"A 組み立て不良": 0, "B 配当の下振れ": 0}
    for x in rows:
        fl = _j.loads(x["flags"]) if isinstance(x["flags"], str) else (x["flags"] or {})
        mult = float(fl.get("mult") or 0) or None
        ss = sels.get(x["id"]) or []
        stk = int(x["stk"] or 0)
        # 組み立て時点の最小倍率（保存したオッズで再計算する）
        pairs = [(s["combo"], int(s["stake"]), s["odds_at_pred"]) for s in ss]
        ok = [(cb, st, float(od)) for cb, st, od in pairs if od is not None and st > 0]
        built = min((st * od / stk for _, st, od in ok), default=None) if stk else None
        if not x["hit"]:
            continue
        hits += 1
        ratio = (int(x["pot"] or 0) / stk) if stk else 0.0
        win = next((t for t in pairs if t[0] == x["tri"]), None)
        od_pred = float(win[2]) if win and win[2] is not None else None
        st_win = int(win[1]) if win else 0
        pay100 = float(x["pay"] or 0) / 100.0
        nm = (f"{str(x['rd'])[5:]} {STADIUMS.get(int(x['st']), x['st'])} "
              f"{int(x['rno']):>2}R {str(x['ca'])[11:16]}")
        if mult and ratio + 1e-9 >= mult:
            typer.echo(f"  OK   {nm}  投資{stk:,}円 → 払戻{int(x['pot'] or 0):,}円  ×{ratio:.2f}（保証×{mult:.2f}）")
            continue
        broke += 1
        drop = (pay100 / od_pred - 1.0) * 100 if (od_pred and od_pred > 0) else float("nan")
        cause = "A 組み立て不良" if (built is not None and mult and built + 1e-9 < mult and
                                 od_pred and abs(pay100 - od_pred) / od_pred < 0.01) else "B 配当の下振れ"
        causes[cause] += 1
        typer.echo(f"  NG   {nm}  投資{stk:,}円 → 払戻{int(x['pot'] or 0):,}円  "
                   f"**×{ratio:.2f}**（保証×{mult:.2f} のはず）")
        typer.echo(f"       的中 {x['tri']}  賭け金{st_win:,}円  予想オッズ{od_pred if od_pred else '—'}倍 → "
                   f"公式配当{int(x['pay'] or 0):,}円（＝{pay100:.1f}倍・{drop:+.1f}%）")
        typer.echo(f"       組み立て時の最小倍率（保存オッズで再計算）: "
                   f"{('×%.2f' % built) if built is not None else '—'}   返還 {int(x['rfp'] or 0)}点/{int(x['rfs'] or 0):,}円")
        typer.echo(f"       → 原因: {cause}")
    typer.echo(f"\n的中 {hits}R / うち保証割れ {broke}R"
               + (f"（{broke/hits*100:.0f}%）" if hits else ""))
    if broke:
        typer.echo("  内訳: " + "、".join(f"{k} {v}R" for k, v in causes.items() if v))
        typer.echo("  A が出たら実装の不具合。B なら倍率を上げる（設定タブ）か、避けられない同着・返還。")


@app.command("winner-drift")
def winner_drift(days: int = typer.Option(14, "--days", help="直近何日分を見るか"),
                 mult: float = typer.Option(1.5, "--mult", help="確定時点で守りたい倍率")):
    """**当たった目のオッズは、他の目より深く下がるのか。**

    保証つき配分は「予想時のオッズ」で組み、払戻は「確定オッズ」で決まる。
    払戻を決めるのは**実際に当たった1点**であり、当たる目には締切直前の資金が集まりやすい。
    もしそうなら ×mult の保証は構造的に楽観で、揺らぎではなく**偏り**として不足する。

    予想に保存した odds_used と、翌朝取り込む確定オッズ（turnmark_final）を突き合わせて、
    当たった目とそれ以外の下落率を**同じオッズ帯の中で**比べる。
    """
    import json as _j

    import numpy as np
    from sqlalchemy import text as _text

    from boatlab.store.db import get_engine
    from boatlab.util import now_jst
    eng = get_engine()
    since = str((now_jst() - __import__("datetime").timedelta(days=days)).date())
    with eng.connect() as c:
        rows = c.execute(_text("""
            SELECT p.race_id AS rid, p.odds_used AS ou, p.flags AS fl,
                   sc.actual_trifecta AS tri,
                   (SELECT o.odds FROM odds_snapshots o
                     WHERE o.race_id = p.race_id AND o.bet_type='3t' AND o.source='turnmark_final'
                     ORDER BY o.captured_at DESC LIMIT 1) AS fin
            FROM predictions p
            JOIN races r ON r.id = p.race_id
            JOIN scoring sc ON sc.prediction_id = p.id
            WHERE p.stage='final' AND p.role='honmei' AND sc.valid=1
              AND r.race_date >= :since AND sc.actual_trifecta IS NOT NULL
            GROUP BY p.race_id"""), {"since": since}).mappings().all()

    BANDS = [(1, 2), (2, 4), (4, 7), (7, 12), (12, 20), (20, 50), (50, 1e9)]
    win_d = {b: [] for b in BANDS}
    oth_d = {b: [] for b in BANDS}
    win_q = {b: [] for b in BANDS}   # 当たった目の「予想時オッズ」。**aw と同じ帯順で並べる**（zip で揃える）
    n_race = n_est = 0
    for x in rows:
        if not x["fin"]:
            continue
        fl = _j.loads(x["fl"]) if isinstance(x["fl"], str) else (x["fl"] or {})
        if fl.get("odds_estimated"):
            n_est += 1
            continue
        pre = _j.loads(x["ou"]) if isinstance(x["ou"], str) else (x["ou"] or {})
        fin = _j.loads(x["fin"]) if isinstance(x["fin"], str) else (x["fin"] or {})
        if not pre or not fin:
            continue
        n_race += 1
        for cb, a in pre.items():
            b = fin.get(cb)
            if a is None or b is None or not (a > 0 and b > 0):
                continue
            band = next((t for t in BANDS if t[0] <= a < t[1]), None)
            if band is None:
                continue
            if cb == x["tri"]:
                win_d[band].append(b / a - 1.0); win_q[band].append(float(a))
            else:
                oth_d[band].append(b / a - 1.0)

    if not n_race:
        typer.echo(f"直近{days}日に、予想時オッズと確定オッズの両方が揃ったレースがありません。"
                   "（確定オッズは翌朝06:10の取込で入ります）")
        return
    typer.echo(f"当たった目とそれ以外のオッズの動き（予想時 → 確定）　対象 {since} 以降（--days {days}）")
    typer.echo(f"突き合わせできたレース {n_race}（推定オッズのため除外 {n_est}）\n")
    typer.echo("予想時の帯 | 当たった目 中央値 | 件数 | それ以外 中央値 | 件数 | 差")
    typer.echo("|---|---:|---:|---:|---:|---:|")
    aw, ao, aq = [], [], []
    for b in BANDS:
        w, o = win_d[b], oth_d[b]
        aw += w; ao += o; aq += win_q[b]      # aw と aq は同じ順番（帯順）で伸びる
        if len(w) < 5:
            continue
        mw, mo = float(np.median(w)), float(np.median(o)) if o else float("nan")
        lab = f"{b[0]}〜{b[1]:g}倍" if b[1] < 1e9 else f"{b[0]}倍〜"
        typer.echo(f"| {lab} | **{mw*100:+.1f}%** | {len(w)} | {mo*100:+.1f}% | {len(o):,} | "
                   f"{(mw-mo)*100:+.1f}pt |")
    if len(aw) < 5:
        typer.echo("\n当たった目の標本が5件未満。まだ判断できない。")
        return
    MW, MO = float(np.median(aw)), float(np.median(ao))
    typer.echo(f"| **全体** | **{MW*100:+.1f}%** | {len(aw)} | {MO*100:+.1f}% | {len(ao):,} | {(MW-MO)*100:+.1f}pt |")
    typer.echo(f"\n当たった目の下落の分布: 25%点 {np.percentile(aw,25)*100:+.1f}% / "
               f"中央 {MW*100:+.1f}% / 75%点 {np.percentile(aw,75)*100:+.1f}%")
    # ---- 本命10点の「当たり目」が実際に乗る帯だけで見る
    # 全帯を混ぜた中央値は 50倍〜（下落が浅く件数が多い）に引っ張られて**甘く出る**。
    # 保証を壊すのは本線10点の中の当たり目で、その予想時オッズはおおむね 2〜30倍に収まる。
    # **必要倍率はこの帯で決める。**（2026-09-16: 全帯の中央値で ×1.62 としたのは誤りだった）
    LO, HI = 2.0, 30.0
    core = [d for d, q in zip(aw, aq) if LO <= q < HI]
    typer.echo(f"\n## 本命10点の当たり目が乗る帯（{LO:g}〜{HI:g}倍）だけで見る\n")
    if len(core) < 10:
        typer.echo(f"この帯の当たった目が {len(core)} 件しかない。まだ決められない。")
    else:
        ca = np.array(core)
        typer.echo(f"| 分位 | 下落 | ×{mult:.2f} を守るのに必要な組み立て倍率 |")
        typer.echo("|---|---:|---:|")
        for q, nm in ((10, "10%点（10回に9回守る）"), (25, "25%点（4回に3回）"),
                      (50, "中央値（2回に1回）"), (75, "75%点")):
            v = float(np.percentile(ca, q))
            need_q = mult / (1.0 + v) if (1.0 + v) > 0 else float("nan")
            typer.echo(f"| {nm} | {v*100:+.1f}% | **×{need_q:.2f}** |")
        # 「当たって元本割れ」になるのは下落が 1/mult - 1 を下回ったとき
        for m2 in (1.50, 1.62, 1.76, 2.00):
            thr = 1.0 / m2 - 1.0
            typer.echo(f"  ×{m2:.2f} で組むと、当たって元本割れになるのは下落 {thr*100:.0f}% 未満のとき＝"
                       f"**この帯の {float((ca < thr).mean())*100:.0f}%**")
        typer.echo(f"\n（この帯の当たった目 {len(ca)}件。全帯の中央値 {MW*100:+.1f}% より深いのは、"
                   "50倍〜の当たり目の下落が浅く件数が多いため。**混ぜると甘く出る。**）")

    # ---- 予想を締切に近づけたら、この下落はどれだけ縮むか
    # role='honmei'（8分前）と role='hon_late'（3分前）は同じレースの odds_used を別の時刻で持っている。
    # **当たった目**について 早→確定 と 直前→確定 を同じレース集合で比べる（母集団を揃えないと意味がない）。
    with eng.connect() as c:
        lr = c.execute(_text("""
            SELECT e.race_id AS rid, e.odds_used AS oe, l.odds_used AS ol, sc.actual_trifecta AS tri,
                   (SELECT o.odds FROM odds_snapshots o
                     WHERE o.race_id = e.race_id AND o.bet_type='3t' AND o.source='turnmark_final'
                     ORDER BY o.captured_at DESC LIMIT 1) AS fin
            FROM predictions e
            JOIN predictions l ON l.race_id = e.race_id AND l.stage='final' AND l.role='hon_late'
            JOIN races r ON r.id = e.race_id
            JOIN scoring sc ON sc.prediction_id = e.id
            WHERE e.stage='final' AND e.role='honmei' AND sc.valid=1
              AND r.race_date >= :since AND sc.actual_trifecta IS NOT NULL
            GROUP BY e.race_id"""), {"since": since}).mappings().all()
    de, dl = [], []
    for x in lr:
        if not x["fin"]:
            continue
        def _g(v):
            return _j.loads(v) if isinstance(v, str) else (v or {})
        a, b, f = _g(x["oe"]), _g(x["ol"]), _g(x["fin"])
        t = x["tri"]
        oa, ob, of = a.get(t), b.get(t), f.get(t)
        if not (oa and ob and of and oa > 0 and ob > 0 and of > 0):
            continue
        de.append(of / oa - 1.0); dl.append(of / ob - 1.0)
    typer.echo("\n## 予想を締切に近づけたら下落は縮むか（当たった目・同じレースで比較）\n")
    if len(de) < 5:
        typer.echo(f"比較できたレースが {len(de)} 件しかない。直前版（hon_late）の記録が貯まるまで判断できない。")
    else:
        me, ml = float(np.median(de)), float(np.median(dl))
        typer.echo(f"| 予想の時刻 | 当たった目の下落 中央値 | ×{mult:.2f} に必要な組み立て倍率 |")
        typer.echo("|---|---:|---:|")
        typer.echo(f"| いまの8分前 | {me*100:+.1f}% | ×{mult/(1+me):.2f} |")
        typer.echo(f"| 直前（2〜4分前） | {ml*100:+.1f}% | ×{mult/(1+ml):.2f} |")
        typer.echo(f"\n比較したレース {len(de)}件。"
                   + (f"**縮む分は {abs(ml-me)*100:.1f}pt**（必要倍率で ×{mult/(1+me):.2f} → ×{mult/(1+ml):.2f}）。"
                      if ml > me + 0.005 else
                      "**直前にしても下落はほとんど縮まない。** 下落の大半は最後の2〜3分に起きている。"))

    need = mult / (1.0 + MW) if (1.0 + MW) > 0 else float("nan")
    need25 = mult / (1.0 + float(np.percentile(aw, 25))) if (1.0 + float(np.percentile(aw, 25))) > 0 else float("nan")
    typer.echo(f"\n→ 確定時点で ×{mult:.2f} を**中央値で**守るには、予想時に **×{need:.2f}** で組む必要がある。")
    typer.echo(f"   4回に3回守るなら（25%点まで耐える） **×{need25:.2f}**。")
    typer.echo("   ※ 倍率を上げるほど成立レースは減る（`min15.md`: ×1.5で44%、×2.0で16%）。"
               "設定は自動では変えない。")


@app.command("honmei-pace")
def honmei_pace(days: int = typer.Option(7, "--days", help="直近何日分を見るか")):
    """**枠が朝のうちに埋まる**のはなぜか。しきい値の配り方を1日ずつ分解する。

    honmei は「残り枠 ÷ この先に残る成立見込み本数」の分位点をしきい値にする（秘書問題）。
    朝に埋まるなら、原因は次のどれか。記録した flags（p10・threshold・slots_left・races_left）で切り分ける。

      A 成立率の見積もりが低すぎる … races_left を小さく見積もる → しきい値が下がる → 早く埋まる
      B しきい値表が実運用とずれている … live の確率合計が表より高い側に寄っていると、朝から通ってしまう
      C 単にその日の上位レースが朝に集中した … 表もしきい値も正しい（この場合は何もしない）

    表は確定オッズのバックテストで作った。実運用は締切前オッズなので**本命のオッズが高めに出て
    成立しやすく**、成立レースの顔ぶれ自体が変わる。A と B は同じ原因（確定 vs 締切前）から来る。
    """
    import json as _j

    import numpy as np
    from sqlalchemy import text as _text

    from boatlab.config import STADIUMS
    from boatlab.model.modes import ModeParams
    from boatlab.store.db import get_engine
    from boatlab.util import now_jst
    eng = get_engine()
    since = str((now_jst() - __import__("datetime").timedelta(days=days)).date())
    with eng.connect() as c:
        rows = c.execute(_text("""
            SELECT r.race_date AS rd, r.stadium_code AS st, r.race_no AS rno, r.closed_at AS ca,
                   p.decision AS dec, p.skip_reason AS sr, p.flags AS fl
            FROM predictions p JOIN races r ON r.id = p.race_id
            WHERE p.stage='final' AND p.role='honmei' AND r.race_date >= :since
            ORDER BY r.race_date, r.closed_at"""), {"since": since}).mappings().all()
    if not rows:
        typer.echo(f"直近{days}日に honmei の記録がありません。")
        return
    prm = ModeParams()
    grid = np.array(prm.honmei_conf_grid)
    days_d = {}
    for x in rows:
        fl = _j.loads(x["fl"]) if isinstance(x["fl"], str) else (x["fl"] or {})
        days_d.setdefault(str(x["rd"]), []).append((x, fl))

    typer.echo(f"honmei の枠の配り方　対象 {since} 以降（--days {days}）")
    typer.echo(f"設定: 倍率×{prm.honmei_multiple:g}／枠{prm.honmei_slots}／成立率の見積もり {prm.honmei_feasible_rate:g}\n")
    all_p10 = []
    for d, xs in sorted(days_d.items()):
        real = [(x, f) for x, f in xs if x["sr"] != "odds_estimated"]
        ng = [1 for x, f in real if x["sr"] == "no_guarantee"]
        feas = [(x, f) for x, f in real if x["sr"] != "no_guarantee"]
        fired = [(x, f) for x, f in feas if x["dec"] == "buy"]
        rate = (len(feas) / len(real)) if real else 0.0
        all_p10 += [float(f["p10"]) for _, f in feas if f.get("p10") is not None]
        typer.echo(f"── {d}　実オッズ{len(real)}R／成立{len(feas)}R（実際の成立率 **{rate:.2f}**、"
                   f"見積もり {prm.honmei_feasible_rate:g}）／発火{len(fired)}R")
        for x, f in fired:
            p10 = f.get("p10"); thr = f.get("threshold")
            typer.echo(f"     {str(x['ca'])[11:16]}  {STADIUMS.get(int(x['st']), x['st'])} {int(x['rno']):>2}R  "
                       f"確率合計 {p10 if p10 is None else format(float(p10), '.3f')}  "
                       f"≥ しきい値 {thr if thr is None else format(float(thr), '.3f')}  "
                       f"（残り枠{f.get('slots_left')}／残り見込み{f.get('races_left')}本）")
        sf = [x for x, f in feas if x["sr"] == "slots_full"]
        if sf:
            typer.echo(f"     → {str(fired[-1][0]['ca'])[11:16]} で枠を使い切り、以降 {len(sf)}R が slots_full"
                       if fired else f"     → slots_full {len(sf)}R")
    if not all_p10:
        typer.echo("\n確率合計（p10）が記録されていません。")
        return
    a = np.array(all_p10)
    typer.echo(f"\n成立レースの「本線10点の確率合計」 実運用 {len(a):,}件 vs しきい値表（確定オッズのバックテスト）")
    typer.echo("| 分位 | 実運用 | 表 | 差 |")
    typer.echo("|---|---:|---:|---:|")
    for q in (10, 25, 50, 75, 90, 95):
        lv = float(np.percentile(a, q)); gv = float(grid[q])
        typer.echo(f"| {q}% | {lv:.3f} | {gv:.3f} | **{lv-gv:+.3f}** |")
    med_gap = float(np.median(a)) - float(grid[50])
    typer.echo("")
    if med_gap > 0.01:
        pct = float((grid < np.median(a)).mean()) * 100
        typer.echo(f"→ **B: 実運用の確率合計が表より高い側に寄っている**（中央値で {med_gap:+.3f}、"
                   f"実運用の中央値は表の {pct:.0f}% 分位に相当）。")
        typer.echo("   朝から多くのレースがしきい値を超えるので枠が早く埋まる。**表を実運用の分布で作り直す**のが筋。")
    elif med_gap < -0.01:
        typer.echo(f"→ 実運用の確率合計は表より低い（中央値で {med_gap:+.3f}）。枠が埋まらない側の心配。")
    else:
        typer.echo("→ 表と実運用の分布はほぼ一致。枠が早く埋まるなら原因は成立率の見積もり（A）か、"
                   "その日の並び（C）。上の日別の「実際の成立率」と見積もりを比べること。")
