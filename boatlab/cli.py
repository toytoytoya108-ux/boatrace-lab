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
        typer.echo(f"OK: 3連単 {len(rec.odds)} 通り読めています。例 1-2-3={rec.odds.get('1-2-3')}")
        return
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
    try:
        doc = httpx.get(OPENAPI_API_TODAY, timeout=60).json()
        day = list(doc["programs"].values())[0]
        for code, st in day.items():
            rs = _races(st)
            up[int(code)] = (sum(1 for x in rs if x.get("result")), len(rs))
        tot = sum(v[1] for v in up.values())
        typer.echo(f"上流 today.json: {len(up)}場 {tot}レース / 結果あり {sum(v[0] for v in up.values())}")
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
    typer.echo(f"DB: {len(df)}場 {int(df['n'].sum())}レース / 結果あり {int(df['done'].sum())}")
    typer.echo("場      DB結果  上流結果  DBで結果のある最後の締切")
    for _, x in df.iterrows():
        u = up.get(int(x["c"]))
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
    typer.echo(f"{d} の3モード記録（確定予想・仮想）")
    with eng.connect() as c:
        rows = c.execute(_text("""
            SELECT p.role, r.stadium_code, r.race_no, r.closed_at, p.decision, p.skip_reason, p.rationale_text,
                   (SELECT SUM(ps.stake) FROM prediction_selections ps WHERE ps.prediction_id=p.id) stake,
                   sc.valid, sc.hit, sc.pnl
            FROM predictions p JOIN races r ON r.id=p.race_id LEFT JOIN scoring sc ON sc.prediction_id=p.id
            WHERE r.race_date=:d AND p.stage='final' AND p.role IN ('ana','katai','place')
            ORDER BY p.role, r.closed_at""" ), {"d": str(d)}).mappings().all()
    if not rows:
        typer.echo("  記録なし（確定予想は各レースの締切4〜10分前に保存されます）")
        raise typer.Exit()
    est = sum(1 for x in rows if x["role"] == "ana" and x["skip_reason"] == "odds_estimated")
    n_ana = sum(1 for x in rows if x["role"] == "ana")
    typer.echo(f"  実オッズで作られたレース {n_ana - est} / 推定オッズ（市場ベースの2モードは見送り） {est}")
    for role, nm in (("ana", "3連単（穴狙い）"), ("katai", "3連単（堅い予想）"), ("place", "複勝・単勝")):
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
    prefix = f"odds3t/{d:%Y%m%d}/{stadium:02d}_{race:02d}_"
    with get_engine().connect() as c:
        rows = c.execute(_text("SELECT key, path, fetched_at FROM raw_files WHERE source='official_web' AND key LIKE :k "
                               "ORDER BY fetched_at DESC"), {"k": prefix + "%"}).fetchall()
    if not rows:
        typer.echo(f"{STADIUMS.get(stadium)} {race}R: 生HTMLの記録がありません（{prefix}*）")
        raise typer.Exit(1)
    key, path, fa = rows[0]
    p = _P(path) if path else None
    if not p or not p.exists():
        typer.echo(f"記録はあるがファイルが無い: key={key} path={path}")
        raise typer.Exit(1)
    html = p.read_bytes().decode("utf-8", errors="replace")
    typer.echo(f"{STADIUMS.get(stadium)} {race}R  {key}  {fa}  {len(html):,} bytes")
    odds = parse_odds3t(html)
    fin = {k: v for k, v in odds.items() if isinstance(v, (int, float)) and v == v}
    missing = [k for k in _PL if odds.get(k) is None]
    typer.echo(f"  パース: キー {len(odds)} / 数値 {len(fin)} / 数値でない {len(missing)}")
    typer.echo(f"  数値でない組: {missing[:40]}{' …' if len(missing) > 40 else ''}")
    # 表の全セルのうち、「1桁の数字」でも「小数」でもない文字列を集計する（＝パーサが None にしたもの）
    text = _re.sub(r"<!--.*?-->", "", html, flags=_re.S)
    odd_cells = {}
    for tb in _re.findall(r"<table[^>]*>(.*?)</table>", text, flags=_re.S):
        for r in _re.findall(r"<tr[^>]*>(.*?)</tr>", tb, flags=_re.S):
            for cell in _re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, flags=_re.S):
                v = _TAG.sub("", cell).replace("\\n", "").strip()
                if v and not _re.fullmatch(r"\\d", v) and not _re.fullmatch(r"\\d+\\.\\d+", v) and len(v) <= 12:
                    odd_cells[v] = odd_cells.get(v, 0) + 1
    top = sorted(odd_cells.items(), key=lambda x: -x[1])[:25]
    typer.echo("  数字でも小数でもないセル（出現回数）: " + ", ".join(f"{repr(k)}×{n}" for k, n in top))
