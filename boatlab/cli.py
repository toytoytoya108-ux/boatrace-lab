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
    import pandas as pd
    from sqlalchemy import text as _text

    from boatlab.model.trifecta import PERM_LABELS, PERMS
    from boatlab.store.db import get_engine
    FIRST = np.array([p[0] for p in PERMS])
    SECOND = np.array([p[1] for p in PERMS])
    lab2 = {l: i for i, l in enumerate(PERM_LABELS)}
    eng = get_engine()
    df = pd.read_sql_query(_text("""
        SELECT o.race_id, o.source, o.captured_at, o.odds, res.payouts
        FROM odds_snapshots o LEFT JOIN results res ON res.race_id = o.race_id
        WHERE o.bet_type='3t' AND o.source IN ('official_web','turnmark_final')
        ORDER BY o.race_id, o.captured_at"""), eng)
    if not len(df):
        typer.echo("3連単オッズがありません")
        raise typer.Exit(1)

    def place_prob(js):
        d = _json.loads(js) if isinstance(js, str) else js
        inv = np.zeros(120)
        for k, v in d.items():
            j = lab2.get(k)
            if j is not None and v and float(v) > 0:
                inv[j] = 1.0 / float(v)
        if (inv > 0).sum() < 100:
            return None
        q = inv / inv.sum()
        return np.array([q[(FIRST == a) | (SECOND == a)].sum() for a in range(6)])

    pre, fin, pay = {}, {}, {}
    for _, r in df.iterrows():
        v = place_prob(r["odds"])
        if v is None:
            continue
        (fin if r["source"] == "turnmark_final" else pre)[r["race_id"]] = v
        if r["payouts"] is not None and r["race_id"] not in pay:
            d = _json.loads(r["payouts"]) if isinstance(r["payouts"], str) else r["payouts"]
            pay[r["race_id"]] = {int(str(e["combination"]).strip()): float(e["amount"] or 0)
                                 for e in (d.get("place") or []) if str(e.get("combination", "")).strip().isdigit()}
    both = sorted(set(pre) & set(fin))
    typer.echo(f"締切前 {len(pre)}R / 確定 {len(fin)}R / 突合 {len(both)}R（閾値 {threshold}）")
    if len(both) < 20:
        typer.echo("突合できたレースが少なすぎます。数日ぶん貯めてから再実行してください。")
        raise typer.Exit(1)
    a = np.array([pre[r] for r in both])
    b = np.array([fin[r] for r in both])
    sa, sb = a.argmax(1), b.argmax(1)
    ca, cb = a.max(1), b.max(1)
    typer.echo(f"  いちばん堅い艇が一致: {(sa == sb).mean()*100:.1f}%")
    typer.echo(f"  確率の差: 中央値 {np.median(cb - ca)*100:+.2f}pt / 絶対差の中央値 {np.median(np.abs(cb - ca))*100:.2f}pt")
    pa, pb = ca >= threshold, cb >= threshold
    inter = int((pa & pb).sum())
    typer.echo(f"  買い対象レース: 締切前基準 {int(pa.sum())}R / 確定基準 {int(pb.sum())}R / 共通 {inter}R")
    if pb.sum():
        typer.echo(f"  確定基準で買うべきレースのうち締切前でも選べた割合: {inter / max(int(pb.sum()), 1)*100:.1f}%")
    if pa.sum():
        typer.echo(f"  締切前基準で買ったレースが確定基準でも妥当だった割合: {inter / max(int(pa.sum()), 1)*100:.1f}%")
    # 締切前の選定で実際に買った場合の成績（結果が揃っているものだけ）
    ret, n = [], 0
    for k, r in enumerate(both):
        if not pa[k] or r not in pay or not pay[r]:
            continue
        n += 1
        ret.append(pay[r].get(int(sa[k]) + 1, 0.0))
    if n:
        ret = np.array(ret)
        typer.echo(f"  締切前の選定で複勝1点100円: n={n} 的中{(ret > 0).mean()*100:.1f}% "
                   f"回収{ret.sum()/(100*n)*100:.1f}% 元返し{(ret == 100).mean()*100:.1f}%")
        typer.echo("  ※本数が少ないうちは回収率は大きく振れます。見るべきは上の一致率の方です。")


if __name__ == "__main__":
    app()
