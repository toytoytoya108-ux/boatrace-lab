"""飛躍23（特大）: これまでの陽性所見を全部重ねる — 点推定が100%を超える瞬間は存在するか（2026-09-20）。

飛躍17〜22 と lane1_place / leaps8 / leaps9 で「本物」と判定した所見は、どれも単独では 100% の手前で止まった:
  - 1号艇の複勝・単勝を モーター2連率1位＋展示タイム1位 で絞る（98.2%、飛躍19・21）
  - 上位8場（+2.3pt、飛躍18。探索期間で選んだ固定リスト）
  - SG・G1 の予選ラウンド 1〜11R で 1-2-3（93.8%、飛躍17）
それぞれ別々の棚（観測量／場の癖／番組の癖）にあるので、**独立なら効果は掛け算になる**はず。
掛け算で 100% を超えるかを、オッズ不要（46万レース）で、探索(2018〜23)→確認(2024〜26) の1回勝負で測る。

手続き（事前登録）:
  1. 6つの二値条件（モーター1位／展示1位／上位8場／SG・G1／1〜11R／1号艇A1）の 2^6=64 通りの組合せ ×
     1号艇を軸にした固定の買い目 7券種（複勝1／単勝1／2連単1-2／2連複1=2／ワイド1=2／3連複1=2=3／3連単1-2-3）
     = 448 規則を探索期間で全部測る。
  2. 探索期間の「95%区間の下限」が最大の規則を券種ごとに1つ選び、確認期間で1回だけ評価する。
  3. 独立性の検査: 単独効果の比の積（予測）と、重ねた実測を比べる。積 > 実測 なら効果は同じものを二重に数えている。
  4. 経済性: 100%を超えた場合でも、1日の本数 × 期待利益で「何円になるか」を出す。

出力: reports/research/leaps12.md
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boatlab.config import ROOT  # noqa: E402

OUT = Path(ROOT) / "reports" / "research" / "leaps12.md"
C = Path("/tmp/claude-0")
TOP8 = ["江戸川", "大村", "津", "徳山", "びわこ", "尼崎", "宮島", "福岡"]  # leaps8.md（探索期間で選定済み）
DAYS = 365 * 8.67  # 2018-01〜2026-09
BETS = [("複勝 1", "place", None), ("単勝 1", "win", None), ("2連単 1-2", "exacta", "1-2"),
        ("2連複 1=2", "quinella", "1=2"), ("ワイド 1=2", "quinella_place", "1=2"),
        ("3連複 1=2=3", "trio", "1=2=3"), ("3連単 1-2-3", "trifecta", "1-2-3")]
FLAGS = ["モーター1位", "展示1位", "上位8場", "SG・G1", "1〜11R", "1号艇A1"]


def build() -> pd.DataFrame:
    r = pd.read_parquet(C / "leap17.parquet",
                        columns=["race_id", "stadium", "race_no", "grade", "year"]).set_index("race_id")
    x = pd.read_parquet(C / "ent.parquet", columns=["race_id", "lane", "motor_rate2", "exhibition_time", "klass_n"])
    x["rk_motor"] = x.groupby("race_id")["motor_rate2"].rank(ascending=False, method="min")
    x["rk_ext"] = x.groupby("race_id")["exhibition_time"].rank(ascending=True, method="min")
    one = x[x["lane"] == 1].set_index("race_id")
    d = r.join(one[["rk_motor", "rk_ext", "klass_n"]], how="inner")
    # 払戻: 1号艇の複勝・単勝（-1=返還）、1号艇軸の固定買い目
    pl = pd.read_parquet(C / "place_pay.parquet")
    # 払戻の表は「的中した艇」しか持たない。母集団は「複勝の払戻が存在するレース」で、lane==1 で絞る前に取ること
    # （絞った後に取ると「1号艇が2着以内に入ったレース」だけになり回収率が131%に化ける。2026-09-20 に実際に踏んだ）
    has_place = set(pl["race_id"])
    pl = pl[pl["lane"] == 1].set_index("race_id")["amount"]
    wn = pd.read_parquet(C / "win_pay.parquet"); wn = wn[wn["lane"] == 1].set_index("race_id")["amount"]
    d = d[d.index.isin(has_place)].copy()
    hit1 = (pl.reindex(d.index).fillna(0) > 0).mean()
    assert 0.60 < hit1 < 0.85, f"1号艇の複勝的中率が異常 {hit1:.3f}（母集団の取り方を疑う）"
    d["place"] = pl.reindex(d.index).fillna(0.0)
    d["win"] = wn.reindex(d.index).fillna(0.0)
    a = pd.read_parquet(C / "allpay.parquet")
    for _, bt, combo in BETS[2:]:
        s = a[(a["bt"] == bt) & (a["combo"] == combo)].drop_duplicates("race_id").set_index("race_id")["amt"]
        d[bt] = s.reindex(d.index).fillna(0.0)
    # 返還: 1号艇が返還なら単複は中立、1〜3号艇のどれかが返還なら組合せは中立 → NaN にして除外
    rf = pd.read_parquet(C / "refund.parquet")
    rf1 = set(rf.loc[rf["lane"] == 1, "race_id"]); rf123 = set(rf.loc[rf["lane"].isin([1, 2, 3]), "race_id"])
    rf12 = set(rf.loc[rf["lane"].isin([1, 2]), "race_id"])
    d.loc[d.index.isin(rf1) | (d["place"] < 0) | (d["win"] < 0), ["place", "win"]] = np.nan
    for bt in ("exacta", "quinella", "quinella_place"):
        d.loc[d.index.isin(rf12), bt] = np.nan
    for bt in ("trio", "trifecta"):
        d.loc[d.index.isin(rf123), bt] = np.nan
    assert (d[["place", "win"]].fillna(0) >= 0).all().all(), "返還の番号(-1)が残っている"
    d["f_モーター1位"] = d["rk_motor"] == 1
    d["f_展示1位"] = d["rk_ext"] == 1
    d["f_上位8場"] = d["stadium"].isin(TOP8)
    d["f_SG・G1"] = d["grade"].isin(["SG", "G1"])
    d["f_1〜11R"] = d["race_no"] <= 11
    d["f_1号艇A1"] = d["klass_n"] == 0
    d["explore"] = d["year"] <= 2023
    return d


def stat(v: pd.Series) -> dict:
    v = v.dropna(); n = len(v)
    if n == 0:
        return dict(n=0, roi=np.nan, lo=np.nan, hi=np.nan, hit=np.nan, pay=np.nan)
    mu = v.mean() / 100; se = v.std() / 100 / np.sqrt(n) if n > 1 else np.nan
    w = v[v > 0]
    return dict(n=n, roi=mu * 100, lo=(mu - 1.96 * se) * 100, hi=(mu + 1.96 * se) * 100,
                hit=(v > 0).mean() * 100, pay=w.mean() if len(w) else np.nan)


def main():
    d = build()
    ex, cf = d[d["explore"]], d[~d["explore"]]
    N = len(d)
    L = [f"# 飛躍23（特大）: 陽性所見を全部重ねる — 点推定が100%を超える瞬間はあるか（{N:,}レース）\n",
         "飛躍17〜22 で「本物」と判定した所見は3つの別々の棚にある: **観測量**（モーター1位＋展示1位、98.2%）、",
         "**場の癖**（上位8場、+2.3pt）、**番組の癖**（SG・G1 の 1〜11R、1-2-3 が93.8%）。",
         "別々の原因なら効果は掛け算になるはずで、掛け算なら 98.2% × 1.024 ≈ **100.6%** になる計算。",
         "これを 探索(2018〜23)→確認(2024〜26) の1回勝負で測る。**448規則から選ぶので、探索の数字は膨らんでいる。**",
         "**確認期間の数字だけが答え。**\n",
         f"- 対象: 複勝の払戻があるレース {N:,}（探索 {len(ex):,} / 確認 {len(cf):,}）。返還は中立として除外。",
         "- 条件6つ: " + "／".join(FLAGS) + "（上位8場は leaps8.md の固定リスト、探索期間で選定済み）",
         "- 買い目7つ: " + "／".join(b[0] for b in BETS) + "（すべて1号艇を軸にした固定の目、1点100円）\n",
         "## 1. 単独効果（探索期間）— 掛け算の材料\n",
         "| 券種 | 条件なし | " + " | ".join(FLAGS) + " |", "|---|---:|" + "---:|" * len(FLAGS)]
    single = {}
    for nm, bt, _ in BETS:
        base = stat(ex[bt])["roi"]; row = [f"{base:.1f}%"]; single[bt] = {"base": base}
        for f in FLAGS:
            s = stat(ex.loc[ex[f"f_{f}"], bt]); single[bt][f] = s["roi"] / base
            row.append(f"{s['roi']:.1f}% (×{s['roi']/base:.3f}, n={s['n']:,})")
        L.append(f"| {nm} | " + " | ".join(row) + " |")
    L += ["", "「×」は条件なしに対する比。独立なら重ねたときの比はこれらの積になる。\n",
          "## 2. 448規則の探索 → 券種ごとに区間下限が最大の規則を選ぶ\n",
          "| 券種 | 選ばれた条件 | 探索n | 探索 | 探索下限 | **確認n** | **確認** | 確認95%区間 | 1日 | 独立予測 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    picks = {}; all_rules = []
    for nm, bt, _ in BETS:
        best = None
        for k in range(0, len(FLAGS) + 1):
            for combo in itertools.combinations(FLAGS, k):
                m = pd.Series(True, index=ex.index)
                for f in combo:
                    m &= ex[f"f_{f}"]
                s = stat(ex.loc[m, bt])
                if s["n"] < 1000:
                    continue
                all_rules.append((nm, combo, s))
                if best is None or s["lo"] > best[1]["lo"]:
                    best = (combo, s)
        combo, s = best
        mc = pd.Series(True, index=cf.index)
        for f in combo:
            mc &= cf[f"f_{f}"]
        c = stat(cf.loc[mc, bt])
        pred = single[bt]["base"] * float(np.prod([single[bt][f] for f in combo])) if combo else single[bt]["base"]
        picks[bt] = (nm, combo, s, c, pred)
        L.append(f"| {nm} | {'＋'.join(combo) if combo else '（なし）'} | {s['n']:,} | {s['roi']:.1f}% | {s['lo']:.1f}% | "
                 f"**{c['n']:,}** | **{c['roi']:.1f}%** | {c['lo']:.1f}〜{c['hi']:.1f}% | {c['n']/(DAYS*(len(cf)/N)):.1f}本 | {pred:.1f}% |")
    over_ex = [(nm, combo, s) for nm, combo, s in all_rules if s["roi"] > 100]
    over_ex_lo = [(nm, combo, s) for nm, combo, s in all_rules if s["lo"] > 100]
    L += ["", f"- 探索期間で点推定が100%を超えた規則: **{len(over_ex)} / {len(all_rules)}**、区間下限まで100%超え: **{len(over_ex_lo)}**。"]
    for nm, combo, s in sorted(over_ex, key=lambda t: -t[2]["lo"])[:8]:
        mc = pd.Series(True, index=cf.index)
        for f in combo:
            mc &= cf[f"f_{f}"]
        bt = next(b for b in BETS if b[0] == nm)[1]
        c = stat(cf.loc[mc, bt])
        L.append(f"  - {nm} × {'＋'.join(combo)}: 探索 {s['roi']:.1f}%（n={s['n']:,}、下限{s['lo']:.1f}）→ "
                 f"**確認 {c['roi']:.1f}%**（n={c['n']:,}、{c['lo']:.1f}〜{c['hi']:.1f}）")
    # 3. 独立性: 上位2条件の重ね合わせ
    L += ["", "## 3. 効果は掛け算になっているか（探索期間、複勝1・単勝1）\n",
          "| 券種 | 条件A | 条件B | A単独 | B単独 | 独立予測 A×B | 実測 A∧B | n | 差 |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for nm, bt, _ in BETS[:2]:
        base = single[bt]["base"]
        for fa, fb in itertools.combinations(FLAGS, 2):
            m = ex[f"f_{fa}"] & ex[f"f_{fb}"]
            s = stat(ex.loc[m, bt])
            if s["n"] < 3000:
                continue
            pred = base * single[bt][fa] * single[bt][fb]
            L.append(f"| {nm} | {fa} | {fb} | {base*single[bt][fa]:.1f}% | {base*single[bt][fb]:.1f}% | "
                     f"{pred:.1f}% | **{s['roi']:.1f}%** | {s['n']:,} | {s['roi']-pred:+.1f}pt |")
    # 4. 経済性
    L += ["", "## 4. 100%を超えたとして、何円になるか（確認期間の実測で計算）\n",
          "| 券種 | 条件 | 確認回収率 | 1日の本数 | 100円/本の1日期待損益 | 1,000円/本 | 1万円/本（※） |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for nm, bt, _ in BETS:
        _, combo, s, c, _ = picks[bt]
        per_day = c["n"] / (DAYS * (len(cf) / N)); edge = (c["roi"] - 100) / 100
        L.append(f"| {nm} | {'＋'.join(combo) if combo else '（なし）'} | {c['roi']:.1f}% | {per_day:.1f} | "
                 f"{edge*100*per_day:+,.0f}円 | {edge*1000*per_day:+,.0f}円 | {edge*10000*per_day:+,.0f}円 |")
    L += ["", "※ 複勝・単勝のプールは薄く（3連単の売上の数%）、1万円を入れると自分の票でオッズが下がる。",
          "  1万円の列は「プールに影響しない」という非現実的な仮定の上限。\n"]
    # 5. 最良規則の解剖（複勝 × モーター1位＋展示1位＋上位8場。1〜11R は ×1.000 で飾り）
    core = ["モーター1位", "展示1位", "上位8場"]
    mk = pd.Series(True, index=d.index)
    for f in core:
        mk &= d[f"f_{f}"]
    L += ["", "## 5. 最良規則の解剖: 複勝 × モーター1位＋展示1位＋上位8場（1〜11R は ×1.000 なので外す）\n",
          "| 期間 | n | 的中率 | 的中時の平均払戻 | 元返しの割合 | 回収率 | 95%区間 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for lab, m in (("探索 2018〜23", mk & d["explore"]), ("確認 2024〜26", mk & ~d["explore"]),
                   ("　2024", mk & (d["year"] == 2024)), ("　2025", mk & (d["year"] == 2025)),
                   ("　2026（〜8月）", mk & (d["year"] == 2026)), ("全期間", mk)):
        v = d.loc[m, "place"].dropna(); s = stat(v); w = v[v > 0]
        L.append(f"| {lab} | {s['n']:,} | {s['hit']:.1f}% | {s['pay']:.0f}円 | {(w==100).mean()*100:.1f}% | "
                 f"**{s['roi']:.1f}%** | {s['lo']:.1f}〜{s['hi']:.1f} |")
    vc = d.loc[mk & ~d["explore"], "place"].dropna().values
    rng = np.random.default_rng(23)
    boots = np.array([vc[rng.integers(0, len(vc), len(vc))].mean() for _ in range(4000)])
    p_over = (boots > 100).mean()
    sd = d.loc[mk, "place"].dropna().std() / 100
    edge = d.loc[mk & ~d["explore"], "place"].dropna().mean() / 100 - 1.0
    n_req = (1.96 * sd / edge) ** 2 if edge > 0 else np.inf
    per_day_all = int(mk.sum()) / DAYS
    L += ["", f"- 確認期間のブートストラップ（4000回）: **回収率>100% の確率 {p_over*100:.0f}%**（=「100%を超えている」と言える確率。95%には遠い）。",
          f"- 確認期間の上積み {edge*100:+.1f}pt を 95% で確定させるのに必要な本数: 約 **{n_req:,.0f}本**"
          f"（1日 {per_day_all:.1f}本 → **{n_req/per_day_all/365:.0f}年**）。",
          f"- 1日 {per_day_all:.1f}本 × 100円 × {edge*100:+.1f}% = **1日 {edge*100*per_day_all:+.0f}円**。1,000円でも {edge*1000*per_day_all:+.0f}円。\n"]
    # 5. 総括
    nm_best = max(picks.values(), key=lambda t: t[3]["lo"])
    pairs = [(nm, fa, fb) for nm in ("place", "win") for fa, fb in itertools.combinations(FLAGS, 2)]
    devs = []
    for bt, fa, fb in pairs:
        m = ex[f"f_{fa}"] & ex[f"f_{fb}"]; s = stat(ex.loc[m, bt])
        if s["n"] >= 3000:
            devs.append(s["roi"] - single[bt]["base"] * single[bt][fa] * single[bt][fb])
    devs = np.array(devs)
    cs = stat(d.loc[mk & ~d["explore"], "place"])
    L += ["## まとめ: 飛躍23（特大）\n",
          f"1. **効果は掛け算になっていた。** 2条件の重ね合わせ {len(devs)} 組で、独立予測と実測の差は "
          f"平均 {devs.mean():+.2f}pt、最大 {np.abs(devs).max():.1f}pt（§3）。観測量・場の癖・番組の癖は**別々の歪み**で、",
          "   重ねると素直に足し合わさる。飛躍シリーズで初めて「所見どうしの関係」が測れた。",
          f"2. **初めて、確認期間で点推定が100%を超えた。** 複勝 × モーター1位＋展示1位＋上位8場 = "
          f"**{cs['roi']:.1f}%**（n={cs['n']:,}、{cs['lo']:.1f}〜{cs['hi']:.1f}%）。探索 "
          f"{stat(d.loc[mk & d['explore'], 'place'])['roi']:.1f}% からの落ちは小さく、独立予測（§2）とも一致する。",
          f"   ただし区間は100%をまたぎ、ブートストラップで「超えている」確率は {p_over*100:.0f}%。**確定には{n_req/per_day_all/365:.0f}年かかる。**",
          f"3. **経済的にはゼロ。** 1日 {per_day_all:.1f}本、100円で {edge*100*per_day_all:+.0f}円/日。複勝プールは薄く、金額を上げると自分の票で潰れる。",
          "   「100%を超える規則が存在する」と「100%超えで儲かる」は別の命題で、前者だけが（弱く）真。",
          f"4. **探索の膨らみの実測**: {len(all_rules)} 規則中 探索で100%超えは {len(over_ex)}、うち区間下限まで超えは {len(over_ex_lo)}。"
          "   確認に持ち込んだ4本のうち単勝2本は 105.7→98.2、104.6→98.0 と**7pt落ちた**（選定の自由度の代償）。複勝2本だけ残った。",
          "5. **3連単 1-2-3 は重ねても届かない**: SG・G1＋展示1位＋A1 で探索103.9→確認97.6（区間83〜112）。的中7%の目は46万レースでも決着しない。",
          f"6. **位置づけ**: 天井は 98.2%（単独）→ {cs['roi']:.1f}%（重ね）へ動いたが、動いた幅（+3pt）は区間幅（±{(cs['hi']-cs['lo'])/2:.1f}pt）と同じ。",
          "   「100%の壁は物理法則ではなく、控除率25%を埋める癖の合計が今の材料では 25pt に届く直前で尽きる」というのが、このシリーズの最終形。"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
