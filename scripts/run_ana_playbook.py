"""穴狙いの「買い方の構造」と「レースの選び方」の通説を、2026年の確定オッズ＋公式払戻で総当たり検証する。

通説の出典（2026-09-12 に収集）: 競艇AIバズーカー、kcbn.jp、kyoutei-navi、okaturi、paris-montagne、funaban、fukuoka-kyotei。
  買い方: 4-全-全 / 2〜4頭×5・6の2着 / 1号艇2着づけ・3着づけ / ボックス / 3連複ボックス / 出目買い / 1号艇消し
  レース: 1号艇が弱い / 前づけ / 荒れる場 / 女子・ルーキー戦 / 人気が割れている / 4カド条件 / 初日

すべて同じ物差し: 各レース100円×点数、払戻は公式。実測÷市場（万舟率）も併記。
出目買いは 2018〜2025 の万舟出目で作り 2026 に当てる（アウトオブサンプル）。

出力: reports/research/ana_playbook.md
"""
from __future__ import annotations

import json
import sqlite3
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from boatlab.backtest.metrics import roi_bootstrap
from boatlab.config import ROOT, STADIUMS
from boatlab.model.trifecta import PERM_LABELS, PERMS

OUT = Path(ROOT) / "reports" / "research" / "ana_playbook.md"
FEAT = Path(ROOT) / "reports" / "research" / "manshu_features.parquet"
DB = str(Path(ROOT) / "data" / "lab.db")
A = np.array([p[0] for p in PERMS]); B = np.array([p[1] for p in PERMS]); C = np.array([p[2] for p in PERMS])
LAB2 = {l: i for i, l in enumerate(PERM_LABELS)}


def load():
    df = pd.read_parquet(FEAT)
    con = sqlite3.connect(DB)
    Q = {}
    for rid, js in con.execute("SELECT race_id, odds FROM odds_snapshots WHERE bet_type='3t' AND source='turnmark_final'"):
        d0 = json.loads(js) if isinstance(js, str) else js
        inv = np.zeros(120)
        for k, v in d0.items():
            j = LAB2.get(k)
            if j is not None and v and float(v) > 0:
                inv[j] = 1.0 / float(v)
        if (inv > 0).sum() >= 110:
            Q[int(rid)] = inv / inv.sum()
    pay3, payr = {}, {}
    for rid, js in con.execute("SELECT race_id, payouts FROM results WHERE race_id >= 202601010000 AND payouts IS NOT NULL"):
        d0 = json.loads(js) if isinstance(js, str) else js
        if not isinstance(d0, dict):
            continue
        t = (d0.get("trifecta") or [{}])[0]
        pay3[int(rid)] = float(t.get("amount") or 0)
        for e in d0.get("trio") or []:
            parts = [x for x in str(e.get("combination") or "").replace("=", "-").split("-") if x.isdigit()]
            if len(parts) == 3:
                payr[(int(rid), frozenset(int(x) for x in parts))] = float(e.get("amount") or 0)
    # 4カド条件・女子/ルーキー用に per-lane の勝率・STとタイトル
    ent = pd.read_sql_query("SELECT race_id, lane, nat_win_rate nwr, avg_st FROM entries WHERE race_id >= 202601010000 AND is_absent=0", con)
    ttl = pd.read_sql_query("SELECT id race_id, title FROM races WHERE race_date >= '2026-01-01'", con)
    con.close()
    d26 = df[df["race_id"].isin(Q) & df["race_id"].isin(pay3)].copy().reset_index(drop=True)
    Qm = np.stack([Q[r] for r in d26["race_id"]])
    d26["q_man"] = np.where(Qm <= 0.0075, Qm, 0.0).sum(1)
    d26["top1_odds"] = 0.75 / Qm.max(1)
    d26["win_idx"] = [LAB2.get(t, -1) for t in d26["trifecta"]]
    d26["pay3"] = [pay3[r] for r in d26["race_id"]]
    piv = ent.pivot_table(index="race_id", columns="lane", values=["nwr", "avg_st"])
    piv.columns = [f"{a}{b}" for a, b in piv.columns]
    d26 = d26.merge(piv, left_on="race_id", right_index=True, how="left").merge(ttl, on="race_id", how="left")
    d26["kado"] = ((d26["nwr4"] > d26["nwr3"]) & (d26["avg_st4"] < d26["avg_st3"])).astype(int)
    d26["rookie_women"] = d26["title"].fillna("").str.contains("ルーキー|女子|レディース|新人").astype(int)
    d26["score"] = ((d26["l1_klass_n"] <= 1).astype(int) + (d26["nwr_gap"] < 0).astype(int)
                    + (d26["l1_ext_rank"] >= 4).astype(int) + (d26["l1_stx_rank"] >= 4).astype(int)
                    + (d26["n_maezuke"] >= 1).astype(int) + (d26["wave"] >= 5).astype(int) + (d26["ws"] >= 6).astype(int))
    return d26, Qm, payr, df


# ---------------------------------------------------------------- 買い方の構造（120通りのマスク、または人気順）
def mask_from(sets):
    """sets = [(1着候補, 2着候補, 3着候補)] の和集合。艇番は1起点。"""
    m = np.zeros(120, bool)
    for f, s, t in sets:
        m |= np.isin(A + 1, f) & np.isin(B + 1, s) & np.isin(C + 1, t)
    return m


ALL = [1, 2, 3, 4, 5, 6]
STRUCT_FIXED = {
    "4-全-全（20点）": mask_from([([4], ALL, ALL)]),
    "2〜4頭・5,6が2着・3着全（24点）": mask_from([([2, 3, 4], [5, 6], ALL)]),
    "2〜4頭・2着全・3着全（60点）": mask_from([([2, 3, 4], ALL, ALL)]),
    "1号艇2着づけ X-1-全（20点）": mask_from([([2, 3, 4, 5, 6], [1], ALL)]),
    "1号艇3着づけ X-全-1（20点）": mask_from([([2, 3, 4, 5, 6], ALL, [1])]),
    "1号艇消し（60点）": mask_from([([2, 3, 4, 5, 6], [2, 3, 4, 5, 6], [2, 3, 4, 5, 6])]),
    "ボックス 2,3,4（6点）": mask_from([([2, 3, 4], [2, 3, 4], [2, 3, 4])]),
    "ボックス 2,3,4,5（24点）": mask_from([([2, 3, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5])]),
    "ボックス 3,4,5,6（24点）": mask_from([([3, 4, 5, 6], [3, 4, 5, 6], [3, 4, 5, 6])]),
    "ボックス 2,4,6（6点）": mask_from([([2, 4, 6], [2, 4, 6], [2, 4, 6])]),
}
TRIO_SETS = {"3連複 ボックス 2,3,4,5（4点）": [2, 3, 4, 5], "3連複 ボックス 2,3,4,5,6（10点）": [2, 3, 4, 5, 6],
             "3連複 ボックス 3,4,5,6（4点）": [3, 4, 5, 6]}


def evaluate_fixed(d, mask, rows_idx):
    n = len(rows_idx); k = int(mask.sum())
    ret = np.array([d["pay3"].values[i] if (d["win_idx"].values[i] >= 0 and mask[d["win_idx"].values[i]]) else 0.0 for i in rows_idx])
    return ret, np.full(n, 100.0 * k)


def evaluate_rank(d, Qm, rows_idx, lo, hi, exclude_l1_head=False, deme=None):
    ret, stake = [], []
    for i in rows_idx:
        q = Qm[i]
        order = np.argsort(-q)
        if exclude_l1_head:
            order = order[A[order] != 0]
        pts = order[lo - 1: hi]
        w = d["win_idx"].values[i]
        ret.append(d["pay3"].values[i] if (w >= 0 and w in set(pts.tolist())) else 0.0)
        stake.append(100.0 * len(pts))
    return np.array(ret), np.array(stake)


def evaluate_trio(d, payr, rows_idx, lanes):
    combos = [frozenset(c) for c in __import__("itertools").combinations(lanes, 3)]
    ret = np.array([sum(payr.get((int(d["race_id"].values[i]), c), 0.0) for c in combos) for i in rows_idx])
    return ret, np.full(len(rows_idx), 100.0 * len(combos))


def summarize(ret, stake):
    lo, hi = roi_bootstrap(stake, ret, n_boot=300)
    hit = (ret > 0).mean()
    return dict(n=len(ret), roi=ret.sum() / stake.sum(), lo=lo, hi=hi, hit=hit,
                avg=(ret[ret > 0].mean() if hit else 0.0), stake=stake.mean())


def main():
    d, Qm, payr, hist = load()
    N = len(d)
    # 出目買い: 2018〜2025 の万舟出目、場ごとの上位20
    h = hist[(hist["year"] <= 2025) & (hist["man"] == 1)]
    deme = {}
    for st, s in h.groupby("stadium_code"):
        top = s["trifecta"].value_counts().head(20).index
        m = np.zeros(120, bool)
        for t in top:
            if t in LAB2:
                m[LAB2[t]] = True
        deme[st] = m

    SEL = {
        "全レース": np.ones(N, bool),
        "市場の万舟確率 上位10%（現行）": d["q_man"].values >= np.quantile(d["q_man"], 0.90),
        "市場の万舟確率 上位30%": d["q_man"].values >= np.quantile(d["q_man"], 0.70),
        "1番人気 12倍以上（人気が割れている）": d["top1_odds"].values >= 12,
        "荒れ条件スコア 4以上": d["score"].values >= 4,
        "1号艇がB級": d["l1_klass_n"].values <= 1,
        "前づけあり": d["n_maezuke"].values >= 1,
        "荒れる場（平和島・鳴門・戸田・江戸川）": d["stadium_code"].isin([4, 14, 2, 3]).values,
        "女子戦・ルーキー戦": d["rookie_women"].values == 1,
        "4カド条件（4号艇が3号艇より勝率高くST速い）": d["kado"].values == 1,
        "節の初日": d["day_no"].values == 1,
        "1号艇の展示タイム4位以下": d["l1_ext_rank"].values >= 4,
    }
    base_man = d["man"].mean()
    L = [f"# 穴狙いの買い方とレース選びの通説を検証する（2026年・{N:,}R・確定オッズ）\n",
         "各レース100円×点数、払戻は公式の確定配当。**帰無（市場が正しい）なら回収率は 75%**。",
         "出目買いは 2018〜2025 の万舟出目で作って 2026 に当てる（アウトオブサンプル）。\n",
         "## 1. 万舟のとき1号艇はどこにいたか（全期間 79,546本）\n",
         "| 1号艇の着順 | 割合 |", "|---|---:|", "| 着外 | **49.3%** |", "| 2着 | 19.6% |", "| 3着 | 19.5% |", "| 1着 | 11.5% |",
         "\n「1号艇2着づけ」は万舟の2割しか拾えない。**万舟の半分は1号艇が消えている。**\n",
         "## 2. 買い方の構造 × レースの選び方（回収率）\n",
         "列＝レースの選び方、行＝買い方。太字は 75% 超。\n"]
    sel_keys = ["全レース", "市場の万舟確率 上位10%（現行）", "市場の万舟確率 上位30%", "1番人気 12倍以上（人気が割れている）", "荒れ条件スコア 4以上"]
    L.append("| 買い方 | " + " | ".join(sel_keys) + " |")
    L.append("|---|" + "---:|" * len(sel_keys))
    structs = [("人気20〜40（21点・現行）", lambda idx: evaluate_rank(d, Qm, idx, 20, 40)),
               ("人気16〜30（15点）", lambda idx: evaluate_rank(d, Qm, idx, 16, 30)),
               ("人気30〜50（21点）", lambda idx: evaluate_rank(d, Qm, idx, 30, 50)),
               ("人気20〜40のうち1号艇1着以外", lambda idx: evaluate_rank(d, Qm, idx, 20, 40, exclude_l1_head=True)),
               ("1号艇1着を除いた人気10〜30（21点）", lambda idx: evaluate_rank(d, Qm, idx, 10, 30, exclude_l1_head=True))]
    structs += [(nm, (lambda m: lambda idx: evaluate_fixed(d, m, idx))(m)) for nm, m in STRUCT_FIXED.items()]
    structs += [(nm, (lambda ln: lambda idx: evaluate_trio(d, payr, idx, ln))(ln)) for nm, ln in TRIO_SETS.items()]
    structs.append(("出目買い（場ごとの万舟頻出20）", lambda idx: (
        np.array([d["pay3"].values[i] if (d["win_idx"].values[i] >= 0 and deme.get(d["stadium_code"].values[i], np.zeros(120, bool))[d["win_idx"].values[i]]) else 0.0 for i in idx]),
        np.full(len(idx), 2000.0))))
    grid = {}
    for nm, fn in structs:
        cells = []
        for sk in sel_keys:
            idx = np.flatnonzero(SEL[sk])
            r = summarize(*fn(idx))
            grid[(nm, sk)] = r
            cells.append(f"{'**' if r['roi'] > .75 else ''}{r['roi']*100:.1f}%{'**' if r['roi'] > .75 else ''}")
        L.append(f"| {nm} | " + " | ".join(cells) + " |")

    # 詳細: 現行選択（上位10%）での各構造
    sk = "市場の万舟確率 上位10%（現行）"
    L += [f"\n### 詳細: {sk}（{int(SEL[sk].sum()):,}R）\n",
          "| 買い方 | 1レース投資 | 的中率 | 平均払戻 | 回収率 | 95%区間 |", "|---|---:|---:|---:|---:|---|"]
    for nm, _ in structs:
        r = grid[(nm, sk)]
        L.append(f"| {nm} | {r['stake']:,.0f}円 | {r['hit']*100:.1f}% | {r['avg']:,.0f}円 | **{r['roi']*100:.1f}%** | {r['lo']*100:.0f}〜{r['hi']*100:.0f}% |")

    # レース選択の通説: 現行の買い方（人気20〜40）で比較、実測÷市場つき
    L += ["\n## 3. レースの選び方の通説（買い方は人気20〜40の21点で固定）\n",
          "| 選び方 | レース数 | 1日あたり | 万舟率 | 市場が示す万舟率 | 実測÷市場 | 的中率 | 回収率 | 95%区間 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    days = d["race_date"].nunique()
    for sk, m in SEL.items():
        idx = np.flatnonzero(m)
        if len(idx) < 200:
            L.append(f"| {sk} | {len(idx):,} | — | — | — | — | — | — | 少なすぎ |")
            continue
        r = summarize(*evaluate_rank(d, Qm, idx, 20, 40))
        ratio = d["man"].values[idx].mean() / max(d["q_man"].values[idx].mean(), 1e-9)
        L.append(f"| {sk} | {len(idx):,} | {len(idx)/days:.1f}R | {d['man'].values[idx].mean()*100:.1f}% | "
                 f"{d['q_man'].values[idx].mean()*100:.1f}% | {'**' if ratio>1 else ''}{ratio:.3f}{'**' if ratio>1 else ''} | "
                 f"{r['hit']*100:.1f}% | **{r['roi']*100:.1f}%** | {r['lo']*100:.0f}〜{r['hi']*100:.0f}% |")
    # 組合せ: 現行 × 通説
    L += ["\n### 現行（市場万舟確率 上位10%）に通説の条件を重ねる\n",
          "| 追加条件 | レース数 | 1日あたり | 実測÷市場 | 回収率 | 95%区間 |", "|---|---:|---:|---:|---:|---|"]
    base = SEL["市場の万舟確率 上位10%（現行）"]
    for sk in ("1号艇がB級", "前づけあり", "荒れる場（平和島・鳴門・戸田・江戸川）", "女子戦・ルーキー戦", "4カド条件（4号艇が3号艇より勝率高くST速い）",
               "1番人気 12倍以上（人気が割れている）", "荒れ条件スコア 4以上", "1号艇の展示タイム4位以下"):
        idx = np.flatnonzero(base & SEL[sk])
        if len(idx) < 100:
            L.append(f"| ＋{sk} | {len(idx):,} | — | — | — | 少なすぎ |")
            continue
        r = summarize(*evaluate_rank(d, Qm, idx, 20, 40))
        ratio = d["man"].values[idx].mean() / max(d["q_man"].values[idx].mean(), 1e-9)
        L.append(f"| ＋{sk} | {len(idx):,} | {len(idx)/days:.1f}R | {ratio:.3f} | **{r['roi']*100:.1f}%** | {r['lo']*100:.0f}〜{r['hi']*100:.0f}% |")
    # ---- 前半（1〜5月）／後半（6〜8月）で再現するか。8通り試して最大を取っているので必須。
    half = (d["race_date"] <= "2026-05-31").values
    L += ["\n### 前半（1〜5月）と後半（6〜8月）で再現するか\n",
          "上の表は8通り試して良いものを見ているので、偶然の最大値が混ざる。半分ずつで同じ向きに出るかを確認する。",
          "**両方で現行（80.5%）を上回って初めて「効いた」とみなす。**\n",
          "| 構成 | 前半 n | 前半 回収率 | 後半 n | 後半 回収率 | 判定 |", "|---|---:|---:|---:|---:|---|"]
    def half_roi(mask):
        out = []
        for hm in (half, ~half):
            idx = np.flatnonzero(mask & hm)
            r = summarize(*evaluate_rank(d, Qm, idx, 20, 40)) if len(idx) >= 50 else None
            out.append((len(idx), r))
        return out
    cands = [("現行（上位10%）のみ", base)]
    for sk in ("1号艇がB級", "荒れる場（平和島・鳴門・戸田・江戸川）", "4カド条件（4号艇が3号艇より勝率高くST速い）",
               "1番人気 12倍以上（人気が割れている）", "荒れ条件スコア 4以上", "1号艇の展示タイム4位以下", "前づけあり"):
        cands.append((f"現行＋{sk}", base & SEL[sk]))
    ref = None
    for nm, m in cands:
        (n1, r1), (n2, r2) = half_roi(m)
        if r1 is None or r2 is None:
            L.append(f"| {nm} | {n1} | — | {n2} | — | 少なすぎ |"); continue
        if ref is None:
            ref = (r1["roi"], r2["roi"])
            L.append(f"| {nm} | {n1:,} | {r1['roi']*100:.1f}% | {n2:,} | {r2['roi']*100:.1f}% | 基準 |"); continue
        ok = r1["roi"] > ref[0] and r2["roi"] > ref[1]
        L.append(f"| {nm} | {n1:,} | {r1['roi']*100:.1f}% | {n2:,} | {r2['roi']*100:.1f}% | {'**両方で上回る**' if ok else ('片方だけ' if (r1['roi'] > ref[0]) != (r2['roi'] > ref[1]) else '両方で下回る')} |")
    # 買い方の構造も同じ確認（上位10%の中で）
    L += ["\n### 買い方の構造も前半・後半で確認（市場万舟確率 上位10%の中）\n",
          "| 買い方 | 前半 回収率 | 後半 回収率 | 判定（現行 人気20〜40 との比較） |", "|---|---:|---:|---|"]
    ref2 = None
    for nm, fn in structs:
        rs = []
        for hm in (half, ~half):
            idx = np.flatnonzero(base & hm)
            rs.append(summarize(*fn(idx))["roi"])
        if ref2 is None:
            ref2 = rs
            L.append(f"| {nm} | {rs[0]*100:.1f}% | {rs[1]*100:.1f}% | 基準 |"); continue
        ok = rs[0] > ref2[0] and rs[1] > ref2[1]
        L.append(f"| {nm} | {rs[0]*100:.1f}% | {rs[1]*100:.1f}% | {'**両方で上回る**' if ok else ('片方だけ' if (rs[0] > ref2[0]) != (rs[1] > ref2[1]) else '両方で下回る')} |")

    L += ["\n## 4. 結論と、ツールに反映すべき内容\n",
          "### 買い方の構造 → 現行の「人気20〜40の21点」を維持",
          "予想サイトが勧める構造（4-全-全、2〜4頭×5・6の2着、1号艇2着づけ、1号艇消し、ボックス、3連複、出目買い）は",
          "**18通りすべてが前半・後半とも人気20〜40を下回った。** 市場の人気順は、固定の型では拾えないレース固有の情報を持っている。",
          "唯一「両方で上回った」ボックス2,4,6（6点）は的中3.2%・的中数約120本で95%区間が63〜108%と広く、",
          "{2,4,6}を選ぶ仕組みの説明もない（偶数艇の迷信）。**採用しない。**",
          "",
          "### レースの選び方 → 現行の「市場が示す万舟確率 上位10%」を維持",
          "通説の条件（1号艇B級・前づけ・荒れる場・女子戦・4カド・初日・展示順位）を**単独で**使うと、",
          "どれも実測÷市場 < 1 で回収率 67〜75%。**女子戦・ルーキー戦は通説と逆で堅い**（万舟率15.2% < 全体16.5%、回収率66.8%）。",
          "唯一 実測÷市場 > 1 なのは市場の万舟確率上位10%（1.006、80.5%）。",
          "",
          "### 現行に重ねると効きそうな条件（前半・後半とも上回った3つ）→ **選定には入れず、タグとして記録して前向きに検証する**",
          "| 追加条件 | 全体 | 前半 | 後半 | 1日あたり |", "|---|---:|---:|---:|---:|",
          "| ＋1号艇がB級 | 82.3% | 78.5% | 88.8% | 10.0R |",
          "| ＋4カド条件（4号艇が3号艇より勝率高くST速い） | 87.6% | 84.3% | 93.5% | 3.9R |",
          "| ＋1号艇の展示タイム4位以下 | 81.8% | 79.3% | 85.9% | 6.2R |",
          "",
          "7つ試して3つ通った。帰無でも1つは「両方で上回る」を通る（25%×7≒1.75）ので、3つのうち1〜2つは偶然でもおかしくない。",
          "**ここで選定に入れると winner's curse を踏む。** 事前登録して、9/13以降の締切前オッズでの記録（実オッズ）で確認する。",
          "合格基準（今決める）: 発火300レース以上たまった時点で、タグ付きの回収率 > タグ無しの回収率、かつ差が5pt以上。",
          "",
          "### ツールに反映する内容（確定）",
          "1. 穴モードの選定・買い目は変えない（人気20〜40 × 市場万舟確率上位10%）。",
          "2. 穴モードの各レースに **荒れ理由タグ** を付けて記録・表示する: 1号艇B級／4カド条件／1号艇展示T4位以下／",
          "   荒れる場／前づけあり／1番人気12倍以上。**表示は理解のため、記録は前向き検証のため。** 選定には使わない。",
          "3. 「堅い」の注意タグも出す: 女子戦・ルーキー戦・企画レース（万舟率10〜15%）は穴向きでない。",
          "4. `lab modes` にタグ別の集計を足し、300レース到達で判定できるようにする。",
          "",
          "### 出典（通説の収集元）",
          "競艇AIバズーカー（kyotei-ai.com/article/34）、kcbn.jp/mansyu-method、kyoutei-navi.com/beginner/ana、okaturi.com/mansyu-torikata、",
          "paris-montagne.org/beginner/manshuken、funaban.com/wp/post-409、fukuoka-kyotei.com/beginner/sanrenpuku-box"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
