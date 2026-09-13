"""穴狙いの「軸艇」構造の検証。

ユーザー提案（2026-09-12）: 「荒れると市場が予測している以上、何番かの舟がその中心にいるはず。
それを軸に10〜15点ほど（それ以上でも可）で買う」。

軸の決め方はすべて確定3連単オッズだけから計算する（結果・展示・出走表は使わない）。
  A1 1号艇以外で市場の1着確率が最大の艇
  A2 100倍以上の買い目の確率の総和に最も多く現れる艇（どの着でも）＝荒れの中心
  A3 100倍以上の買い目で1着に最も多く置かれる艇
  A4 市場の1着確率が最大の艇（1号艇を含む。対照）
  A5 市場の1着確率が2番目の艇

構造（軸=a、相手=全、点数は固定または可変）
  軸1着・全流し(20) / 軸1着・人気上位10 / 軸1着・人気6〜20 / 軸1着・1号艇を相手から外す(12) /
  軸1着・1号艇を2着か3着に置く(8) / 軸1着・100倍以上のみ / 軸1-2着流し・人気上位15 /
  軸1-2着流し・人気11〜30 / 軸1-2着流し・100倍以上のみ / 2軸(A1,A5)-全-全 上位15 / 軸1着×人気20〜40の交差

物差しは ana_playbook.py と同じ: 各レース100円×点数、払戻は公式、帰無=75%、
現行「人気20〜40の21点」との比較、前半（1〜5月）／後半（6〜8月）での再現。
レース選択: 全レース / 市場万舟確率 上位30% / 上位10%（現行）。

出力: reports/research/ana_axis.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ana_playbook import A, B, C, evaluate_rank, load, summarize  # noqa: E402

from boatlab.config import ROOT  # noqa: E402

OUT = Path(ROOT) / "reports" / "research" / "ana_axis.md"
MAN = 0.0075  # 100倍 ⇔ 市場確率 0.0075


# ---------------------------------------------------------------- 軸の決め方（q: 120通りの市場確率）
def axes_for(q):
    q1 = np.bincount(A, weights=q, minlength=6)              # 1着周辺確率
    man = q <= MAN
    any_mass = np.zeros(6)
    for pos in (A, B, C):
        any_mass += np.bincount(pos, weights=q * man, minlength=6)
    head_mass = np.bincount(A, weights=q * man, minlength=6)
    order1 = np.argsort(-q1)
    return {
        "A1": int(np.argmax(np.where(np.arange(6) == 0, -1, q1))),
        "A2": int(np.argmax(any_mass)),
        "A3": int(np.argmax(head_mass)),
        "A4": int(order1[0]),
        "A5": int(order1[1]),
    }


AXIS_NAMES = {
    "A1": "1号艇以外で1着確率が最大の艇",
    "A2": "100倍以上の買い目に最も多く現れる艇（荒れの中心）",
    "A3": "100倍以上の買い目で1着に最も多い艇",
    "A4": "1着確率が最大の艇（1号艇を含む・対照）",
    "A5": "1着確率が2番目の艇",
}


# ---------------------------------------------------------------- 構造: (q, a, a2) -> 買う120通りのインデックス
def _rank_within(q, mask, lo, hi):
    idx = np.flatnonzero(mask)
    order = idx[np.argsort(-q[idx])]
    return order[lo - 1: hi]


def s_head_all(q, a, a2):          # 軸-全-全 20点
    return np.flatnonzero(A == a)


def s_head_top10(q, a, a2):
    return _rank_within(q, A == a, 1, 10)


def s_head_6_20(q, a, a2):
    return _rank_within(q, A == a, 6, 20)


def s_head_no1(q, a, a2):          # 軸-{1号艇以外}-{1号艇以外} 12点（軸が1号艇なら20点）
    return np.flatnonzero((A == a) & (B != 0) & (C != 0)) if a != 0 else np.flatnonzero(A == a)


def s_head_with1(q, a, a2):        # 軸-1-全 + 軸-全-1 8点
    return np.flatnonzero((A == a) & ((B == 0) | (C == 0))) if a != 0 else np.flatnonzero(A == a)


def s_head_man(q, a, a2):          # 軸1着で100倍以上のみ
    return np.flatnonzero((A == a) & (q <= MAN))


def s_12_top15(q, a, a2):          # 軸を1着か2着に置く40点のうち人気上位15
    return _rank_within(q, (A == a) | (B == a), 1, 15)


def s_12_11_30(q, a, a2):
    return _rank_within(q, (A == a) | (B == a), 11, 30)


def s_12_man(q, a, a2):
    return np.flatnonzero(((A == a) | (B == a)) & (q <= MAN))


def s_2axis_top15(q, a, a2):       # {a,a2}-全-全 40点のうち上位15
    return _rank_within(q, (A == a) | (A == a2), 1, 15)


def s_2axis_box_all(q, a, a2):     # {a,a2}-{a,a2}-全 8点
    return np.flatnonzero(((A == a) & (B == a2)) | ((A == a2) & (B == a)))


def s_head_band2040(q, a, a2):     # 軸1着 ∩ 全体の人気20〜40
    order = np.argsort(-q)
    band = np.zeros(120, bool); band[order[19:40]] = True
    return np.flatnonzero((A == a) & band)


STRUCTS = [
    ("軸1着・全流し（20点）", s_head_all),
    ("軸1着・人気上位10（10点）", s_head_top10),
    ("軸1着・人気6〜20（15点）", s_head_6_20),
    ("軸1着・1号艇を相手から外す（12点）", s_head_no1),
    ("軸1着・1号艇を2着か3着に置く（8点）", s_head_with1),
    ("軸1着・100倍以上のみ（可変）", s_head_man),
    ("軸1-2着流し・人気上位15（15点）", s_12_top15),
    ("軸1-2着流し・人気11〜30（20点）", s_12_11_30),
    ("軸1-2着流し・100倍以上のみ（可変）", s_12_man),
    ("2軸（軸＋1着2番目）-全-全 上位15（15点）", s_2axis_top15),
    ("2軸ボックス-全（8点）", s_2axis_box_all),
    ("軸1着 ∩ 人気20〜40（可変）", s_head_band2040),
]


def evaluate_struct(d, Qm, AX, rows_idx, axis_key, fn):
    ret, stake = [], []
    wins = d["win_idx"].values; pays = d["pay3"].values
    for i in rows_idx:
        q = Qm[i]; ax = AX[i]
        a = ax[axis_key]; a2 = ax["A5"] if ax["A5"] != a else ax["A4"]
        pts = fn(q, a, a2)
        w = wins[i]
        ret.append(pays[i] if (w >= 0 and w in set(pts.tolist())) else 0.0)
        stake.append(100.0 * len(pts))
    return np.array(ret), np.array(stake)


def max_streak(ret):
    best = cur = 0
    for r in ret:
        cur = cur + 1 if r <= 0 else 0
        best = max(best, cur)
    return best


def fmt(r, ref=None):
    s = f"{r['roi']*100:.1f}%"
    if r["roi"] > .75:
        s = f"**{s}**"
    return s


def main():
    d, Qm, payr, hist = load()
    N = len(d)
    AX = [axes_for(Qm[i]) for i in range(N)]
    days = d["race_date"].nunique()
    half = (d["race_date"] <= "2026-05-31").values
    SEL = {
        "全レース": np.ones(N, bool),
        "市場万舟確率 上位30%": d["q_man"].values >= np.quantile(d["q_man"], 0.70),
        "市場万舟確率 上位10%（現行）": d["q_man"].values >= np.quantile(d["q_man"], 0.90),
    }
    L = [f"# 穴狙いの「軸艇」構造を検証する（2026年・{N:,}R・確定オッズ）\n",
         "提案: 「荒れると市場が予測している以上、何番かの舟がその中心にいるはず。それを軸に10〜15点ほどで買う」。",
         "軸は**確定3連単オッズだけ**から決める。各レース100円×点数、払戻は公式。**帰無（市場が正しい）なら回収率 75%**。",
         "比較対象は現行の「人気20〜40の21点」。\n"]

    # ---- 1. 軸はどの艇になるか・軸の1着率
    L += ["## 1. 軸はどの艇になるか（市場万舟確率 上位10% のレース）\n",
          "| 軸の決め方 | 1号艇 | 2 | 3 | 4 | 5 | 6 | 軸の1着率 | 軸の3着以内率 | 軸1着かつ万舟 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    top = np.flatnonzero(SEL["市場万舟確率 上位10%（現行）"])
    win_lane = np.array([A[w] if w >= 0 else -1 for w in d["win_idx"].values])
    in3 = np.zeros((N, 6), bool)
    for i, w in enumerate(d["win_idx"].values):
        if w >= 0:
            in3[i, [A[w], B[w], C[w]]] = True
    for k, nm in AXIS_NAMES.items():
        ax = np.array([AX[i][k] for i in top])
        dist = np.bincount(ax, minlength=6) / len(top)
        w1 = (win_lane[top] == ax).mean()
        w3 = in3[top, ax].mean()
        wm = ((win_lane[top] == ax) & (d["man"].values[top] == 1)).mean()
        L.append(f"| {k} {nm} | " + " | ".join(f"{x*100:.0f}%" for x in dist) + f" | {w1*100:.1f}% | {w3*100:.1f}% | {wm*100:.1f}% |")
    L.append(f"\n参考: このレース群の万舟率 {d['man'].values[top].mean()*100:.1f}%、1号艇の1着率 {(win_lane[top]==0).mean()*100:.1f}%。\n")

    # ---- 2. 総当たり（軸 × 構造 × 選択）
    grid = {}
    for sk, m in SEL.items():
        idx = np.flatnonzero(m)
        base = summarize(*evaluate_rank(d, Qm, idx, 20, 40))
        grid[("base", "-", sk)] = base
        for ak in AXIS_NAMES:
            for nm, fn in STRUCTS:
                grid[(ak, nm, sk)] = summarize(*evaluate_struct(d, Qm, AX, idx, ak, fn))
    for sk in SEL:
        L += [f"\n## 2. 回収率の総当たり — {sk}（{int(SEL[sk].sum()):,}R）\n",
              f"現行 人気20〜40（21点）: **{grid[('base','-',sk)]['roi']*100:.1f}%**。太字は75%超。列＝軸の決め方。\n",
              "| 構造 | " + " | ".join(AXIS_NAMES) + " |", "|---|" + "---:|" * len(AXIS_NAMES)]
        for nm, _ in STRUCTS:
            L.append(f"| {nm} | " + " | ".join(fmt(grid[(ak, nm, sk)]) for ak in AXIS_NAMES) + " |")

    # ---- 3. 上位10% の詳細（現行との比較）
    sk = "市場万舟確率 上位10%（現行）"
    idx = np.flatnonzero(SEL[sk])
    base_ret, base_stake = evaluate_rank(d, Qm, idx, 20, 40)
    L += [f"\n## 3. 詳細 — {sk}（{len(idx):,}R、{len(idx)/days:.1f}R/日）\n",
          "| 軸 | 構造 | 平均点数 | 1レース投資 | 的中率 | 平均払戻 | 最長連敗 | 回収率 | 95%区間 |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    r = summarize(base_ret, base_stake)
    L.append(f"| — | 現行 人気20〜40 | 21.0 | {r['stake']:,.0f}円 | {r['hit']*100:.1f}% | {r['avg']:,.0f}円 | {max_streak(base_ret)}R | **{r['roi']*100:.1f}%** | {r['lo']*100:.0f}〜{r['hi']*100:.0f}% |")
    rows = []
    for ak in AXIS_NAMES:
        for nm, fn in STRUCTS:
            ret, stake = evaluate_struct(d, Qm, AX, idx, ak, fn)
            r = summarize(ret, stake)
            rows.append((r["roi"], ak, nm, r, max_streak(ret)))
    rows.sort(key=lambda x: -x[0])
    for roi, ak, nm, r, st in rows:
        L.append(f"| {ak} | {nm} | {r['stake']/100:.1f} | {r['stake']:,.0f}円 | {r['hit']*100:.1f}% | {r['avg']:,.0f}円 | {st}R | {fmt(r)} | {r['lo']*100:.0f}〜{r['hi']*100:.0f}% |")

    # ---- 4. 前半・後半で再現するか（上位10%内、現行と比較）
    L += ["\n## 4. 前半（1〜5月）／後半（6〜8月）で再現するか（市場万舟確率 上位10% の中）\n",
          "60通り（5軸×12構造）を試して最大を見ているので偶然の最大値が混ざる。帰無でも約25%が「両方で上回る」を通る（期待15通り）。",
          "**両方で現行を上回り、かつ両方で75%を超える**ものだけを候補にする。\n",
          "| 軸 | 構造 | 前半 回収率 | 後半 回収率 | 判定 |", "|---|---|---:|---:|---|"]
    refs = []
    for hm in (half, ~half):
        ii = np.flatnonzero(SEL[sk] & hm)
        refs.append(summarize(*evaluate_rank(d, Qm, ii, 20, 40))["roi"])
    L.append(f"| — | 現行 人気20〜40 | {refs[0]*100:.1f}% | {refs[1]*100:.1f}% | 基準 |")
    passed = []
    for ak in AXIS_NAMES:
        for nm, fn in STRUCTS:
            rs = []
            for hm in (half, ~half):
                ii = np.flatnonzero(SEL[sk] & hm)
                rs.append(summarize(*evaluate_struct(d, Qm, AX, ii, ak, fn))["roi"])
            ok = rs[0] > refs[0] and rs[1] > refs[1]
            both75 = rs[0] > .75 and rs[1] > .75
            tag = "**両方で上回る**" if ok else ("片方だけ" if (rs[0] > refs[0]) != (rs[1] > refs[1]) else "両方で下回る")
            if ok and both75:
                passed.append((ak, nm, rs))
            L.append(f"| {ak} | {nm} | {rs[0]*100:.1f}% | {rs[1]*100:.1f}% | {tag} |")
    L.append(f"\n両方で現行を上回り、かつ両方で75%超: **{len(passed)}通り**（帰無での期待値 ≈ 15通り）。")
    for ak, nm, rs in passed:
        L.append(f"- {ak} × {nm}: 前半 {rs[0]*100:.1f}% / 後半 {rs[1]*100:.1f}%")

    # ---- 5. 帰無シミュレーション: 市場確率どおりに勝者を引き直したとき、同じ60通りで何通りが「両方で上回る」か
    rng = np.random.default_rng(7)
    L += ["\n## 5. 帰無シミュレーション（市場が正しいとき、同じ60通り×同じ判定で何通り通るか）\n"]
    ii_all = np.flatnonzero(SEL[sk])
    cnt = []
    for rep in range(20):
        # 勝者を q から引き直し、払戻は 75/q（元返し無視）
        wins = np.array([rng.choice(120, p=Qm[i]) for i in ii_all])
        pays = np.array([75.0 / Qm[i][w] for i, w in zip(ii_all, wins)])
        dd = pd.DataFrame({"win_idx": wins, "pay3": pays})
        Qs = Qm[ii_all]; AXs = [AX[i] for i in ii_all]; hs = half[ii_all]
        loc = np.arange(len(ii_all))
        rr = []
        for hm in (hs, ~hs):
            jj = loc[hm]
            rr.append(summarize(*evaluate_rank(dd, Qs, jj, 20, 40))["roi"])
        n_ok = 0
        for ak in AXIS_NAMES:
            for nm, fn in STRUCTS:
                rs = [summarize(*evaluate_struct(dd, Qs, AXs, loc[hm], ak, fn))["roi"] for hm in (hs, ~hs)]
                if rs[0] > rr[0] and rs[1] > rr[1] and rs[0] > .75 and rs[1] > .75:
                    n_ok += 1
        cnt.append(n_ok)
    cnt = np.array(cnt)
    L.append(f"20回の帰無シミュレーションで「両方で現行を上回りかつ両方75%超」を通った数: 平均 {cnt.mean():.1f}通り、"
             f"範囲 {cnt.min()}〜{cnt.max()}（実データ {len(passed)}通り）。")
    L.append(f"実データの {len(passed)}通り以上が出た回数: {(cnt >= len(passed)).sum()}/20。\n")

    L += ["",
          '## 6. 結論（2026-09-13）',
          '',
          '**「荒れの中心にいる舟」は市場のオッズからは1艇に決まらない。** 上位10%レースで、100倍以上の買い目に最も多く現れる艇（A2）は',
          '4号艇31%・5号艇24%・3号艇16%と散らばり、その艇の1着率は9.4%、3着以内率でも39.5%。市場が「荒れる」と見るとき、',
          '荒れ方は特定の艇に集中せず2〜6号艇に分散している（万舟の決まり手が 差し27%／まくり33%／まくり差し37% に割れるのと同じ）。',
          '',
          '**軸を置く構造（5通りの軸 × 12構造 = 60通り）は、現行の人気20〜40（80.5%、前半78.0／後半84.8）に前半・後半とも勝てない。**',
          '荒れの中心艇（A2・A3）を軸にした構造は 68〜78%。1号艇以外の最有力（A1）を軸にすると 68〜82%。',
          '両半で現行を上回ったのは「本命1着 ∩ 人気20〜40」の2通りだけで、帰無シミュレーション（市場が正しい世界）でも平均9.9通りが通るので、',
          '数としては帰無より少ない。軸を1艇に固定すること自体が、市場の人気順が持つレース固有の情報を捨てている。',
          '',
          '**唯一の手がかりは「本命（多くは1号艇）が1着で、2・3着が荒れる目」**: 帯の中で軸1着の目だけを取ると回収率が上がる（A4 94.3%、区間83〜109%）。',
          '`crowd_bias.md`（市場は本命側を過小評価）と向きが合う。ただし点数5前後・的中6%で区間が広く、`ana_axis2.md` で',
          '探索→確認の手順に載せると **確認期間で 83.9% と現行 84.8% を超えなかった**。ツールには入れない。']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
