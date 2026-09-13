"""軸艇検証の続き: ana_axis.md で唯一残った「本命艇（市場1着確率最大）を1着に固定し、人気20〜40の帯と交差」を掘る。

事前の根拠（後づけでない）: crowd_bias.md は「市場は1号艇（本命）を過小評価し、2〜6号艇を過大評価」、
longshot.md は「人気1〜5番目の比が 1.05〜1.09」。本命が1着で相手が荒れる目は、市場が安く売っている可能性がある。

手順（winner's curse 対策）
  探索 = 1〜5月で 帯の幅 × 軸の決め方 × レース選択 を総当たりし、最良（回収率の95%区間下限が最大）を1つ選ぶ。
  確認 = 6〜8月でその1つだけを評価する。
  帰無 = 勝者を市場確率から引き直したデータで同じ探索→確認を20回行い、確認回収率の分布を出す。

出力: reports/research/ana_axis2.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ana_axis import axes_for, max_streak  # noqa: E402
from run_ana_playbook import A, B, C, evaluate_rank, load, summarize  # noqa: E402

from boatlab.config import ROOT  # noqa: E402

OUT = Path(ROOT) / "reports" / "research" / "ana_axis2.md"

HEADS = {"本命（1着確率最大）": "A4", "1号艇（固定）": "L1", "1号艇以外の最有力": "A1"}
BANDS = [(20, 40), (15, 40), (20, 50), (20, 60), (10, 40), (30, 60), (15, 60), (10, 60)]


def pick(q, head_lane, lo, hi, within=False):
    order = np.argsort(-q)
    if within:
        order = order[A[order] == head_lane]
        return order[lo - 1: hi]
    band = order[lo - 1: hi]
    return band[A[band] == head_lane]


def head_of(ax, key):
    return 0 if key == "L1" else ax[key]


def evaluate(wins, pays, Qm, AX, rows_idx, key, lo, hi, within=False):
    ret, stake = [], []
    for i in rows_idx:
        pts = pick(Qm[i], head_of(AX[i], key), lo, hi, within)
        w = wins[i]
        ret.append(pays[i] if (w >= 0 and len(pts) and w in set(pts.tolist())) else 0.0)
        stake.append(100.0 * len(pts))
    return np.array(ret), np.array(stake)


def search(wins, pays, Qm, AX, sel_masks, explore_mask):
    """探索期間で全候補を評価し、(下限が最大の候補, 全結果) を返す。"""
    res = []
    for sk, m in sel_masks.items():
        idx = np.flatnonzero(m & explore_mask)
        for hn, hk in HEADS.items():
            for lo, hi in BANDS:
                ret, stake = evaluate(wins, pays, Qm, AX, idx, hk, lo, hi)
                if stake.sum() == 0:
                    continue
                r = summarize(ret, stake)
                res.append((sk, hn, hk, lo, hi, False, r))
            for lo, hi in ((6, 20), (6, 15), (8, 20), (10, 25), (11, 30)):
                ret, stake = evaluate(wins, pays, Qm, AX, idx, hk, lo, hi, within=True)
                r = summarize(ret, stake)
                res.append((sk, hn, hk, lo, hi, True, r))
    best = max(res, key=lambda x: x[6]["lo"])
    return best, res


def main():
    d, Qm, payr, hist = load()
    N = len(d)
    AX = [axes_for(Qm[i]) for i in range(N)]
    wins = d["win_idx"].values; pays = d["pay3"].values
    days = d["race_date"].nunique()
    half = (d["race_date"] <= "2026-05-31").values
    SEL = {
        "全レース": np.ones(N, bool),
        "市場万舟確率 上位30%": d["q_man"].values >= np.quantile(d["q_man"], 0.70),
        "市場万舟確率 上位10%（現行）": d["q_man"].values >= np.quantile(d["q_man"], 0.90),
    }
    L = [f"# 軸艇検証（続き）: 本命1着 × 穴の相手（2026年・{N:,}R・確定オッズ）\n",
         "`ana_axis.md` で60通り中、前半・後半とも現行を上回ったのは「本命（1着確率最大）を1着に固定 ∩ 人気20〜40」など2通りだけだった。",
         "事前の根拠: `crowd_bias.md`（市場は本命側を過小評価）、`longshot.md`（人気1〜5番目の比 1.05〜1.09）。",
         "**探索（1〜5月）で1つ選び、確認（6〜8月）で1回だけ評価する。**\n"]

    # ---- 1. 探索
    best, res = search(wins, pays, Qm, AX, SEL, half)
    L += ["## 1. 探索（1〜5月）: 軸 × 帯 × レース選択（" + f"{len(res)}通り）\n",
          "帯は「全120通りの人気順で lo〜hi 番目のうち軸が1着のもの」。「軸内」は「軸1着の20通りの中での人気順 lo〜hi」。",
          "選ぶ基準は**回収率の95%区間下限が最大**（点数の少ない偶然の高値を避ける）。上位15件:\n",
          "| レース選択 | 軸 | 帯 | 平均点数 | 的中率 | 回収率 | 95%区間 |", "|---|---|---|---:|---:|---:|---|"]
    for sk, hn, hk, lo, hi, within, r in sorted(res, key=lambda x: -x[6]["lo"])[:15]:
        L.append(f"| {sk} | {hn} | {'軸内' if within else ''}{lo}〜{hi} | {r['stake']/100:.1f} | {r['hit']*100:.1f}% | **{r['roi']*100:.1f}%** | {r['lo']*100:.0f}〜{r['hi']*100:.0f}% |")
    sk, hn, hk, lo, hi, within, r0 = best
    L.append(f"\n→ 選んだ候補: **{sk} × {hn} × {'軸内' if within else ''}{lo}〜{hi}**（探索 {r0['roi']*100:.1f}%、下限 {r0['lo']*100:.0f}%）\n")

    # ---- 2. 確認
    idx_c = np.flatnonzero(SEL[sk] & ~half)
    ret, stake = evaluate(wins, pays, Qm, AX, idx_c, hk, lo, hi, within)
    rc = summarize(ret, stake)
    base_ret, base_stake = evaluate_rank(d, Qm, idx_c, 20, 40)
    rb = summarize(base_ret, base_stake)
    days_c = d["race_date"][~half].nunique()
    L += ["## 2. 確認（6〜8月）: 選んだ1つだけを評価\n",
          "| 買い方 | レース数 | 1日あたり | 平均点数 | 1レース投資 | 的中率 | 平均払戻 | 最長連敗 | 回収率 | 95%区間 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
          f"| 現行 人気20〜40（21点） | {len(idx_c):,} | {len(idx_c)/days_c:.1f}R | 21.0 | {rb['stake']:,.0f}円 | {rb['hit']*100:.1f}% | {rb['avg']:,.0f}円 | {max_streak(base_ret)}R | **{rb['roi']*100:.1f}%** | {rb['lo']*100:.0f}〜{rb['hi']*100:.0f}% |",
          f"| 候補 {hn} × {'軸内' if within else ''}{lo}〜{hi} | {len(idx_c):,} | {len(idx_c)/days_c:.1f}R | {rc['stake']/100:.1f} | {rc['stake']:,.0f}円 | {rc['hit']*100:.1f}% | {rc['avg']:,.0f}円 | {max_streak(ret)}R | **{rc['roi']*100:.1f}%** | {rc['lo']*100:.0f}〜{rc['hi']*100:.0f}% |"]
    # 帯の残り（候補以外の部分）も出す: 現行の帯のうち軸1着でない目
    if not within:
        rest_ret, rest_stake = [], []
        for i in idx_c:
            q = Qm[i]; order = np.argsort(-q); band = order[lo - 1: hi]
            pts = band[A[band] != head_of(AX[i], hk)]
            w = wins[i]
            rest_ret.append(pays[i] if (w >= 0 and w in set(pts.tolist())) else 0.0); rest_stake.append(100.0 * len(pts))
        rr = summarize(np.array(rest_ret), np.array(rest_stake))
        L.append(f"| 同じ帯のうち軸1着**以外**の目 | {len(idx_c):,} | — | {rr['stake']/100:.1f} | {rr['stake']:,.0f}円 | {rr['hit']*100:.1f}% | {rr['avg']:,.0f}円 | — | {rr['roi']*100:.1f}% | {rr['lo']*100:.0f}〜{rr['hi']*100:.0f}% |")
    # 月別
    L += ["\n確認期間の月別:\n", "| 月 | レース数 | 的中 | 回収率 | 損益 |", "|---|---:|---:|---:|---:|"]
    mon = d["race_date"].values[idx_c].astype(str)
    for m in sorted(set(x[:7] for x in mon)):
        mm = np.array([x[:7] == m for x in mon])
        L.append(f"| {m} | {mm.sum():,} | {(ret[mm]>0).sum()} | {ret[mm].sum()/max(stake[mm].sum(),1)*100:.1f}% | {ret[mm].sum()-stake[mm].sum():+,.0f}円 |")

    # ---- 3. 帰無: 同じ探索→確認を、市場確率どおりに引き直した勝者で20回
    rng = np.random.default_rng(11)
    conf = []
    for rep in range(20):
        w_null = np.array([rng.choice(120, p=Qm[i]) for i in range(N)])
        p_null = 75.0 / Qm[np.arange(N), w_null]
        b, _ = search(w_null, p_null, Qm, AX, SEL, half)
        sk2, hn2, hk2, lo2, hi2, within2, _ = b
        ii = np.flatnonzero(SEL[sk2] & ~half)
        r2, s2 = evaluate(w_null, p_null, Qm, AX, ii, hk2, lo2, hi2, within2)
        conf.append(r2.sum() / s2.sum())
    conf = np.array(conf)
    L += ["\n## 3. 帰無シミュレーション（市場が正しい世界で同じ探索→確認を20回）\n",
          f"確認回収率: 平均 {conf.mean()*100:.1f}%、範囲 {conf.min()*100:.1f}〜{conf.max()*100:.1f}%、"
          f"実データの {rc['roi']*100:.1f}% 以上が出た回数 **{(conf >= rc['roi']).sum()}/20**。",
          "（帰無では元返しを無視して払戻＝75÷市場確率にしているので、帰無の回収率はちょうど75%前後になる）\n"]

    # ---- 4. 全期間での帯別（1号艇固定・上位10%）: 帯の形がなだらかか
    idx_all = np.flatnonzero(SEL["市場万舟確率 上位10%（現行）"])
    L += ["## 4. 参考: 上位10%レース・1号艇1着固定で帯を動かす（全期間、探索に使った数字を含むので割り引いて読む）\n",
          "| 帯（全体の人気順） | 平均点数 | 的中率 | 平均払戻 | 回収率 | 95%区間 | 前半 | 後半 |", "|---|---:|---:|---:|---:|---|---:|---:|"]
    for lo2, hi2 in ((1, 10), (5, 20), (10, 30), (20, 40), (20, 60), (30, 60), (40, 80), (60, 120)):
        ret_a, st_a = evaluate(wins, pays, Qm, AX, idx_all, "L1", lo2, hi2)
        ra = summarize(ret_a, st_a)
        hs = []
        for hm in (half, ~half):
            ii = np.flatnonzero(SEL["市場万舟確率 上位10%（現行）"] & hm)
            r_, s_ = evaluate(wins, pays, Qm, AX, ii, "L1", lo2, hi2)
            hs.append(r_.sum() / max(s_.sum(), 1))
        L.append(f"| {lo2}〜{hi2} | {ra['stake']/100:.1f} | {ra['hit']*100:.1f}% | {ra['avg']:,.0f}円 | **{ra['roi']*100:.1f}%** | {ra['lo']*100:.0f}〜{ra['hi']*100:.0f}% | {hs[0]*100:.1f}% | {hs[1]*100:.1f}% |")
    L += ["",
          '## 5. 結論（2026-09-13）',
          '',
          '- 事前登録した手順（探索1〜5月で95%区間下限が最大の1つを選ぶ → 確認6〜8月で1回評価）で選ばれたのは',
          '  「上位10%レース × 1号艇1着固定 × 全体の人気10〜40のうち1号艇1着の目（平均9.1点）」。**確認 83.9%（73〜97%）で、',
          '  現行の人気20〜40（84.8%）を超えなかった。** 同じ帯の残り（1号艇1着以外）は 77.8% で、本命側の方が良い向きは確認期間でも出ている。',
          '- 帰無（市場が正しい世界）で同じ探索→確認を20回やると確認回収率は平均76.7%、83.9%以上は1/20。',
          '  「本命1着の目は市場より少し良い」は帰無より上に見えるが、**現行の帯全体（84.8%）との差は無い**。',
          '- §4 の 1号艇固定 20〜40（89.3%）・30〜60（94.2%）は全期間の数字で、前半・後半が 84.5/97.7、77.7/122.8 と割れ、',
          '  点数4〜5・的中4〜6%なので後半の高値は少数の万舟が作っている。**採用しない。**',
          '- **軸艇方式はツールに反映しない。** 記録として `ana_axis.md` / `ana_axis2.md` を残す。',
          '  次に穴の買い方を動かすなら、帯の中の目の選び方でなく、締切前オッズでの現行の実測（`lab modes` の累計）を先に見る。']
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
