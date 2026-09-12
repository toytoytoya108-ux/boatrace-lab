"""条件ベースの2モード選定ルール（固定の帯・固定の本数を使わない）。

## 「期待値の高いものを選ぶ」をどう実装するか
モデルの期待値は使えない。`market_combine.md` の実測では、モデル単独が期待値1.0超えと見た
買い目は26.8%あり、その実回収率は **69.8%**。モデルの期待値は高く出るほど当たらない。

代わりに **過去の実測比から較正した期待値** を使う。市場のオッズだけから決まる箱に分け、
箱ごとに「実勝率 ÷ 市場確率」を探索期間で測る。この比は定義上1.00が基準で、
モデルが市場に勝つ主張を一切含まない。

    較正期待値(候補) = 実測比(その候補が属する箱) × 0.75

箱は2軸:
  - 候補自身の市場確率（＝オッズの水準。`longshot.md` の人気順曲線: 比 1.085〜0.610）
  - そのレースの荒れ具合＝市場が示す万舟確率（`manshu.md` の較正表: 比 0.736〜1.006）

探索 〜2026-05-31 で比の表を作り、確認 2026-06〜08 で1回だけ評価する。
プラセボは帰無「市場確率が真」（y は市場確率から生成）。

## 重要な但し書き
どの箱も比 1.333 には届かない。**閾値は「儲かる」ではなく「いちばんマシ」を選ぶためのもの。**
ツールの画面には必ず実測回収率と月の期待損失を併記する。

出力: reports/backtest/condition_rules.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from boatlab.backtest.metrics import roi_bootstrap
from boatlab.config import ROOT

from run_sweet_spot import load, masks, payout_of  # noqa: E402

OUT = Path(ROOT) / "reports" / "backtest" / "condition_rules.md"
EXPLORE_END = "2026-05-31"
RATE = 0.75
Q_MAN = 0.0075
RNG = np.random.default_rng(20260912)
# 候補の市場確率の箱（オッズに直すと 0.75/q 倍）
QB = np.array([0, .0015, .003, .005, .0075, .012, .02, .035, .06, .10, 1.0])


def main():
    q3, raws, date = load()
    n = len(q3)
    e = date <= EXPLORE_END
    c = ~e
    q_man = np.where(q3 <= Q_MAN, q3, 0.0).sum(1)
    mb = np.digitize(q_man, np.quantile(q_man[e], [.2, .4, .6, .8]))      # レースの荒れ 5段階
    ob = np.digitize(q3, QB) - 1                                          # 候補のオッズ 10段階
    # 当たりの位置
    win = np.zeros((n, 120), bool)
    trif = []
    for i in range(n):
        pays = (raws[i] or {}).get("trifecta") or []
        trif.append(str(pays[0].get("combination", "")).strip() if pays else "")
    from boatlab.model.trifecta import PERM_LABELS
    lab2 = {l: j for j, l in enumerate(PERM_LABELS)}
    for i, t in enumerate(trif):
        j = lab2.get(t)
        if j is not None:
            win[i, j] = True
    ok = win.any(1)

    # ---- 探索期間で比の表を作る
    MB = np.repeat(mb[:, None], 120, 1)
    tab = np.full((5, len(QB) - 1), np.nan)
    cnt = np.zeros((5, len(QB) - 1))
    for m in range(5):
        for o in range(len(QB) - 1):
            s = e[:, None] & ok[:, None] & (MB == m) & (ob == o)
            if s.sum() < 2000:
                continue
            tab[m, o] = win[s].mean() / max(q3[s].mean(), 1e-12)
            cnt[m, o] = s.sum()

    L = [f"# 条件ベースの選定ルール（{n:,}R・2026年）\n",
         "**モデルの期待値は使わない。** `market_combine.md`: モデルが期待値1.0超えと見た買い目の",
         "実回収率は 69.8%。代わりに**市場のオッズだけで決まる箱ごとの実測比**を較正表として使う。\n",
         f"探索 〜{EXPLORE_END}（{int(e.sum()):,}R）で表を作り、確認（{int(c.sum()):,}R）で1回だけ評価。\n",
         "## 1. 較正表: 実勝率 ÷ 市場確率（探索期間）\n",
         "縦＝レースの荒れ具合（市場が示す万舟確率の5分割）、横＝候補の市場確率。",
         "**1.333 を超えれば回収率100%。1.00 が「市場は正しい」。**\n"]
    hdr = "| レースの荒れ | " + " | ".join(
        f"{QB[o]*100:.2f}〜{QB[o+1]*100:.2f}%" if QB[o+1] < 1 else f"{QB[o]*100:.0f}%〜" for o in range(len(QB)-1)) + " |"
    L += [hdr, "|---" * (len(QB)) + "|"]
    names = ["堅い(下位20%)", "やや堅い", "ふつう", "やや荒れ", "荒れ(上位20%)"]
    for m in range(5):
        L.append(f"| {names[m]} | " + " | ".join(
            "—" if not np.isfinite(tab[m, o]) else f"{tab[m,o]:.3f}" for o in range(len(QB)-1)) + " |")
    L.append(f"\n表の最大値 **{np.nanmax(tab):.3f}**（必要なのは 1.333）。有効な箱 {int((cnt>0).sum())}個。\n")

    # ---- 確認期間で「条件が揃ったときだけ買う」
    ev = np.where(np.isfinite(tab[mb][np.arange(n)[:, None], ob]),
                  tab[mb][np.arange(n)[:, None], ob], 0.0) * RATE
    L += ["## 2. 穴モード: 較正期待値の高い候補だけを上位k点\n",
          "候補ごとに `較正期待値 = 実測比 × 0.75`。市場確率2%未満（オッズ37倍以上）の候補のうち",
          "**較正期待値が高い順に上位k点**、かつ最良候補が閾値以上のレースだけ発火。\n",
          "**帰無（市場確率が真）なら回収率はちょうど 75% になる。75%を超えられるかがすべて。**\n",
          "| k点 | しきい値 | 発火 | 1日あたり | 1日の投資 | 的中率 | 平均払戻 | 最長連敗 | 回収率 | 95%区間 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    days_c = pd.Series(date[c]).nunique()
    cand_ana = (q3 < 0.02)
    ev_ana = np.where(cand_ana, ev, -1.0)
    rank = np.argsort(-ev_ana, 1)
    best_ev = np.take_along_axis(ev_ana, rank[:, :1], 1).ravel()
    for k in (5, 10, 20):
        for th in (0.0, 0.72, 0.75):
            pick = np.zeros_like(cand_ana)
            np.put_along_axis(pick, rank[:, :k], True, 1)
            pick &= cand_ana
            fire = c & (pick.sum(1) > 0) & (best_ev >= th)
            idx = np.flatnonzero(fire)
            if len(idx) < 100:
                continue
            ret = np.array([sum(payout_of(raws[i], "3連単", int(j)) for j in np.flatnonzero(pick[i]))
                            for i in idx])
            stake = 100.0 * pick[idx].sum(1)
            lo, hi = roi_bootstrap(stake, ret, n_boot=400)
            st = 0; best = 0
            for h in (ret > 0):
                st = 0 if h else st + 1
                best = max(best, st)
            lab = "条件なし" if th == 0 else f"{th:.2f}以上"
            L.append(f"| {k}点 | {lab} | {len(idx):,} | {len(idx)/days_c:.1f}R | "
                     f"{stake.sum()/days_c:,.0f}円 | {(ret>0).mean()*100:.1f}% | "
                     f"{ret[ret>0].mean():,.0f}円 | {best}R | **{ret.sum()/stake.sum()*100:.1f}%** | "
                     f"{lo*100:.0f}〜{hi*100:.0f}% |")

    # ---- プラセボ
    L += ["\n### プラセボ（帰無「市場確率が真」・10回）\n",
          "同じ手順で、結果だけ市場確率から生成する。比が 1.0 付近なら手順は数字を作っていない。\n",
          "| 設定 | 回収率（平均） | （最大） |", "|---|---:|---:|"]
    cum = q3[c].cumsum(1)
    for th in (0.0, 0.75):
        pick = np.zeros_like(cand_ana)
        np.put_along_axis(pick, rank[:, :10], True, 1)
        pick &= cand_ana
        fire = c & (pick.sum(1) > 0) & (best_ev >= th)
        sub = fire[c]
        vals = []
        for _ in range(10):
            j = (RNG.random((int(c.sum()), 1)) > cum).sum(1).clip(0, 119)
            wsim = np.zeros((int(c.sum()), 120), bool)
            wsim[np.arange(int(c.sum())), j] = True
            pk = pick[c][sub]
            odds = RATE / np.clip(q3[c][sub], 1e-12, None)
            got = (np.where(pk & wsim[sub], odds * 100, 0.0)).sum(1)
            vals.append(got.sum() / (100 * pk.sum()))
        L.append(f"| 10点・{'条件なし' if th==0 else f'{th:.2f}以上'} | {np.mean(vals)*100:.1f}% | {np.max(vals)*100:.1f}% |")

    # ---- 的中率モードの条件
    L += ["\n## 3. 的中率モード: 市場の確信度がしきい値以上のレースだけ買う\n",
          "複勝1点。選定も条件も市場の2着以内確率のみ（モデルを使わない）。\n",
          "| しきい値 | 発火 | 1日あたり | 的中率 | 平均払戻 | 元返し | 最長連敗 | 回収率 | 95%区間 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    M = masks()
    P2 = np.stack([q3[:, m].sum(1) for m in M["複勝"]], 1)
    conf, sel = P2.max(1), P2.argmax(1)
    for th in (0.80, 0.85, 0.88, 0.90, 0.92):
        fire = c & (conf >= th)
        idx = np.flatnonzero(fire)
        if len(idx) < 100:
            continue
        ret = np.array([payout_of(raws[i], "複勝", int(sel[i])) for i in idx])
        lo, hi = roi_bootstrap(np.full(len(idx), 100.0), ret, n_boot=400)
        st = 0; best = 0
        for h in (ret > 0):
            st = 0 if h else st + 1
            best = max(best, st)
        L.append(f"| {th:.2f} | {len(idx):,} | {len(idx)/days_c:.1f}R | {(ret>0).mean()*100:.1f}% | "
                 f"{ret[ret>0].mean():.0f}円 | {(ret==100).mean()*100:.1f}% | {best}R | "
                 f"**{ret.sum()/(100*len(idx))*100:.1f}%** | {lo*100:.0f}〜{hi*100:.0f}% |")
    # ---- 4. 実測でいちばん良かった構成を、プラセボ込みで確定させる
    L += ["\n## 4. 穴モードで実測いちばん良かった構成（プラセボ込み）\n",
          "較正期待値による候補選定は効かなかった（上の§2、どれも帰無75%以下）。",
          "効いたのは**レース側の条件**だけ。`two_modes.md` の構成を確認期間で1回評価し直す。\n",
          "| 構成 | 発火 | 1日あたり | 1日の投資 | 的中率 | 平均払戻 | 最長連敗 | 回収率 | 95%区間 | プラセボ |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    ordq = np.argsort(-q3, 1)
    cumc = q3[c].cumsum(1)
    for thr, tl in ((0.0, "絞らない"), (0.70, "市場の万舟確率 上位30%"), (0.90, "上位10%")):
        band = np.zeros((n, 120), bool)
        np.put_along_axis(band, ordq[:, 19:40], True, 1)          # 人気20〜40
        fire = c if thr == 0 else (c & (q_man >= np.quantile(q_man[e], thr)))
        idx = np.flatnonzero(fire)
        ret = np.array([sum(payout_of(raws[i], "3連単", int(j)) for j in np.flatnonzero(band[i]))
                        for i in idx])
        stake = np.full(len(idx), 2100.0)
        lo, hi = roi_bootstrap(stake, ret, n_boot=600)
        st = 0; best = 0
        for h in (ret > 0):
            st = 0 if h else st + 1
            best = max(best, st)
        sub = fire[c]
        pv = []
        for _ in range(10):
            j = (RNG.random((int(c.sum()), 1)) > cumc).sum(1).clip(0, 119)
            ws = np.zeros((int(c.sum()), 120), bool)
            ws[np.arange(int(c.sum())), j] = True
            odds = RATE / np.clip(q3[c][sub], 1e-12, None)
            pv.append(np.where(band[c][sub] & ws[sub], odds * 100, 0.0).sum() / (2100.0 * sub.sum()))
        L.append(f"| 人気20〜40（21点）・{tl} | {len(idx):,} | {len(idx)/days_c:.1f}R | "
                 f"{stake.sum()/days_c:,.0f}円 | {(ret>0).mean()*100:.1f}% | {ret[ret>0].mean():,.0f}円 | "
                 f"{best}R | **{ret.sum()/stake.sum()*100:.1f}%** | {lo*100:.0f}〜{hi*100:.0f}% | "
                 f"{np.mean(pv)*100:.1f}% |")
    L += ["",
          "**判定**: 較正期待値による候補選定は不発。レース条件（市場が示す万舟確率）だけが効く。",
          "ただし最良構成でも95%区間の下限が帰無（75%）に接するところまでしか来ない。",
          "**「でたらめに穴を買うよりわずかにマシ」が上限で、利益が出る設定は存在しない。**"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
