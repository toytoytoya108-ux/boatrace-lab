"""全レースで「穴」を買ったらどうなるか（オッズ上限なし）。

これまで測ってきたのは**本命側の端**（`sweet_spot.md`・`favorite_edge.md`）。
その鏡像を測る。各券種で市場がいちばん**確からしくない**と見る1点を、全レース100円ずつ。
オッズ上限は設けない（3連単なら万舟どころか十万舟も買う）。

あわせて「何番目に人気か」の段階も出し、本命端から穴端までの曲線を1枚にする。

## 読み方
市場が完全に正しければ、どの買い目でも回収率はちょうど**払戻率 75%** になる。
75%からのズレが、そのまま群衆の歪み。

    回収率 = 実的中率 ÷ 市場確率 × 0.75
    → 回収率 ÷ 0.75 = 実的中率 ÷ 市場確率 ＝ 歪みの比（1.00 なら市場は正しい）

`crowd_bias.md` の実測では市場は2〜6号艇を過大評価しており（6号艇が最大 +0.228）、
穴側の比は 1.00 を**下回る**はず。つまり回収率は 75% より悪くなるという予想。

パラメータはゼロ（選定に学習を使わない）なので、探索/確認の分割は不要。2026年の全レースを使う。

出力: reports/backtest/longshot.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from boatlab.backtest.metrics import roi_bootstrap
from boatlab.config import ROOT

from run_sweet_spot import KEY, load, masks, payout_of  # noqa: E402  同じ土台を使う

OUT = Path(ROOT) / "reports" / "backtest" / "longshot.md"
RATE = 0.75


def row(bt, raws, P, sel, label):
    n = len(sel)
    ret = np.array([payout_of(raws[i], bt, int(sel[i])) for i in range(n)])
    stake = np.full(n, 100.0)
    lo, hi = roi_bootstrap(stake, ret, n_boot=600)
    roi = ret.sum() / stake.sum()
    qm = P[np.arange(n), sel].mean()
    hit = (ret > 0).mean()
    return dict(bt=bt, label=label, n=n, q=qm, hit=hit, nhit=int((ret > 0).sum()),
                avg=(ret[ret > 0].mean() if (ret > 0).any() else 0.0),
                mx=ret.max(), roi=roi, lo=lo, hi=hi,
                bias=(hit / qm if qm > 0 else float("nan")))


def main():
    q3, raws, date = load()
    M = masks()
    n = len(date)
    L = [f"# 全レースで穴を買ったらどうなるか（{n:,}R・2026年・オッズ上限なし）\n",
         "各券種で市場がいちばん**確からしくないと見る1点**を、全レース100円ずつ。",
         "選定に学習を使わないのでパラメータはゼロ。探索/確認の分割は不要で、2026年の全レースを使う。\n",
         "市場が完全に正しければ、どの買い目でも回収率はちょうど払戻率 **75%** になる。",
         "`歪みの比 = 回収率 ÷ 0.75 = 実的中率 ÷ 市場確率`。1.00 なら市場は正しい。\n",
         "## 1. 穴だけを買った結果\n",
         "| 券種 | 買った点 | 市場確率 | 的中 | 的中数 | 平均払戻 | 最高払戻 | 回収率 | 95%区間 | 歪みの比 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    rows = []
    for bt, ms in M.items():
        P = np.stack([q3[:, m].sum(1) for m in ms], 1)
        r = row(bt, raws, P, P.argmin(1), "いちばん人気がない1点")
        rows.append(r)
        L.append(f"| {bt} | {r['label']} | {r['q']*100:.3f}% | {r['hit']*100:.2f}% | {r['nhit']:,} | "
                 f"{r['avg']:,.0f}円 | {r['mx']:,.0f}円 | **{r['roi']*100:.1f}%** | "
                 f"{r['lo']*100:.0f}〜{r['hi']*100:.0f}% | {r['bias']:.3f} |")

    L += ["\n## 2. 本命端から穴端までの曲線（単勝・人気順）\n",
          "市場の1着確率が高い順に1位〜6位。全レースでその順位の艇を100円。\n",
          "| 買った点 | 市場確率 | 的中 | 平均払戻 | 回収率 | 95%区間 | 歪みの比 |",
          "|---|---:|---:|---:|---:|---|---:|"]
    Pw = np.stack([q3[:, m].sum(1) for m in M["単勝"]], 1)
    order = np.argsort(-Pw, 1)
    for k in range(6):
        r = row("単勝", raws, Pw, order[:, k], f"人気{k+1}番目")
        L.append(f"| {r['label']} | {r['q']*100:.2f}% | {r['hit']*100:.2f}% | {r['avg']:,.0f}円 | "
                 f"**{r['roi']*100:.1f}%** | {r['lo']*100:.0f}〜{r['hi']*100:.0f}% | {r['bias']:.3f} |")

    L += ["\n## 3. 3連単を人気順の帯で見る\n",
          "120通りを市場の確率が高い順に並べ、その順位の1点を全レースで買う。\n",
          "| 買った点 | 市場確率 | 的中 | 的中数 | 平均払戻 | 最高払戻 | 回収率 | 95%区間 | 歪みの比 |",
          "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    P3 = q3
    ord3 = np.argsort(-P3, 1)
    for k in (0, 1, 4, 9, 19, 39, 59, 79, 99, 109, 114, 119):
        r = row("3連単", raws, P3, ord3[:, k], f"人気{k+1}番目")
        L.append(f"| {r['label']} | {r['q']*100:.4f}% | {r['hit']*100:.3f}% | {r['nhit']:,} | "
                 f"{r['avg']:,.0f}円 | {r['mx']:,.0f}円 | **{r['roi']*100:.1f}%** | "
                 f"{r['lo']*100:.0f}〜{r['hi']*100:.0f}% | {r['bias']:.3f} |")

    # --- 穴が過大評価なら、その「逆」は取れないのか
    L += ["\n## 4. 穴の逆は取れないのか（人気薄を外して残りを払戻均等で買う）\n",
          "穴が過大評価なら、穴を**売りたい**。しかしパリミュチュエルに売り側は無い。",
          "いちばん近いのは「穴を外して残り全部を買う」。どの艇が来ても払戻が同額になるように",
          "賭け金を市場確率に比例させる（payout_equal）。\n",
          "| 外した点 | 残り点数 | 外した点の市場確率 | 外した点の実勝率 | 回収率 | 95%区間 |",
          "|---|---:|---:|---:|---:|---|"]
    roi_ladder = []
    ordw = np.argsort(Pw, 1)                      # 人気がない順
    winner = np.array([int(np.argmax([payout_of(raws[i], "単勝", b) for b in range(6)]))
                       for i in range(n)])
    pay = np.array([[payout_of(raws[i], "単勝", b) for b in range(6)] for i in range(n)])
    for k in range(0, 6):
        ex = ordw[:, :k]                          # 外す k 点
        inc = np.ones((n, 6), bool)
        np.put_along_axis(inc, ex, False, 1)
        wq = np.where(inc, Pw, 0.0)
        wq = wq / wq.sum(1, keepdims=True)        # 払戻均等＝市場確率に比例
        stake = wq * 1000.0
        got = (stake * pay / 100.0 * (pay > 0)).sum(1)
        lo, hi = roi_bootstrap(stake.sum(1), got, n_boot=600)
        exq = np.take_along_axis(Pw, ex, 1).sum(1)
        exhit = np.take_along_axis((pay > 0), ex, 1).any(1)
        lab = "外さない（全6点）" if k == 0 else f"人気がない{k}点"
        L.append(f"| {lab} | {6-k}点 | {exq.mean()*100:.2f}% | {exhit.mean()*100:.2f}% | "
                 f"**{got.sum()/stake.sum()*100:.1f}%** | {lo*100:.0f}〜{hi*100:.0f}% |")
        roi_ladder.append(got.sum()/stake.sum())
    q_L = Pw.min(1).mean()
    roi_L = [r for r in rows if r["bt"] == "単勝"][0]["roi"]
    leak = q_L * (RATE - roi_L)
    L += ["",
          f"**外すほど良くなるが、それは「本命に寄る」ことと同じ**。賭け金を市場確率に比例させるので、",
          f"最不人気を外すと、その分の金が本命側に移る。並びは単調に上がり、最後は人気1点だけの "
          f"{roi_ladder[-1]*100:.1f}% ＝ すでに `sweet_spot.md` で測った単勝の本命端に一致する。",
          "**100%を越える点はどこにも無い。**",
          "",
          "「穴が過大評価なら、その逆で儲かるのでは」への答え: **逆を取る取引が存在しない。**",
          f"仮に最不人気を空売りできれば、市場確率 {q_L*100:.2f}% に対し実勝率 "
          f"{[r for r in rows if r['bt']=='単勝'][0]['hit']*100:.2f}% なので 100円につき約 "
          f"{(1-(RATE/q_L)*[r for r in rows if r['bt']=='単勝'][0]['hit'])*100:.0f}円 の利益になるが、",
          "公営競技に売り側は無い。買いで近づけるのは「それ以外を全部買う」だけで、上の表のとおり。",
          "",
          f"穴側の誤りが市場全体に対して持つ大きさも小さい。最不人気の買い手が払戻率より損している分は",
          f"プール全体の **{leak*100:.2f}%**（{q_L*100:.2f}% × ({RATE*100:.0f}% − {roi_L*100:.1f}%)）で、",
          "これが残り全員に配られても **+0.7ポイント程度**。",
          "**市場のいちばん大きな誤りは、いちばん小さな確率の上に乗っている。**",
          "",
          "※ 注意: この表の賭け金は3連単プールが示す確率、払戻は単勝の**確定**配当を使っている。",
          "2つのプールの値付けの差（`pool_arbitrage.md`: 単勝プール 1.2405 / 3連単プール 1.1474）が",
          "一部混ざっている。その差は締切前には取れないことを観測フェーズで確認済み（中央値 −50%）。",
          ""]

    w = [r for r in rows if r["bt"] == "単勝"][0]
    t = [r for r in rows if r["bt"] == "3連単"][0]
    L += ["\n## 結論\n",
          f"- 単勝でいちばん人気がない艇を全レース: 回収率 **{w['roi']*100:.1f}%**（歪みの比 {w['bias']:.3f}）",
          f"- 3連単でいちばん人気がない1点を全レース: 回収率 **{t['roi']*100:.1f}%**"
          f"（的中 {t['nhit']:,}本／{t['n']:,}R、歪みの比 {t['bias']:.3f}）",
          "",
          "穴側の歪みの比は 1.00 を**下回る**。市場は人気薄を過大評価しているので、"
          "穴を買うと払戻率75%よりさらに悪くなる。`crowd_bias.md` が艇番ダミーで測った"
          "「市場は2〜6号艇を過大評価（6号艇が最大 +0.228）」と同じことが、回収率の形で出ている。",
          "",
          "**最良の買い方（複勝・市場確信度上位10%の99.1%）と、最悪の買い方の差が、"
          "この競技で選択が動かせる幅の全体。**"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
