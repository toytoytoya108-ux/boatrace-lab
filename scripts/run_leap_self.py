"""飛躍1: 市場を市場自身で均す（自己整合化）。外の情報をゼロにして歪みを取れるか。

■ 着想
  これまでの検証は全部「市場に外から情報を足す」形で、5族すべて 0.001 nats に収束した（`goal1_summary.md`）。
  逆をやる。**外の情報を一切足さず、市場の中の矛盾だけを使う。**

  3連単オッズ q(120) から、市場自身の周辺確率を取り出す:
      p1_a = Σ_{b,c} q(a,b,c)（aが1着）, p2_a = aが2着以内, p3_a = aが3着以内
  これをツール本体と同じ構造（位置別割引つき Plackett–Luce）に通して整合版 q* を作る。
      q*(a→b→c) = p1_a/Σp1 × p2_b^λ2/Σ_{j≠a} × p3_c^λ3/Σ_{j≠a,b}
  q と q* のズレは「市場が自分の周辺確率と矛盾している量」＝群衆のノイズ（出目の迷信、ゾロ目、
  人気の組み合わせへの集中）の可能性がある。**ノイズなら、均した q* の方が真の確率に近いはず。**

  賭けの向き: q < q* の買い目 ＝ 市場が自分の構造より**安く**売っている点。比 r = q/q* が小さいほど割安。

■ 飛躍の度合い
  この操作は情報を一切追加しない（q の中だけで閉じている）。だから「市場を超える情報が無い」という
  これまでの結論と矛盾しない。それでも効く可能性があるのは、**校正ではなく денoise だから**。

■ 手順（先読み防止）
  λ2,λ3 は探索期間（1〜5月）で KL(q‖q*) を最小化して決め、確認期間（6〜8月）はその値を当てるだけ。
  帰無は y ~ Multinomial(q)（勝者をシャッフルしない。`goal1_summary.md` の教訓）。

出力: reports/research/leap_self.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boatlab.backtest.metrics import roi_bootstrap  # noqa: E402
from boatlab.config import ROOT  # noqa: E402
from boatlab.model.trifecta import PERMS  # noqa: E402

OUT = Path(ROOT) / "reports" / "research" / "leap_self.md"
CACHE = Path("/tmp/claude-0/mix10_cache.npz")
_A = np.array([p[0] for p in PERMS]); _B = np.array([p[1] for p in PERMS]); _C = np.array([p[2] for p in PERMS])
RATE = 0.75


def marginals(Q):
    """(N,120) → 1着 / 2着以内 / 3着以内 の周辺確率 (N,6)。"""
    n = len(Q)
    p1 = np.zeros((n, 6)); p2 = np.zeros((n, 6)); p3 = np.zeros((n, 6))
    for a in range(6):
        p1[:, a] = Q[:, _A == a].sum(1)
        p2[:, a] = Q[:, (_A == a) | (_B == a)].sum(1)
        p3[:, a] = Q[:, (_A == a) | (_B == a) | (_C == a)].sum(1)
    return p1, p2, p3


def pl_from_marginals(p1, p2, p3, lam2, lam3, eps=1e-9):
    """周辺確率 → 位置別割引つき Plackett–Luce の120通り（本体 trifecta.py と同じ形）。"""
    w = np.clip(p1, 0, None) + eps
    w = w / w.sum(1, keepdims=True)
    s2 = np.power(np.clip(p2, 0, None) + eps, lam2)
    s3 = np.power(np.clip(p3, 0, None) + eps, lam3)
    S2 = s2.sum(1, keepdims=True); S3 = s3.sum(1, keepdims=True)
    p = w[:, _A] * (s2[:, _B] / (S2 - s2[:, _A])) * (s3[:, _C] / (S3 - s3[:, _A] - s3[:, _B]))
    return p / p.sum(1, keepdims=True)


def kl(q, qs):
    return float(np.sum(q * np.log(np.clip(q, 1e-12, None) / np.clip(qs, 1e-12, None)), axis=1).mean())


def summarize(ret, stake, seed=0):
    if stake.sum() <= 0:
        return None
    lo, hi = roi_bootstrap(stake, ret, n_boot=300)
    roi = ret.sum() / stake.sum()
    return dict(n=len(ret), roi=roi, lo=lo, hi=hi, hit=(ret > 0).mean(), ratio=roi / RATE,
                avg=ret[ret > 0].mean() if (ret > 0).any() else 0.0, stake=stake.mean())


def main():
    z = np.load(CACHE, allow_pickle=True)
    date, O, W, P = pd.DatetimeIndex(z["date"]), z["O"], z["W"], z["P"]
    inv = np.where(np.isfinite(O), 1.0 / np.nan_to_num(O, nan=1e9), 0.0)
    Q = inv / inv.sum(1, keepdims=True)
    N = len(Q)
    half = np.asarray(date <= "2026-05-31")
    p1, p2, p3 = marginals(Q)

    # λ は探索期間で KL(q‖q*) を最小化（確認期間は当てるだけ）
    best = (None, 1e9)
    grid = [round(x, 2) for x in np.arange(0.30, 1.31, 0.05)]
    for l2 in grid:
        for l3 in grid:
            v = kl(Q[half], pl_from_marginals(p1[half], p2[half], p3[half], l2, l3))
            if v < best[1]:
                best = ((l2, l3), v)
    (lam2, lam3), kl_ex = best
    QS = pl_from_marginals(p1, p2, p3, lam2, lam3)
    R = Q / np.clip(QS, 1e-12, None)                      # 比。1未満＝市場が自分の構造より安く売っている
    kl_race = np.sum(Q * np.log(np.clip(Q, 1e-12, None) / np.clip(QS, 1e-12, None)), axis=1)
    win_r = R[np.arange(N), W]

    L = [f"# 飛躍1: 市場を市場自身で均す（2026年・{N:,}R・確定オッズ）\n",
         "外の情報を一切足さず、3連単オッズ q から市場自身の周辺確率（1着・2着以内・3着以内）を取り出し、",
         "本体と同じ位置別割引つき Plackett–Luce に通して**内部整合な q\\*** を作る。",
         "`r = q / q*` が小さい買い目＝**市場が自分の構造より安く売っている点**。",
         f"λ は探索期間（1〜5月）で KL(q‖q\\*) 最小化 → **λ2={lam2}, λ3={lam3}**（確認期間は当てるだけ）。\n",
         "## 1. 市場はどれだけ自分と矛盾しているか\n",
         f"- 探索期間の平均 KL(q‖q\\*) = **{kl_ex:.4f} nats**（市場の全情報量 0.6353 nats の {kl_ex/0.6353*100:.1f}%）。",
         f"- 確認期間 = {kl(Q[~half], QS[~half]):.4f} nats。レース間のばらつき: "
         f"中央値 {np.median(kl_race):.4f}、上位10% {np.quantile(kl_race, 0.9):.4f}、最大 {kl_race.max():.4f}。",
         f"- 比 r の分布: 最小 {R.min():.3f}、5%点 {np.quantile(R, 0.05):.3f}、中央値 {np.median(R):.3f}、",
         f"  95%点 {np.quantile(R, 0.95):.3f}、最大 {R.max():.3f}。",
         "",
         "**矛盾は実在する。** 問題はそれがノイズ（均せば良くなる）か、情報（市場が構造に入らない何かを知っている）か。\n",
         "## 2. どちらが正しいか — 実際の勝ち目の r はどこにあるか\n",
         "勝った買い目の r が1より小さい側に偏るなら「市場は安い目を過小評価していた」＝均すのが正しい。\n",
         "| | 全買い目 | 実際に勝った買い目 |", "|---|---:|---:|",
         f"| 比 r の中央値 | {np.median(R):.3f} | {np.median(win_r):.3f} |",
         f"| r < 1 の割合 | {(R < 1).mean()*100:.1f}% | {(win_r < 1).mean()*100:.1f}% |",
         f"| r < 0.8 の割合 | {(R < 0.8).mean()*100:.1f}% | {(win_r < 0.8).mean()*100:.1f}% |",
         "",
         "## 3. 対数損失: q と q\\* のどちらが実際の結果に近いか（確認期間）\n"]
    for nm, M in (("市場そのまま q", Q), (f"自己整合版 q* (λ2={lam2}, λ3={lam3})", QS)):
        ll = -np.log(np.clip(M[np.arange(N), W], 1e-12, None))
        L.append(f"- {nm}: 確認期間の3連単対数損失 **{ll[~half].mean():.4f}**（探索 {ll[half].mean():.4f}）")
    mix = []
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        M = Q ** (1 - a) * QS ** a
        M = M / M.sum(1, keepdims=True)
        ll = -np.log(np.clip(M[np.arange(N), W], 1e-12, None))
        mix.append((a, ll[half].mean(), ll[~half].mean()))
    L += ["", "混ぜたらどうか（`q^(1−a) × q*^a` を正規化。a=0が市場、a=1が整合版）:", "",
          "| a | 探索 対数損失 | 確認 対数損失 |", "|---|---:|---:|"]
    for a, e, c in mix:
        L.append(f"| {a:g} | {e:.4f} | {c:.4f} |")

    # ---- 4. 買ってみる（いちばん飛躍した形）
    order_r = np.argsort(R, axis=1)                 # r の小さい順＝割安な順
    order_q = np.argsort(-Q, axis=1)
    L += ["\n## 4. いちばん飛躍した買い方: 割安な順に k 点（他の条件を一切付けない）\n",
         "**帰無（市場が正しい）なら回収率は75%。** 比 = 回収率 ÷ 0.75。\n",
          "| 買い方 | 点数 | 的中率 | 平均払戻 | 回収率 | 比 | 95%区間 | 前半 | 後半 |", "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    def buy(sel_idx, mask=None):
        m = np.ones(N, bool) if mask is None else mask
        idx = np.flatnonzero(m)
        hit = np.array([W[i] in set(sel_idx[i].tolist()) for i in idx])
        ret = np.where(hit, P[idx], 0.0)
        stake = np.full(len(idx), 100.0 * sel_idx.shape[1])
        return ret, stake
    rows = []
    for k in (1, 3, 5, 10, 15, 20, 30):
        for nm, od in (("割安順（自己整合）", order_r), ("人気順（対照）", order_q)):
            sel = od[:, :k]
            r = summarize(*buy(sel))
            hs = [summarize(*buy(sel, m)) for m in (half, ~half)]
            rows.append((nm, k, r, hs))
            L.append(f"| {nm} | {k} | {r['hit']*100:.2f}% | {r['avg']:,.0f}円 | **{r['roi']*100:.1f}%** | "
                     f"{r['ratio']:.3f} | {r['lo']*100:.0f}〜{r['hi']*100:.0f}% | {hs[0]['roi']*100:.1f}% | {hs[1]['roi']*100:.1f}% |")

    # ---- 5. 帰無シミュレーション
    rng = np.random.default_rng(17)
    L += ["\n## 5. 帰無シミュレーション（市場が正しい世界で同じ買い方を20回）\n",
          "| 買い方 | 点数 | 実測 | 帰無 平均 | 帰無 範囲 | 実測以上 |", "|---|---:|---:|---:|---|---:|"]
    cum = np.cumsum(Q, axis=1)
    sims = np.zeros((20, N), int)
    for s in range(20):
        u = rng.random(N)
        sims[s] = (cum < u[:, None]).sum(1).clip(0, 119)
    for k in (5, 10, 20):
        for nm, od in (("割安順（自己整合）", order_r), ("人気順（対照）", order_q)):
            sel = od[:, :k]
            real = summarize(*buy(sel))["roi"]
            vals = []
            for s in range(20):
                w2 = sims[s]
                hit = np.array([w2[i] in set(sel[i].tolist()) for i in range(N)])
                pay = np.where(hit, RATE * 100 / np.clip(Q[np.arange(N), w2], 1e-12, None), 0.0)
                vals.append(pay.sum() / (100.0 * k * N))
            v = np.array(vals)
            L.append(f"| {nm} | {k} | {real*100:.1f}% | {v.mean()*100:.1f}% | {v.min()*100:.1f}〜{v.max()*100:.1f}% | {(v >= real).sum()}/20 |")

    # ---- 6. ズレは「どんな形」をしているか（艇番の組み合わせごとの平均 r）
    L += ["\n## 6. 市場が Plackett–Luce からズレる『形』（1着艇 × 2着艇 ごとの平均 r）\n",
          "r > 1 ＝ 市場はその組み合わせを、自分の周辺確率から作った構造版より**高く**売っている。",
          "PL は「2着は個々の強さだけで決まる」と仮定するので、**誰が1着かによって2着の顔ぶれが変わる効果**を表現できない。\n",
          "| 1着＼2着 | 1号艇 | 2 | 3 | 4 | 5 | 6 |", "|---|---:|---:|---:|---:|---:|---:|"]
    M12 = np.zeros((6, 6))
    for a in range(6):
        cells = []
        for b in range(6):
            if a == b:
                cells.append("—"); continue
            m = (_A == a) & (_B == b)
            v = float(np.average(R[:, m], weights=np.broadcast_to(Q[:, m], R[:, m].shape)))
            M12[a, b] = v
            cells.append(f"{'**' if v > 1.10 else ''}{v:.2f}{'**' if v > 1.10 else ''}")
        L.append(f"| {a+1}号艇が1着 | " + " | ".join(cells) + " |")
    L += ["", "（市場確率で重みづけした平均。太字は1.10超＝市場が構造版より1割以上高く売っている組み合わせ）\n"]

    # ---- 7. その形を補正項として学習し、市場の情報をどれだけ取り戻せるか
    L += ["## 7. ズレを補正項として学習すると、どれだけ取り戻せるか\n",
          "1着艇×2着艇の6×6補正を**探索期間の市場オッズだけ**から作り（結果は使わない）、確認期間に当てる。",
          "これは「市場が知っていて PL が表現できない形」を、市場から写し取る操作。\n"]
    C12 = np.ones((6, 6))
    for a in range(6):
        for b in range(6):
            if a == b:
                continue
            m = (_A == a) & (_B == b)
            C12[a, b] = float(np.average(R[half][:, m], weights=np.broadcast_to(Q[half][:, m], R[half][:, m].shape)))
    corr = C12[_A, _B]
    QC = QS * corr
    QC = QC / QC.sum(1, keepdims=True)
    L += ["| 版 | 探索 対数損失 | 確認 対数損失 | 市場との差（確認） |", "|---|---:|---:|---:|"]
    llq = -np.log(np.clip(Q[np.arange(N), W], 1e-12, None))
    for nm, M in (("市場そのまま q", Q), ("自己整合版 q*（PLのみ）", QS), ("q* × 1着×2着 補正", QC)):
        ll = -np.log(np.clip(M[np.arange(N), W], 1e-12, None))
        L.append(f"| {nm} | {ll[half].mean():.4f} | {ll[~half].mean():.4f} | {ll[~half].mean() - llq[~half].mean():+.4f} |")
    gap = (-np.log(np.clip(QS[np.arange(N), W], 1e-12, None))[~half].mean() - llq[~half].mean())
    got = gap - (-np.log(np.clip(QC[np.arange(N), W], 1e-12, None))[~half].mean() - llq[~half].mean())
    L += ["", f"**PL 構造が失っていた {gap:.4f} nats のうち、1着×2着の補正だけで {got:.4f} nats（{got/gap*100:.0f}%）を取り戻した。**",
          "補正は市場オッズだけから作っており、結果を一切使っていない（探索期間で作って確認期間に当てている）。\n"]
    L += ["## 8. 結論",
          "",
          "### 飛躍の仮説は外れた。しかも向きが逆だった",
          "「市場の内部矛盾はノイズだから均せば良くなる」は**明確に否定された**。",
          "- 勝った買い目の r は中央値 1.049（全買い目 1.008）。**市場が構造版より高く売っている目ほど、実際に来る。**",
          "- 対数損失は 市場 3.7058 < 自己整合版 3.7443。混ぜても単調に悪化する（a=0.25 で既に悪い）。",
          "- 割安な順に買うと回収率 64〜69%（比 0.85〜0.92）。帰無シミュレーションで **20/20** が実測を上回った。",
          "  **でたらめに買うより明確に悪い。** 対照の人気順は 76.6〜77.0%（比 1.02）で帰無 0/20。",
          "",
          "**市場の『矛盾』は雑音ではなく知識だった。** これは今回いちばん価値のある否定結果で、",
          "「市場を市場自身で改良する」という道は閉じた。",
          "",
          "### 代わりに見えたもの: 構造の不足は 0.0385 nats。これまでの全努力の38倍",
          "市場の周辺確率をそのまま使っても、Plackett–Luce に通すだけで **0.0385 nats** 失う。",
          "比較のため: 風・モーター・今節成績・艇番バイアス・公表2連率の5族を足して得られたのは**全部 0.001 nats前後**",
          "（`goal1_summary.md`）。**構造の穴は、情報の穴より一桁半大きい。**",
          "",
          "**そして本体の Model 1.0 はまさにこの Plackett–Luce を使っている**（`boatlab/model/trifecta.py`）。",
          "モデルの対数損失 3.78 と市場 3.70 の差 0.08 のうち、**約半分（0.039）は情報ではなく表現力の不足**である可能性が高い。",
          "",
          "### ズレの形は物理的に読める",
          "6×6 の表は決まり手そのものだった:",
          "- **1号艇が逃げたとき**、市場は 2着に 2号艇(1.17)・3号艇(1.11) を構造版より高く、5号艇(0.87)・6号艇(0.82) を安く売る。",
          "- **4号艇がまくったとき**、2着に 5号艇 **2.31**・6号艇 **2.11**。5号艇が1着なら 6号艇 **2.25**、4号艇 **1.50**。",
          "  外側が1着になるレースでは、**その外側の艇が引き連れてきた艇が2着に残る**。",
          "- PL は「2着は個々の強さだけで決まる」と仮定するので、この『誰が1着かで2着の顔ぶれが変わる』効果を原理的に書けない。",
          "- `manshu.md` の「万舟の決まり手は 逃げ3.6% / 差し27.0% / まくり33.2% / まくり差し36.5%」と同じ絵を、",
          "  今度は**オッズの側から**見ている。",
          "",
          "### 引き算して残るもの（次にやること）",
          "1. **Model 1.3 = trifecta.py に 1着艇×2着艇の相互作用を入れる。** 今回は市場オッズから写した粗い6×6で",
          "   0.0385 のうち 0.0102（26%）を取り戻した。本来は**結果データから学習**でき、場別・コース別に細分もできる。",
          "   必要なデータは無い（既にある）。**手持ちで動かせる最大の一手。**",
          "2. ただし期待しすぎないこと。`model12_wf.md` では対数損失が改善しても回収率は横ばいだった（78.9→79.1%）。",
          "   **精度が上がっても、市場を超えない限り回収率は動かない。** 今回の 0.039 は必要量 0.2877 の13%にすぎない。",
          "3. 買い方としては何も採用しない。割安順の買いは帰無より悪い。",
          "",
          "### この実験の位置づけ",
          "外の情報をゼロにして歪みを取ろうとしたら、**市場の側の構造知識**が見えた。",
          "「市場に勝つ情報」は見つからなかったが、「**我々のモデルが市場に負けている理由の内訳**」が初めて分解できた。",
          ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
