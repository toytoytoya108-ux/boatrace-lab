"""3連単・回収率100%超えの買い方を約1,000パターンで総当たり検証する（2026-10-01）。

目的: ユーザー依頼「3連単で回収率100%を超える買い方を、レースの絞り込みも含めて探す。1,000通りで一度報告」。
データ: research_data/bt2026_top20.npz（2026年1〜8月・36,924R・Model 1.0 上位20点の確率と確定3連単オッズ・実際の3連単と公式払戻）。
       lab.db（46万R）はこの環境に無いので使えない。固定の目（族D）だけは結果と払戻のみで計算できる。

手順（事前登録。データを見てから変えない）:
  - 探索 = 2026-01〜05、確認 = 2026-06〜08。しきい値（自信の分位点など）は探索期間の分布だけから決める。
  - 第1段: 探索で n ≥ 200 かつ 回収率 ≥ 1.00 のパターンを候補にする。
  - 第2段: 候補を確認期間で1回だけ評価。合格 = 確認の回収率 ≥ 1.00 かつ 95%区間下限 > 0.95。
  - 帰無対照: 「市場確率が真」（勝ち目 ~ Categorical(0.75/オッズ)）で結果を20回作り直し、同じ第1段・第2段を通して
    偶然の候補数・合格数を数える（族Dは120通りの市場確率が無いので帰無なし、区間だけ）。
  - 注意: オッズは確定オッズ。締切前オッズで買うと当たった目は中央値 −9.4%（2〜30倍帯、9/16以降の実測）下がる。

出力: reports/research/grid1000.csv（全パターン）、reports/research/grid1000.md（報告）。
"""
from __future__ import annotations

import itertools
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "research_data" / "bt2026_top20.npz"
OUT_CSV = ROOT / "reports" / "research" / "grid1000.csv"
OUT_MD = ROOT / "reports" / "research" / "grid1000.md"

EXPLORE_END = "2026-05-31"
MIN_N = 200
N_NULL = 20
PASS_ROI = 1.00
PASS_CI_LO = 0.95
RNG = np.random.default_rng(20261001)

PERMS = list(itertools.permutations(range(6), 3))
LABELS = [f"{a+1}-{b+1}-{c+1}" for a, b, c in PERMS]


# ---------------------------------------------------------------- データ
def load():
    z = np.load(NPZ, allow_pickle=False)
    top = z["top"].astype(int)                 # (n,20) 120通りの添字（モデル確率降順）
    P = z["p"].astype(np.float64)              # (n,20) モデル確率
    O = z["odds"].astype(np.float64)           # (n,20) 確定オッズ
    tri = z["tri"].astype(int)                 # 実際の3連単（添字）
    pay = z["payout"].astype(np.float64)       # 公式払戻（100円あたり）
    d = z["date"].astype(str)
    rid = z["race_id"].astype(np.int64)
    n = len(tri)
    # 勝ち目が上位20点の何番目か（無ければ -1）
    hit = top == tri[:, None]
    wj = np.where(hit.any(1), hit.argmax(1), -1)
    assert n == 36924 and (pay > 0).all() and (O >= 1).all()
    feats = dict(
        date=d, explore=(d <= EXPLORE_END), stadium=z["stadium"].astype(int), race_no=(rid % 100).astype(int),
        weekday=np.array([date.fromisoformat(x).weekday() for x in d]),
        p1=P[:, 0], conf3=P[:, :3].sum(1), conf5=P[:, :5].sum(1), conf10=P[:, :10].sum(1),
        o1=O[:, 0], omin=O.min(1), rough=(O >= 100).sum(1), sinv=(1 / O).sum(1),
        ev=P * O, orank=np.argsort(np.argsort(O, axis=1), axis=1),  # orank 0 = 上位20点内で最も人気
    )
    return dict(n=n, top=top, P=P, O=O, tri=tri, pay=pay, wj=wj, **feats)


def null_winners(D, sims: int) -> np.ndarray:
    """帰無「市場確率が真」。上位20点の確率 q=0.75/オッズ、残りは『その他』（買い目外なので払戻0）。"""
    q = 0.75 / D["O"]
    other = np.clip(1 - q.sum(1), 0, None)
    cum = np.cumsum(np.concatenate([q, other[:, None]], axis=1), axis=1)
    cum /= cum[:, -1:]
    u = RNG.random((sims, D["n"], 1))
    w = (u > cum[None]).sum(2)           # 0..20（20 = その他）
    return np.where(w >= 20, -1, w)


# ---------------------------------------------------------------- 評価
def roi_ci(stake: np.ndarray, pay: np.ndarray, mask: np.ndarray) -> tuple[int, int, float, float, float]:
    """比推定量 ROI=Σpay/Σstake と、デルタ法の95%区間。mask は評価対象レース。"""
    st = stake[mask]; pa = pay[mask]
    used = st > 0
    st, pa = st[used], pa[used]
    n = int(len(st)); hits = int((pa > 0).sum())
    S = st.sum()
    if S <= 0:
        return 0, 0, float("nan"), float("nan"), float("nan")
    roi = pa.sum() / S
    a = pa - roi * st
    se = np.sqrt((a ** 2).sum()) / S
    return n, hits, float(roi), float(roi - 1.96 * se), float(roi + 1.96 * se)


def eval_top20(D, F: np.ndarray, S: np.ndarray, W: np.ndarray | None = None) -> dict:
    """上位20点内の賭け S (n,20)（円）、レース条件 F (n,)。W は帰無の勝ち目添字 (sims,n)（None なら実データ）。"""
    stake = S.sum(1) * F
    if W is None:
        wj = D["wj"]
        won = np.where(wj >= 0, S[np.arange(D["n"]), np.clip(wj, 0, None)], 0.0)
        pay = won * D["pay"] / 100 * F
        out = {}
        for tag, m in (("exp", D["explore"]), ("conf", ~D["explore"])):
            n, h, r, lo, hi = roi_ci(stake, pay, m)
            out.update({f"n_{tag}": n, f"hits_{tag}": h, f"roi_{tag}": r, f"lo_{tag}": lo, f"hi_{tag}": hi})
        # 当たった目のオッズ（締切前換算の判断用）
        wo = D["O"][np.arange(D["n"]), np.clip(wj, 0, None)]
        sel = (wj >= 0) & (won > 0) & (F > 0)
        out["win_odds_med"] = float(np.median(wo[sel])) if sel.any() else float("nan")
        return out
    res = []
    for s in range(W.shape[0]):
        w = W[s]
        won = np.where(w >= 0, S[np.arange(D["n"]), np.clip(w, 0, None)], 0.0)
        pay = won * np.where(w >= 0, D["O"][np.arange(D["n"]), np.clip(w, 0, None)], 0.0) * F
        row = {}
        for tag, m in (("exp", D["explore"]), ("conf", ~D["explore"])):
            n, h, r, lo, hi = roi_ci(stake, pay, m)
            row.update({f"n_{tag}": n, f"roi_{tag}": r, f"lo_{tag}": lo})
        res.append(row)
    return res


def eval_fixed(D, F: np.ndarray, combo: int) -> dict:
    """固定の目を F のレース全部で100円。オッズ不要（結果と払戻だけ）。"""
    stake = 100.0 * F
    pay = np.where(D["tri"] == combo, D["pay"], 0.0) * F
    out = {}
    for tag, m in (("exp", D["explore"]), ("conf", ~D["explore"])):
        n, h, r, lo, hi = roi_ci(stake, pay, m)
        out.update({f"n_{tag}": n, f"hits_{tag}": h, f"roi_{tag}": r, f"lo_{tag}": lo, f"hi_{tag}": hi})
    sel = (D["tri"] == combo) & (F > 0)
    out["win_odds_med"] = float(np.median(D["pay"][sel]) / 100) if sel.any() else float("nan")
    return out


# ---------------------------------------------------------------- パターン生成
def quantile_thresholds(D):
    """自信フィルタのしきい値は探索期間の分布だけから決める（確認期間を覗かない）。"""
    e = D["explore"]
    th = {}
    for key in ("p1", "conf3", "conf5", "conf10"):
        th[key] = {q: float(np.quantile(D[key][e], q / 100)) for q in (50, 70, 90, 95)}
    th["sinv30"] = float(np.quantile(D["sinv"][e], 0.30))
    return th


def conf_filter(D, th, key, q):
    return np.ones(D["n"], bool) if q is None else D[key] >= th[key][q]


ODDS_BANDS = {"any": (0, np.inf), "<5": (0, 5), "5-10": (5, 10), "10-20": (10, 20), ">=20": (20, np.inf)}


def band_filter(D, band):
    lo, hi = ODDS_BANDS[band]
    return (D["o1"] >= lo) & (D["o1"] < hi)


def patterns(D, th):
    n = D["n"]; ar = np.arange(n)
    # 族A: モデル上位k点・100円均等
    for k in (1, 2, 3, 5, 10):
        key = {1: "p1", 2: "conf3", 3: "conf3", 5: "conf5", 10: "conf10"}[k]
        for q in (None, 50, 70, 90, 95):
            for band in ODDS_BANDS:
                S = np.zeros((n, 20)); S[:, :k] = 100
                F = conf_filter(D, th, key, q) & band_filter(D, band)
                yield dict(family="A", name=f"上位{k}点", params=f"自信{key}≥q{q} / 本命オッズ{band}", S=S, F=F)
    # 族B: 期待値 p×オッズ ≥ t の目を期待値順に最大m点
    ev = D["ev"]
    for t in (0.8, 1.0, 1.2, 1.5, 2.0):
        for m in (1, 3, 5, 10, 20):
            for cap in (20, 50, 100, np.inf):
                ok = (ev >= t) & (D["O"] <= cap)
                order = np.argsort(-np.where(ok, ev, -1), axis=1)
                S = np.zeros((n, 20))
                for r in range(m):
                    j = order[:, r]
                    S[ar, j] = np.where(ok[ar, j], 100, 0)
                for q in (None, 70, 90):
                    F = conf_filter(D, th, "conf10", q)
                    yield dict(family="B", name=f"期待値≥{t}・最大{m}点", params=f"オッズ≤{cap} / 自信conf10≥q{q}", S=S, F=F)
    # 族C: 市場人気帯（上位20点内の人気順）
    bands = {"人気1": (0, 0), "人気1-3": (0, 2), "人気1-5": (0, 4), "人気2-5": (1, 4), "人気4-8": (3, 7),
             "人気6-10": (5, 9), "人気11-20": (10, 19), "人気1-10": (0, 9)}
    rough_f = {"荒れ条件なし": np.ones(n, bool), "100倍以上≥3": D["rough"] >= 3, "100倍以上≥6": D["rough"] >= 6,
               "Σ1/オッズ下位30%": D["sinv"] <= th["sinv30"]}
    for bname, (lo, hi) in bands.items():
        S = np.where((D["orank"] >= lo) & (D["orank"] <= hi), 100.0, 0.0)
        for rname, RF in rough_f.items():
            for q in (None, 70, 90):
                F = RF & conf_filter(D, th, "conf10", q)
                yield dict(family="C", name=bname, params=f"{rname} / 自信conf10≥q{q}", S=S, F=F)
    # 族D: 固定の目（オッズ不要）
    rn = D["race_no"]
    for c in range(120):
        for gname, G in (("全R", np.ones(n, bool)), ("1〜11R", rn <= 11), ("12R", rn == 12)):
            yield dict(family="D", name=f"固定 {LABELS[c]}", params=gname, combo=c, F=G)
    for st in sorted(set(D["stadium"].tolist())):
        yield dict(family="D", name="固定 1-2-3", params=f"場{st}", combo=0, F=D["stadium"] == st)
    for r in range(1, 13):
        yield dict(family="D", name="固定 1-2-3", params=f"{r}R", combo=0, F=rn == r)
    # 族E: 上位10点の配分（払戻均等 ∝1/オッズ・確率比例）、合計1,000円
    for sname in ("払戻均等", "確率比例"):
        w = (1 / D["O"][:, :10]) if sname == "払戻均等" else D["P"][:, :10]
        S = np.zeros((n, 20)); S[:, :10] = 1000 * w / w.sum(1, keepdims=True)
        for q in (None, 50, 70, 90, 95):
            for band in ODDS_BANDS:
                F = conf_filter(D, th, "conf10", q) & band_filter(D, band)
                yield dict(family="E", name=f"上位10点・{sname}", params=f"自信conf10≥q{q} / 本命オッズ{band}", S=S, F=F)
    # 族F: 場別・R番号別（上位1点・3点）
    for k in (1, 3):
        S = np.zeros((n, 20)); S[:, :k] = 100
        for st in sorted(set(D["stadium"].tolist())):
            yield dict(family="F", name=f"上位{k}点", params=f"場{st}", S=S, F=D["stadium"] == st)
    S = np.zeros((n, 20)); S[:, :1] = 100
    for r in range(1, 13):
        yield dict(family="F", name="上位1点", params=f"{r}R", S=S, F=rn == r)
    # 族G: 曜日
    for k in (1, 3, 10):
        S = np.zeros((n, 20)); S[:, :k] = 100
        for wd, wn in enumerate("月火水木金土日"):
            yield dict(family="G", name=f"上位{k}点", params=f"{wn}曜", S=S, F=D["weekday"] == wd)


# ---------------------------------------------------------------- 本体
def main():
    D = load()
    th = quantile_thresholds(D)
    W = null_winners(D, N_NULL)
    rows = []
    null_rows = []   # (pattern_idx, sim, n_exp, roi_exp, n_conf, roi_conf, lo_conf)
    for i, pt in enumerate(patterns(D, th)):
        F = pt["F"].astype(float)
        if pt["family"] == "D":
            res = eval_fixed(D, F, pt["combo"])
            has_null = False
        else:
            res = eval_top20(D, F, pt["S"])
            has_null = True
            for s, nr in enumerate(eval_top20(D, F, pt["S"], W)):
                null_rows.append(dict(idx=i, sim=s, **nr))
        rows.append(dict(idx=i, family=pt["family"], name=pt["name"], params=pt["params"], has_null=has_null, **res))
        if (i + 1) % 100 == 0:
            print(f"{i+1} patterns done", file=sys.stderr)
    df = pd.DataFrame(rows)
    nd = pd.DataFrame(null_rows)
    # 第1段・第2段
    df["candidate"] = (df["n_exp"] >= MIN_N) & (df["roi_exp"] >= PASS_ROI)
    df["passed"] = df["candidate"] & (df["roi_conf"] >= PASS_ROI) & (df["lo_conf"] > PASS_CI_LO)
    df["roi_conf_pre"] = df["roi_conf"] * 0.906   # 締切前換算（当たり目2〜30倍帯の実測）。目安
    df.to_csv(OUT_CSV, index=False)
    # 帰無で同じ手順
    nd["candidate"] = (nd["n_exp"] >= MIN_N) & (nd["roi_exp"] >= PASS_ROI)
    nd["passed"] = nd["candidate"] & (nd["roi_conf"] >= PASS_ROI) & (nd["lo_conf"] > PASS_CI_LO)
    null_sum = nd.groupby("sim").agg(cands=("candidate", "sum"), passes=("passed", "sum"),
                                     best_conf=("roi_conf", "max")).reset_index()
    write_report(D, th, df, nd, null_sum)
    print(f"patterns={len(df)} candidates={int(df['candidate'].sum())} passed={int(df['passed'].sum())}")


def fmt(x, pct=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:.1f}%" if pct else f"{x:.3f}"


def table(df: pd.DataFrame, cols, head) -> str:
    out = [head, "|" + "|".join("---:" if c not in ("family", "name", "params") else "---" for c in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in ("roi_exp", "roi_conf", "lo_conf", "hi_conf", "roi_conf_pre"):
                cells.append(fmt(v))
            elif c in ("n_exp", "n_conf", "hits_exp", "hits_conf"):
                cells.append(f"{int(v):,}" if not np.isnan(v) else "—")
            elif c == "win_odds_med":
                cells.append("—" if np.isnan(v) else f"{v:.1f}倍")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_report(D, th, df, nd, null_sum):
    n_exp = int(D["explore"].sum()); n_conf = int((~D["explore"]).sum())
    fam = df.groupby("family").agg(n=("idx", "count"), cands=("candidate", "sum"), passed=("passed", "sum"),
                                   best_exp=("roi_exp", "max"), best_conf=("roi_conf", "max")).reset_index()
    cand = df[df["candidate"]].sort_values("roi_conf", ascending=False)
    passed = df[df["passed"]].sort_values("lo_conf", ascending=False)
    # 事後（post-hoc）: 確認期間だけ見た上位。選定の膨らみの実例として出す
    posthoc = df[df["n_conf"] >= MIN_N].sort_values("roi_conf", ascending=False).head(15)
    both = df[(df["n_exp"] >= MIN_N) & (df["n_conf"] >= MIN_N) & (df["roi_exp"] >= 1) & (df["roi_conf"] >= 1)]
    cols = ["family", "name", "params", "n_exp", "roi_exp", "n_conf", "hits_conf", "roi_conf", "lo_conf", "hi_conf", "win_odds_med", "roi_conf_pre"]
    head = "| 族 | 買い方 | 条件 | 探索n | 探索回収率 | 確認n | 確認的中 | 確認回収率 | 下限 | 上限 | 当たり目中央値 | 締切前換算 |"
    L = []
    L.append("# 3連単・約1,000パターン総当たり（2026-10-01）\n")
    L.append(f"データ: `research_data/bt2026_top20.npz`（2026年1〜8月・{D['n']:,}R、Model 1.0 上位20点の確率と確定3連単オッズ、"
             f"公式払戻）。探索 1〜5月 {n_exp:,}R ／ 確認 6〜8月 {n_conf:,}R。lab.db（46万R）はこの環境に無い。\n")
    L.append("**事前登録した手順**: 第1段＝探索で n≥200 かつ回収率≥100% を候補にする。第2段＝候補を確認期間で1回だけ評価し、"
             "確認の回収率≥100% かつ 95%区間下限>95% を合格とする。帰無対照＝「市場確率が真」で結果を20回作り直し同じ手順を通す。\n")
    L.append("## 1. 結果の要約\n")
    L.append(f"- パターン総数 **{len(df):,}**、候補（探索で100%超）**{int(df['candidate'].sum())}**、"
             f"**合格 {int(df['passed'].sum())}**。")
    L.append(f"- 探索・確認の両方で点推定が100%超（区間条件なし）: **{len(both)}** 本。")
    L.append(f"- 帰無20回: 候補 平均 {null_sum['cands'].mean():.1f} 本（最大 {int(null_sum['cands'].max())}）、"
             f"合格 平均 {null_sum['passes'].mean():.2f} 本（最大 {int(null_sum['passes'].max())}）、"
             f"確認期間の最高回収率 平均 {null_sum['best_conf'].mean()*100:.1f}%（最大 {null_sum['best_conf'].max()*100:.1f}%）。\n")
    L.append("## 2. 族ごとの集計\n")
    L.append("| 族 | 本数 | 候補 | 合格 | 探索の最高 | 確認の最高 |\n|---|---:|---:|---:|---:|---:|")
    for _, r in fam.iterrows():
        L.append(f"| {r['family']} | {int(r['n'])} | {int(r['cands'])} | {int(r['passed'])} | {fmt(r['best_exp'])} | {fmt(r['best_conf'])} |")
    L.append("\n族: A=モデル上位k点 / B=期待値選定 / C=市場人気帯 / D=固定の目（オッズ不要） / E=配分 / F=場・R番号 / G=曜日\n")
    L.append("## 3. 合格したパターン\n")
    L.append(table(passed, cols, head) if len(passed) else "**なし。**\n")
    L.append("\n## 4. 候補（探索で100%超）を確認期間に当てた結果（確認回収率順）\n")
    L.append(table(cand, cols, head) if len(cand) else "候補なし。\n")
    L.append("\n## 5. 事後に確認期間だけで並べた上位15（選定の膨らみの実例。採用根拠にはならない）\n")
    L.append(table(posthoc, cols, head))
    L.append("\n## 6. 帰無対照（市場確率が真の世界で同じ手順を20回）\n")
    L.append("| sim | 候補 | 合格 | 確認の最高回収率 |\n|---:|---:|---:|---:|")
    for _, r in null_sum.iterrows():
        L.append(f"| {int(r['sim'])} | {int(r['cands'])} | {int(r['passes'])} | {fmt(r['best_conf'])} |")
    L.append("\n帰無は族A/B/C/E/F/G（上位20点内の買い目）に対してのみ。族D（固定の目）は120通りの市場確率が無いので区間だけで判断する。\n")
    L.append("## 7. 読み方の注意\n")
    L.append("- 確定オッズでの数字。締切前オッズで買うと、当たった目は中央値で約9.4%下がる（9/16以降の本番実測・2〜30倍帯）。"
             "「締切前換算」列は ×0.906 の目安で、当たり目の中央値が30倍を超えるパターンには当てはまらない。")
    L.append("- 候補の多くは探索で100%を超えても確認で戻る（選定の膨らみ）。帰無でも同じ数だけ候補が出るなら、それは偶然。")
    L.append("- n が小さい区分（場別・R番号別・曜日）は区間が ±10pt 以上あり、点推定だけで判断してはいけない。")
    L.append(f"- 自信フィルタのしきい値（探索期間の分位点）: " + ", ".join(f"{k} q90={v[90]:.3f}" for k, v in th.items() if isinstance(v, dict)) + "。")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
