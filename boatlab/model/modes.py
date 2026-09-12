"""3モード表示（2026-09-12 設計、`reports/backtest/two_modes.md` / `condition_rules.md` / `no_hit_loss.md`）。

  ana     : 3連単（穴狙い）   市場が示す万舟確率が高いレースで、人気20〜40番目の21点を100円ずつ
  katai   : 3連単（堅い予想） 本線（モデル確率順）を「当たれば必ず投資額以上」が成り立つ点数まで削って買う
  fukusho : 複勝・単勝        市場の確信度がしきい値以上のレースで1点

設計上の約束:
  - 選定はすべて **市場のオッズ**（katai の本線の並びだけモデル確率）。モデルの期待値は使わない
    （`market_combine.md`: モデルが期待値1.0超えと見た買い目の実回収率は69.8%）。
  - どのモードも回収率100%には届かない。画面には実測回収率・月の期待損失・最長連敗を必ず併記する。
  - しきい値は確定オッズで決めた値。締切前オッズでは市場の確信度が中央値で約6pt低く出るので、
    記録が貯まったら再較正する（設定は自動で変えない。versioned settings で人が変える）。

modes_version = "modes1"
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from boatlab.model.trifecta import PERMS

MODES_VERSION = "modes1"
UNIT = 100
Q_MAN = 0.0075                                  # オッズ100倍 ⟺ 万舟
_A = np.array([p[0] for p in PERMS])
_B = np.array([p[1] for p in PERMS])

# 実測（確定オッズ・2026年確認期間）。画面の併記用。数字の出典は各 md。
MEASURED = {
    "ana": dict(roi=0.833, roi_lo=0.75, roi_hi=0.92, hit=0.233, avg_payout=7495, max_lose=26,
                per_day=14.0, stake_per_race=2100, source="condition_rules.md §4"),
    "katai": dict(roi=0.814, roi_lo=0.79, roi_hi=0.84, hit=0.662, avg_payout=3360, max_lose=None,
                  per_day=4.5, stake_per_race=2841, loss_on_hit=0.010, source="no_hit_loss.md"),
    "fukusho": dict(roi=0.993, roi_lo=0.98, roi_hi=1.01, hit=0.946, avg_payout=105, max_lose=2,
                    per_day=10.6, stake_per_race=100, source="condition_rules.md §3"),
    "tansho": dict(roi=0.948, roi_lo=0.93, roi_hi=0.96, hit=0.850, avg_payout=112, max_lose=None,
                   per_day=15.0, stake_per_race=100, source="two_modes.md §D"),
}


@dataclass
class ModeParams:
    ana_enabled: bool = True
    ana_qman_min: float = 0.2495        # 市場が示す万舟確率の上位10%（探索期間の分位点）
    ana_rank_lo: int = 20               # 人気20番目から（1起点・含む）
    ana_rank_hi: int = 40               # 人気40番目まで（含む）
    ana_stake: int = 100
    katai_enabled: bool = True
    katai_budget: int = 3000            # 1レース予算＝保証する最低払戻
    katai_points_max: int = 10
    katai_points_min: int = 3           # これ未満しか成立しないなら「堅い予想」ではないので見送り
    katai_confidence_min: float = 0.70  # 本体の購入判定と同じ
    fukusho_enabled: bool = True
    fukusho_q_min: float = 0.90         # 市場の2着以内確率
    tansho_enabled: bool = True
    tansho_q_min: float = 0.7709        # 市場の1着確率の上位10%
    place_stake: int = 100

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict | None) -> "ModeParams":
        d = d or {}
        return ModeParams(**{k: v for k, v in d.items() if k in ModeParams().__dict__})


def market_probs(odds3t: np.ndarray) -> np.ndarray | None:
    """3連単オッズ(120) → 市場確率(120)。100通り未満なら None。"""
    o = np.asarray(odds3t, float)
    inv = np.where(np.isfinite(o) & (o > 0), 1.0 / o, 0.0)
    if (inv > 0).sum() < 100:
        return None
    return inv / inv.sum()


def market_summary(q: np.ndarray) -> dict:
    """市場が示す万舟確率・1着確率・2着以内確率。"""
    win = np.array([q[_A == a].sum() for a in range(6)])
    plc = np.array([q[(_A == a) | (_B == a)].sum() for a in range(6)])
    return dict(q_man=float(np.where(q <= Q_MAN, q, 0.0).sum()),
                q1=win.tolist(), q2=plc.tolist(),
                q1_max=float(win.max()), q1_arg=int(win.argmax()),
                q2_max=float(plc.max()), q2_arg=int(plc.argmax()))


# ---------------------------------------------------------------- 穴
def select_ana(odds3t: np.ndarray, prm: ModeParams) -> dict:
    """市場が示す万舟確率が高いレースで、人気 lo〜hi 番目を等額。"""
    q = market_probs(odds3t)
    if q is None or not prm.ana_enabled:
        return dict(fired=False, reason="odds_missing" if q is None else "disabled", points=[], stakes=[])
    ms = market_summary(q)
    order = np.argsort(-q)
    pts = [int(i) for i in order[prm.ana_rank_lo - 1: prm.ana_rank_hi]]
    fired = ms["q_man"] >= prm.ana_qman_min
    # 見送りでも「買うならこれ」を返す。旧モードと同じく、見送りレースも仮想採点して条件の良し悪しを測る
    return dict(fired=bool(fired), reason=(None if fired else "q_man_low"), q_man=ms["q_man"],
                points=pts, stakes=[prm.ana_stake] * len(pts),
                odds=[float(odds3t[i]) for i in pts], q=[float(q[i]) for i in pts])


# ---------------------------------------------------------------- 堅い
def guaranteed_stakes(odds: np.ndarray, budget: int) -> tuple[int, np.ndarray | None]:
    """後ろから削りながら Σstake ≤ budget かつ 全点の払戻 ≥ budget を満たす最大の点数。
    各点 ceil(budget / odds) を100円単位で切り上げ。成立しなければ (0, None)。"""
    odds = np.asarray(odds, float)
    for k in range(len(odds), 0, -1):
        o = odds[:k]
        if not (np.isfinite(o).all() and (o > 0).all()):
            continue
        st = np.ceil(budget / o / UNIT) * UNIT
        if st.sum() <= budget:
            return k, st.astype(int)
    return 0, None


def select_katai(main_idx: list[int], odds3t: np.ndarray, confidence: float, prm: ModeParams) -> dict:
    """本線（モデル確率順）を保証つき配分で。信頼度が足りなければ発火しない。"""
    if not prm.katai_enabled:
        return dict(fired=False, reason="disabled", points=[], stakes=[])
    main = [int(i) for i in main_idx][: prm.katai_points_max]
    odds = np.array([odds3t[i] for i in main], float)
    ok = np.isfinite(odds) & (odds > 0)
    main = [m for m, o in zip(main, ok) if o]
    odds = odds[ok]
    if not main:
        return dict(fired=False, reason="odds_missing", points=[], stakes=[])
    k, st = guaranteed_stakes(odds, prm.katai_budget)
    if k < prm.katai_points_min:
        return dict(fired=False, reason="too_few_points" if k else "no_guarantee", points=[], stakes=[], k=k)
    fired = confidence >= prm.katai_confidence_min
    return dict(fired=bool(fired), reason=(None if fired else "confidence_low"), points=main[:k],
                stakes=[int(x) for x in st], odds=[float(o) for o in odds[:k]],
                min_payout=int(min(st[i] * odds[i] for i in range(k))), stake_total=int(st.sum()))


# ---------------------------------------------------------------- 複勝・単勝
def select_place(odds3t: np.ndarray, prm: ModeParams) -> dict:
    """複勝・単勝それぞれ、市場の確信度がしきい値以上なら市場がいちばん確信している艇を1点。"""
    q = market_probs(odds3t)
    if q is None:
        return dict(fukusho=dict(fired=False, reason="odds_missing"), tansho=dict(fired=False, reason="odds_missing"))
    ms = market_summary(q)
    fk = dict(fired=bool(prm.fukusho_enabled and ms["q2_max"] >= prm.fukusho_q_min),
              lane=ms["q2_arg"] + 1, q=ms["q2_max"], stake=prm.place_stake,
              reason=None if prm.fukusho_enabled and ms["q2_max"] >= prm.fukusho_q_min else "q_low")
    tn = dict(fired=bool(prm.tansho_enabled and ms["q1_max"] >= prm.tansho_q_min),
              lane=ms["q1_arg"] + 1, q=ms["q1_max"], stake=prm.place_stake,
              reason=None if prm.tansho_enabled and ms["q1_max"] >= prm.tansho_q_min else "q_low")
    return dict(fukusho=fk, tansho=tn, q_man=ms["q_man"])


# ---------------------------------------------------------------- 荒れ理由タグ・堅い注意タグ（2026-09-12、ana_playbook.md）
import re as _re

ROUGH_STADIUMS = {4, 14, 2, 3}                       # 平和島・鳴門・戸田・江戸川（全期間で万舟率 ×1.10〜1.18）
_KIKAKU = _re.compile(r"モーニング|ツッキー|ガチ勝|ピンクル|サンライズ|進入固定|シャイニング")
_WOMEN_ROOKIE = _re.compile(r"女子|ルーキー|レディース|新人")
TAG_NAMES = {
    "l1_b": "1号艇がB級", "kado": "4カド条件（4号艇が3号艇より勝率高くST速い）", "l1_ext4": "1号艇の展示タイム4位以下",
    "rough_stadium": "荒れる場（平和島・鳴門・戸田・江戸川）", "maezuke": "展示で前づけあり", "top1_12": "1番人気12倍以上",
    "women_rookie": "女子戦・ルーキー戦（通説と逆で堅い）", "kikaku": "企画レース（1号艇に強い選手・堅い）",
}
# 前向き検証の対象（事前登録）。合格基準: 発火300R以上で タグ付き回収率 > タグ無し かつ 差5pt以上
PREREGISTERED = ("l1_b", "kado", "l1_ext4")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def race_tags(boat_eval: dict | None, stadium_code: int | None, title: str | None, race_type: str | None,
              q: np.ndarray | None) -> dict:
    """荒れ理由タグ（rough）と堅い注意タグ（solid）。**選定には使わない。** 表示と前向き検証のため。"""
    be = boat_eval or {}
    b = {int(k): v for k, v in be.items() if str(k).isdigit()}
    rough, solid = [], []
    l1 = b.get(1) or {}
    if str(l1.get("klass") or "") in ("B1", "B2"):
        rough.append("l1_b")
    n3, n4 = _f((b.get(3) or {}).get("nat_win_rate")), _f((b.get(4) or {}).get("nat_win_rate"))
    s3, s4 = _f((b.get(3) or {}).get("avg_st")), _f((b.get(4) or {}).get("avg_st"))
    if None not in (n3, n4, s3, s4) and n4 > n3 and s4 < s3:
        rough.append("kado")
    r1 = _f(l1.get("exhibition_rank"))
    if r1 is not None and r1 >= 4:
        rough.append("l1_ext4")
    if stadium_code in ROUGH_STADIUMS:
        rough.append("rough_stadium")
    if any((_f(v.get("course_pred")) or 99) < ln for ln, v in b.items()):
        rough.append("maezuke")
    if q is not None and len(q) and q.max() > 0 and 0.75 / q.max() >= 12:
        rough.append("top1_12")
    text = f"{title or ''} {race_type or ''}"
    if _WOMEN_ROOKIE.search(text):
        solid.append("women_rookie")
    if _KIKAKU.search(text):
        solid.append("kikaku")
    return dict(rough=rough, solid=solid)
