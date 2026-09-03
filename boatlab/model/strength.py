"""[A] 艇別強さモデル：LightGBM で 1着 / 2着以内 / 3着以内 を学習する（docs/04 §3）。

- 時間減衰重み w = 0.5^(Δ年/h)。h=None で均等。
- ベースライン M0（場×コース基準率）と M0b（番組表勝率ロジスティック）も同じインターフェースで提供。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from boatlab.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES, feature_matrix

TARGETS = ("y_win", "y_top2", "y_top3")

DEFAULT_LGB_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, num_threads=2, seed=7,
)


def time_decay_weights(dates: pd.Series, asof: pd.Timestamp, half_life_years: float | None) -> np.ndarray:
    if not half_life_years:
        return np.ones(len(dates))
    dy = (asof - pd.to_datetime(dates)).dt.days.values / 365.25
    return np.power(0.5, np.clip(dy, 0, None) / half_life_years)


@dataclass
class StrengthModel:
    """3つの二値モデル（win/top2/top3）を束ねる。seeds を複数指定すると seed アンサンブル
    （各ターゲットについて seed ごとに学習し、確率の幾何平均を取る。Model 1.1）。
    boosters のキーは "y_win#7" のように target#seed。"""
    boosters: dict[str, lgb.Booster] = field(default_factory=dict)
    num_rounds: int = 400
    params: dict = field(default_factory=lambda: dict(DEFAULT_LGB_PARAMS))
    half_life_years: float | None = 2.0
    seeds: tuple[int, ...] = (7,)
    feature_names: list[str] = field(default_factory=lambda: NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    def _split(self):
        cat = [c for c in self.feature_names if c in CATEGORICAL_FEATURES]
        num = [c for c in self.feature_names if c not in CATEGORICAL_FEATURES]
        return num, cat

    def fit(self, x: pd.DataFrame, asof: pd.Timestamp, valid: pd.DataFrame | None = None) -> "StrengthModel":
        num, cat = self._split()
        m = feature_matrix(x, num, cat)
        w = time_decay_weights(x["race_date"], asof, self.half_life_years)
        for t in TARGETS:
            ok = x[t].notna() & x["finish_pos"].notna()
            ds = lgb.Dataset(m[ok], label=x.loc[ok, t].values, weight=w[ok.values],
                             categorical_feature=cat, free_raw_data=False)
            kwargs = {}
            if valid is not None and len(valid):
                mv = feature_matrix(valid, num, cat)
                okv = valid[t].notna() & valid["finish_pos"].notna()
                dv = lgb.Dataset(mv[okv], label=valid.loc[okv, t].values, reference=ds,
                                 categorical_feature=cat)
                kwargs = dict(valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
            for sd in self.seeds:
                params = {**self.params, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd}
                self.boosters[f"{t}#{sd}"] = lgb.train(params, ds, num_boost_round=self.num_rounds, **kwargs)
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        # 学習時の特徴量名を booster から復元（旧モデルの後方互換も担保）
        names = None
        for b in self.boosters.values():
            names = b.feature_name()
            break
        if names is None:
            names = list(getattr(self, "feature_names", NUMERIC_FEATURES + CATEGORICAL_FEATURES))
        num = [c for c in names if c not in CATEGORICAL_FEATURES]
        cat = [c for c in names if c in CATEGORICAL_FEATURES]
        m = feature_matrix(x, num, cat)
        out = pd.DataFrame(index=x.index)
        for t in TARGETS:
            preds = [b.predict(m, num_iteration=b.best_iteration or None)
                     for k, b in self.boosters.items() if k.split("#")[0] == t]
            if not preds:
                continue
            g = np.exp(np.mean([np.log(np.clip(p, 1e-9, 1)) for p in preds], axis=0))  # 幾何平均
            out[t.replace("y_", "p_")] = g
        return out

    def feature_importance(self) -> pd.DataFrame:
        rows = []
        for k, b in self.boosters.items():
            imp = b.feature_importance(importance_type="gain")
            rows.append(pd.Series(imp, index=b.feature_name(), name=k))
        df = pd.concat(rows, axis=1)
        return df.sort_values(df.columns[0], ascending=False)


class BaselineCourseRate:
    """M0：場×コース（予想進入）の直近1年の基準率だけを使う。"""

    def fit(self, x, asof=None, valid=None):
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "p_win": x["sc_1y_win_rate"].fillna(x["g_course_win_rate"]).fillna(1 / 6).values,
            "p_top2": x["sc_1y_top2_rate"].fillna(x["g_course_top2_rate"]).fillna(1 / 3).values,
            "p_top3": x["sc_1y_top3_rate"].fillna(x["g_course_top3_rate"]).fillna(1 / 2).values,
        }, index=x.index)


class BaselineProgramLogit:
    """M0b：番組表の公開値（勝率・級別・コース・モーター）だけのロジスティック回帰。"""
    COLS = ["course_pred", "klass_ord", "nat_win_rate", "loc_win_rate", "motor_rate2", "avg_st", "sc_1y_win_rate",
            "nat_win_rate_dmean", "avg_st_dmin"]

    def __init__(self):
        self.models: dict[str, LogisticRegression] = {}
        self.fill: pd.Series | None = None

    def _x(self, x):
        m = x[self.COLS].astype(float)
        if self.fill is None:
            self.fill = m.median()
        m = m.fillna(self.fill)
        return pd.get_dummies(m, columns=["course_pred"], dtype=float).reindex(
            columns=self.columns_ if hasattr(self, "columns_") else None, fill_value=0.0)

    def fit(self, x, asof=None, valid=None):
        m = x[self.COLS].astype(float)
        self.fill = m.median()
        m = pd.get_dummies(m.fillna(self.fill), columns=["course_pred"], dtype=float)
        self.columns_ = list(m.columns)
        for t in TARGETS:
            ok = x[t].notna() & x["finish_pos"].notna()
            self.models[t] = LogisticRegression(max_iter=500, C=1.0).fit(m[ok], x.loc[ok, t])
        return self

    def predict(self, x):
        m = self._x(x)
        return pd.DataFrame({t.replace("y_", "p_"): mdl.predict_proba(m)[:, 1] for t, mdl in self.models.items()},
                            index=x.index)
