# 03 データベース設計（PostgreSQL）

方針：原本（raw）→ 正規化（core）→ 予想（pred, 追記専用）→ 採点（score）→ 集計（stats）の5層。すべての時刻は `timestamptz`（UTC保存・JST表示）。

## 1. ER 概要

```
stadiums ─┬─ meetings ─┬─ races ─┬─ entries ─(racer)─ racers ─ racer_periods
          │            │         ├─ race_conditions
          │            │         ├─ results ─ result_entries
          │            │         ├─ odds_snapshots
          │            │         └─ predictions ─┬─ prediction_selections
          │            │                         └─ scoring
model_versions ── predictions        settings_versions ── predictions
fetch_log / job_run / raw_files      metrics_* (集計)
```

## 2. DDL（要点）

```sql
-- ===== core =====
CREATE TABLE stadiums (
  code        smallint PRIMARY KEY,          -- 1..24（公式場コード）
  name        text NOT NULL,
  tz          text NOT NULL DEFAULT 'Asia/Tokyo'
);

CREATE TABLE racers (
  regno       integer PRIMARY KEY,           -- 登録番号
  name        text NOT NULL,
  branch      text, birth_date date, sex text
);

CREATE TABLE racer_periods (                 -- 期別成績（ファン手帳）
  regno       integer REFERENCES racers,
  period      text NOT NULL,                 -- 集計期 '2025H1' など
  published_at date NOT NULL,                -- 公式に配布された日（取得日で代用可）
  apply_from  date NOT NULL,                 -- この期の値を使ってよい最初のレース日（= published_at 以降）
  klass       text, win_rate numeric, rate2 numeric, rate3 numeric,
  avg_st numeric, starts int, wins int, seconds int, thirds int,
  f_count int, l_count int,
  course_stats jsonb,                        -- {course:{starts,win_rate,rate2,rate3,avg_st}}
  PRIMARY KEY (regno, period)
);

CREATE TABLE meetings (                      -- 開催節
  id          bigserial PRIMARY KEY,
  stadium_code smallint REFERENCES stadiums,
  title       text, grade text,              -- SG/G1/G2/G3/一般
  start_date  date NOT NULL, end_date date,
  UNIQUE (stadium_code, start_date)
);

CREATE TABLE races (
  id          bigint PRIMARY KEY,            -- YYYYMMDD*10000 + 場コード*100 + R（決定的ID）
  race_date   date NOT NULL,
  stadium_code smallint REFERENCES stadiums,
  race_no     smallint NOT NULL,
  meeting_id  bigint REFERENCES meetings,
  day_no      smallint,                      -- 節の何日目
  race_type   text,                          -- 予選/準優勝戦/優勝戦/特別選抜 等
  distance_m  smallint,
  post_time   timestamptz,                   -- 締切予定時刻（as-of 基準）
  is_fixed_entry boolean DEFAULT false,      -- 進入固定
  status      text NOT NULL DEFAULT 'scheduled', -- scheduled/finished/cancelled
  UNIQUE (race_date, stadium_code, race_no)
);

CREATE TABLE entries (                       -- 出走表（番組発表時点の値＝リークなし）
  race_id     bigint REFERENCES races,
  lane        smallint NOT NULL,             -- 艇番 1..6
  regno       integer REFERENCES racers,
  age smallint, weight numeric, branch text, klass text,
  nat_win_rate numeric, nat_rate2 numeric, nat_rate3 numeric,
  loc_win_rate numeric, loc_rate2 numeric, loc_rate3 numeric,
  motor_no smallint, motor_rate2 numeric, motor_rate3 numeric,
  boat_no  smallint, boat_rate2 numeric,  boat_rate3 numeric,
  series_results jsonb,                      -- 今節成績（番組表記載。初回取得値を固定し、以後上書きしない）
  program_fetched_at timestamptz,
  is_absent boolean DEFAULT false,           -- 欠場（事前判明分のみ）
  PRIMARY KEY (race_id, lane)
);

CREATE TABLE preview_snapshots (             -- 直前情報（追記専用。取得のたびに行を追加し、上書きしない）
  id bigserial PRIMARY KEY,
  race_id bigint REFERENCES races, lane smallint NOT NULL,
  fetched_at timestamptz NOT NULL, source text NOT NULL,   -- official_web / openapi / official_k(事後)
  exhibition_time numeric, tilt numeric,
  st_exh_course smallint, st_exh_time numeric,             -- スタート展示の進入・展示ST
  parts_replaced jsonb, pre_inspection_time numeric, weight_adjust numeric
);
CREATE INDEX ON preview_snapshots (race_id, fetched_at);

CREATE TABLE race_conditions (               -- 気象・水面（追記専用・時刻付き）
  id bigserial PRIMARY KEY,
  race_id bigint REFERENCES races, source text NOT NULL, observed_at timestamptz NOT NULL,
  weather text, temp_c numeric, water_temp_c numeric,
  wind_dir text, wind_speed_m numeric, wave_cm numeric
);
CREATE INDEX ON race_conditions (race_id, observed_at);

CREATE TABLE results (
  race_id bigint PRIMARY KEY REFERENCES races,
  trifecta text,                             -- '1-2-3'
  kimarite text,                             -- 決まり手
  trifecta_payout int, trifecta_popularity int,
  payouts jsonb,                             -- 全券種 {bet_type:[{combo,payout,popularity}]}
  is_irregular boolean DEFAULT false, irregular_note text, -- 特払い/返還/不成立
  source text, fetched_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE result_entries (
  race_id bigint, lane smallint,
  finish_pos smallint, course smallint, st numeric, race_time numeric,
  abnormal text,                             -- F/L/転/落/エ/失 等
  PRIMARY KEY (race_id, lane)
);

CREATE TABLE odds_snapshots (
  id bigserial PRIMARY KEY,
  race_id bigint REFERENCES races, bet_type text NOT NULL DEFAULT '3t',
  captured_at timestamptz NOT NULL, source text NOT NULL, -- official_web / csv_import / estimated
  odds jsonb NOT NULL                        -- {'1-2-3': 5.6, ...}
);
CREATE INDEX ON odds_snapshots (race_id, captured_at);

-- ===== model / settings =====
CREATE TABLE model_versions (
  version text PRIMARY KEY,                  -- '1.0', '1.1' ...
  created_at timestamptz DEFAULT now(),
  parent_version text, description text,
  feature_set_version text NOT NULL, selection_version text NOT NULL,
  params jsonb NOT NULL,                     -- 時間減衰・穴条件・閾値など
  artifact_path text, code_sha text,
  status text NOT NULL DEFAULT 'candidate',  -- candidate / active / shadow / retired
  backtest_summary jsonb
);

CREATE TABLE settings_versions (             -- 購入候補条件などの運用パラメータ（履歴付き）
  id serial PRIMARY KEY, effective_from timestamptz NOT NULL,
  confidence_min numeric NOT NULL DEFAULT 0.70,
  ev_min numeric NOT NULL DEFAULT 1.00,
  completeness_min numeric NOT NULL DEFAULT 0.6,
  points int NOT NULL DEFAULT 15, main_points int NOT NULL DEFAULT 10,
  stake_per_point int NOT NULL DEFAULT 200,
  readiness jsonb NOT NULL,                  -- 実戦投入条件 {min_races:1000, hit_rate:0.80, roi:1.00, recent100_hit:0.70, target_roi:1.10}
  extra jsonb
);
-- 注：穴最低オッズ・λ・減衰などモデル/選定ロジック固有の値は model_versions.params に置く（settings には置かない）。

-- ===== pred（追記専用） =====
CREATE TABLE predictions (
  id bigserial PRIMARY KEY,
  race_id bigint REFERENCES races NOT NULL,
  model_version text REFERENCES model_versions NOT NULL,
  settings_id int REFERENCES settings_versions NOT NULL,
  stage text NOT NULL,                       -- 'program'(暫定) / 'final'(確定)
  role  text NOT NULL,                       -- 'active'(採用モデル) / 'shadow'(比較用)
  created_at timestamptz NOT NULL DEFAULT now(),   -- トリガで強制
  asof_ts timestamptz NOT NULL,              -- 特徴量の時点
  post_time_at_pred timestamptz NOT NULL,    -- 予想時点の締切予定（後の締切変更に影響されない）
  features_used jsonb NOT NULL,              -- 実際に使った特徴量の値（再現性・監査用）
  preview_snapshot_ids jsonb, odds_snapshot_id bigint, condition_id bigint,
  completeness numeric NOT NULL,             -- 0..1
  missing_fields jsonb,
  flags jsonb,                               -- {low_agreement, low_sample, hole_relaxed, ...}
  boat_eval jsonb NOT NULL,                  -- 艇別評価（統計値と根拠）
  probs jsonb NOT NULL,                      -- 120通り 校正後確率
  odds_used jsonb NOT NULL,                  -- 120通り、{combo:{odds,source,snapshot_id}}
  ev jsonb NOT NULL,                         -- 120通り 期待値
  confidence numeric NOT NULL,
  expected_return numeric NOT NULL,          -- 15点の期待回収率
  decision text NOT NULL,                    -- 'buy' / 'skip'
  skip_reason text,
  rationale jsonb NOT NULL,                  -- 構造化根拠（統計値）
  rationale_text text,                       -- 補助文（LLM/テンプレ）
  input_hash text NOT NULL,                  -- 入力データのハッシュ（再現性）
  UNIQUE (race_id, model_version, stage, role)
);

CREATE TABLE prediction_selections (
  prediction_id bigint REFERENCES predictions,
  rank smallint, combo text NOT NULL, kind text NOT NULL,   -- 'main' / 'hole'
  stake int NOT NULL, prob numeric, odds_at_pred numeric, odds_source text, ev numeric,
  reason jsonb,
  PRIMARY KEY (prediction_id, combo)
);

-- 改変禁止トリガ
CREATE OR REPLACE FUNCTION forbid_change() RETURNS trigger AS $$
BEGIN RAISE EXCEPTION 'predictions are append-only'; END $$ LANGUAGE plpgsql;
CREATE TRIGGER predictions_immutable BEFORE UPDATE OR DELETE ON predictions
  FOR EACH ROW EXECUTE FUNCTION forbid_change();
CREATE TRIGGER selections_immutable BEFORE UPDATE OR DELETE ON prediction_selections
  FOR EACH ROW EXECUTE FUNCTION forbid_change();
CREATE OR REPLACE FUNCTION force_created_at() RETURNS trigger AS $$
BEGIN NEW.created_at := now(); RETURN NEW; END $$ LANGUAGE plpgsql;
CREATE TRIGGER predictions_created_at BEFORE INSERT ON predictions
  FOR EACH ROW EXECUTE FUNCTION force_created_at();

-- ===== score =====
CREATE TABLE scoring (
  prediction_id bigint PRIMARY KEY REFERENCES predictions,
  race_id bigint NOT NULL,
  scored_at timestamptz DEFAULT now(),
  valid boolean NOT NULL, invalid_reason text,     -- created_at >= post_time 等は無効
  actual_trifecta text, actual_payout int, actual_popularity int,
  hit boolean, hit_kind text,                      -- 'main' / 'hole' / NULL
  refunded_points int DEFAULT 0, refunded_stake int DEFAULT 0, -- F/L/欠場艇を含む買い目の返還
  stake_total int, payout_total int, pnl int, roi numeric, -- payout_total = 払戻(100円あたり) × stake/100
  category text,                                   -- buy_hit / buy_miss / skip_would_hit / skip_correct
  odds_final_ratio numeric                         -- 最終オッズ/予想時オッズ（的中買い目）
);
-- 計算式：stake_total = 有効買い目数×200（返還分を除外）、roi = payout_total / stake_total。
-- 累計・直近指標の母集団：role='active' AND stage='final' AND valid（見送りは category で別集計）。

CREATE TABLE backtest_runs (                 -- バックテストの予想・採点は predictions ではなくこちらに保存
  id bigserial PRIMARY KEY, model_version text, params jsonb, period_from date, period_to date,
  split text,                                -- 'validation' / 'test'
  metrics jsonb, created_at timestamptz DEFAULT now(), note text
);
CREATE TABLE backtest_predictions (          -- 明細（レースごとの15点・的中・払戻）
  run_id bigint REFERENCES backtest_runs, race_id bigint, payload jsonb, PRIMARY KEY (run_id, race_id)
);

-- ===== ops =====
CREATE TABLE raw_files (id bigserial PRIMARY KEY, source text, key text, sha256 text, fetched_at timestamptz, path text, UNIQUE(source,key));
CREATE TABLE fetch_log (id bigserial PRIMARY KEY, source text, key text, started_at timestamptz, duration_ms int, http_status int, ok boolean, retry_no int, error text);
CREATE TABLE job_run (id bigserial PRIMARY KEY, job text, started_at timestamptz, finished_at timestamptz, ok boolean, summary jsonb, error text);
```

## 3. 集計（stats）

* `v_scoring_active`：`role='active' AND stage='final' AND valid` の採点行。ヘッドライン指標はこのビューから。
* `metrics_daily(model_version, race_date, decision, kind, n, hits, stake, payout, ...)`：日次で再計算（マテリアライズドビュー）。
* 直近N（500/200/100/50）・最大連敗・最大ドローダウンは、`v_scoring_active` を時系列順に走査して計算（SQLウィンドウ関数またはPython）。
* 分析軸：場別・グレード別・オッズ帯別（予想時オッズ帯＝15点の平均オッズ帯／実結果オッズ帯＝当該レースの的中組み合わせの最終オッズ帯、すなわち荒れ度）・本線／穴別・見送り別・風速帯別・波高帯別・決まり手別・万舟率・場×風向×風速×波高・1号艇級別 等。
* 母集団の定義（実戦投入判定を含む）：**実時間で保存された active 予想（stage=final, valid）を、モデル版をまたいで累計**する（版の系譜 `parent_version` でつなぐ）。併せて版別・settings_id 別の内訳を表示し、現行版単独のサンプルが少ない場合は警告。バックテストの数値は母集団に含めない。

## 4. データ量見積り

* races：約25万行／5年。entries：約150万行。odds_snapshots：運用後1日約300行（jsonb 120要素）。predictions：1日約150〜300行（active＋shadow）。5年運用でも数GB以内。

## 5. 主要な整合性ルール

1. `results` は `races.post_time` 以降に取得したもののみ受理（`fetched_at > post_time`。過去一括取込は `source='official_k'` として例外的に許可し、予想には使わない）。
2. 直前情報・気象は追記専用スナップショット。特徴量は `fetched_at/observed_at <= asof_ts` の最新行だけを使い、使った行IDを `predictions` に記録する。
3. `predictions.stage='final'` は同一 `race_id × model_version × role` で1件のみ（UNIQUE）。修正したい場合は新しいモデル版として別行を追加する（過去行は残る）。`predictions` と `prediction_selections` は1トランザクションで書き、片方だけ残る状態を作らない。
4. 中止レース（`status='cancelled'`）の予想は `scoring.valid=false, invalid_reason='cancelled'`。
5. 自前集計特徴量の as-of 基準は**対象レースの `race_date` の前日まで**（`races.race_date < 対象.race_date` かつ結果あり）。同日の先行レース結果は運用時に間に合わないことが多く、バックテストとの乖離（train/serve skew）を生むため使わない。同日情報は番組表の「今節成績」（初回取得値）に限定する。`entries`（出走予定）にも同じフィルタを適用する。
6. 期別成績は `apply_from <= race_date` の期のみ結合する（集計期にレース日が含まれる期は使わない）。
7. 有効判定は `predictions.created_at < post_time_at_pred` かつ `created_at < races.post_time（最新）` の両方を満たすこと。
