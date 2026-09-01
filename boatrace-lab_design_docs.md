# Phase 0：データ取得元・利用規約 調査結果

調査日：2026-08-31

## 0. 結論サマリー

| 区分 | 取得元 | 取得手段 | 期間 | 主な項目 | 規約/条件 | 採用 |
|---|---|---|---|---|---|---|
| 公式・一次 | BOAT RACE公式「ダウンロード・他」 | 日次LZH（番組表B / 競走成績K）を1日1ファイルDL | 番組表・競走成績とも1997年頃〜現在（5年分は十分） | 出走表（級別・勝率・2連率・当地成績・モーター/ボート番号と2連率・今節成績）、結果（着順・進入・ST・展示タイム・レースタイム・決まり手・天候/風向/風速/波高・全券種払戻と人気） | 公式提供の配布ファイル。サイトポリシー上「大量アクセス等運営に支障を与える行為」禁止 → 1日1ファイルの低頻度DLで問題なし | ◎ 主データ |
| 公式・一次 | 公式「レーサー期別成績（ファン手帳データ）」 | 半期ごとのLZH | 2002年〜 | 選手の期別集計：勝率、2連率、3連率、平均ST、F/L数、コース別成績 等 | 同上 | ◎ 選手属性 |
| 非公式・無料 | Boatrace Open API（GitHub Pages, MITライセンス） | JSON（programs / previews / results, `YYYY/YYYYMMDD.json`, `today.json`） | programs v3：2023-05-01〜、results/previews：概ね同時期〜 | 出走表、直前情報（展示タイム・チルト・スタート展示進入/展示ST・部品交換・気温/水温/風/波）、結果 | MIT。約30分間隔更新、非公式で正確性無保証 | ○ 当日・直前情報の主経路 |
| 公式サイトHTML | boatrace.jp（`odds3t`, `beforeinfo`, `raceresult` 等） | HTML取得（低頻度・単一スレッド・間隔制御） | 当日〜直近のみ | 3連単全120通りオッズ、直前情報、結果 | robots.txtは制限なし。サイトポリシー：著作物の私的使用の範囲内、「大量アクセス・運営支障」禁止、営利目的リンク不可 | △ オッズ取得のみに限定して利用（後述） |
| 商用API | team-nave BRDB-API | HTTPS API | 過去データあり（範囲非公開） | 出走表・直前・オッズ・結果 | 3,300円/月、1日3クエリ・1回555件までと制約が強い | × 制約が強く不採用（将来候補） |
| 第三者サイト | kyoteiodds.com（時系列オッズ）、競艇オッズ保管庫（orangebuoy.net） | 閲覧/CSV？ | 不明 | 過去の時系列オッズ | 規約未確認・機械取得の可否不明 | △ 手動CSVインポート経路のみ用意 |
| 気象 | 気象庁 過去の気象データ / Open-Meteo | CSV / API | 長期 | 地点別の風・気温等 | 気象庁：出典明記で利用可。Open-Meteo：非商用無料 | △ 補助（水面気象は公式K/直前情報にあるため初期は不要） |

## 1. 公式ダウンロードデータ（主データ）

* ページ：https://www.boatrace.jp/owpc/pc/extra/data/download.html
* 実ファイル配布：`https://www1.mbrace.or.jp/od2/B/YYYYMM/bYYMMDD.lzh`（番組表）、`https://www1.mbrace.or.jp/od2/K/YYYYMM/kYYMMDD.lzh`（競走成績）。中身は Shift_JIS の固定書式テキスト（`BYYMMDD.TXT` / `KYYMMDD.TXT`）。公式に「レイアウト説明書」が提供されている。
* 番組表（B）の主な項目：日付・場・レース番号・レース名/種別・距離・締切予定時刻、各艇の 艇番／登番／氏名／年齢／支部／体重／級別／全国勝率／全国2連率／当地勝率／当地2連率／モーター番号・2連率／ボート番号・2連率／今節成績。
* 競走成績（K）の主な項目：レース見出し（天候・風向・風速・波高・決まり手）、各艇の 着順／艇番／登番／氏名／モーター／ボート／展示タイム／進入コース／ST／レースタイム、全券種の払戻金と人気（3連単の払戻・人気を含む）。
* 補足：公式サイト内の「番組表閲覧・オッズ・結果閲覧」の一部旧サービスは2025-03-05に終了しているが、ダウンロード配布は継続している（2026-08時点で2026年分まで掲載）。
* 5年分の規模：24場×約半数開催×12R ≒ 年間約5万レース、5年で約25万レース。1日1ファイル×2種×約1,800日 ≒ 3,600ファイル。初回取込は数時間規模、以後は日次1〜2ファイル。
* 解凍：LZH形式。Linuxでは `lhasa`（apt）または Python `lhafile` で展開可能。

**含まれない項目（重要）**：締切前の全通りオッズ、スタート展示の進入・展示ST、チルト、部品交換、前検タイム、選手3連対率（B。期別成績には有）、モーター3連対率、コース別成績（期別成績には有）。

## 2. レーサー期別成績（ファン手帳データ）

* `https://www.boatrace.jp/static_extra/pc_static/download/data/kibetsu/fan****.lzh`、前期/後期ごと、2002年〜。
* 選手ごとの期別集計。勝率・2連率・3連率・平均ST・出走回数・着回数・F数・L数・コース別成績など。

## 3. Boatrace Open API（非公式・MIT）

* `https://boatraceopenapi.github.io/programs/v3/YYYY/YYYYMMDD.json`（2023-05-01〜）、`.../previews/v3/...`、`.../results/v3/...`、および各 `today.json`。
* 約30分間隔更新。GitHub Actionsで公式サイトから生成している非公式プロジェクト。将来停止・仕様変更のリスクがあるため、**取得層でアダプタ化**し、停止時は公式ファイル＋自前取得に切替できるようにする。
* 直前情報（previews）は展示タイム・チルト・スタート展示（進入・展示ST）・部品交換・気温・水温・風・波を含む（現物JSONで項目名を実装時に確定する）。

## 4. 公式サイトHTML（オッズ取得のみ）

* 3連単オッズ：`https://www.boatrace.jp/owpc/pc/race/odds3t?rno={R}&jcd={場コード}&hd={YYYYMMDD}`（1ページに120通り）。
* 直前情報：`.../beforeinfo?...`、結果：`.../raceresult?...`（Open API停止時のフォールバック）。
* 規約観点：robots.txt は `Disallow:` なし。サイトポリシーは「不正アクセス、大量の情報送受信及び大量のアクセスなど、本サイトの運営に支障を与える行為」を禁止、コンテンツは私的使用の範囲内での利用。→ 本システムは**個人の私的利用**、**1リクエスト間隔3秒以上・単一スレッド・1レースあたり最大2〜3回**（直前情報公開後・締切数分前）に限定し、1日あたり数百リクエスト程度に抑える。取得したHTMLは再配布しない。
* 過去の締切前オッズは公式には配布されていない。→ バックテストでのオッズは「的中買い目の払戻（K）」で回収率を厳密に計算し、非的中買い目のオッズは**市場オッズ推定モデル**で近似する（04_model_design 参照）。ペーパートレード開始日から実オッズを蓄積し、実オッズでの再検証を行う。

## 5. その他

* team-nave BRDB-API：3,300円/月。1ライセンス1日3クエリ・1リクエスト555件の制約があり、日次全レース運用には不向き。将来の過去オッズ補完候補としてのみ記録。
* kyoteiodds.com / 競艇オッズ保管庫：時系列オッズを掲載。利用規約と機械取得可否は未確認のため自動取得はしない。ユーザーが手動で入手したCSVを取り込む「手動インポート経路」を用意する。
* 気象庁データ：公式K/直前情報に水面気象（風向・風速・波高・天候）が含まれるため、初期は外部気象を使わない。将来「予報を使った前日予想」を行う場合に追加。

## 6. 自動取得可否マトリクス（指示書7章の項目に対応）

| 指示書の項目 | 過去5年（バックテスト用） | 当日（予想用） | 備考 |
|---|---|---|---|
| 開催日・場・節・R・グレード・時刻 | ◎ B/K | ◎ Open API programs / B | |
| レース結果・進入・決まり手・3連単払戻・人気 | ◎ K | ◎ Open API results / K（翌日） | 予想保存後にのみ取得 |
| 3連単オッズ（120通り） | × 公式配布なし → 推定モデル＋手動CSV | ○ 公式HTML `odds3t`（低頻度） | 最大の制約事項 |
| 選手：名前・登番・級別・全国/当地勝率・2連率・年齢・体重・支部・今節成績 | ◎ B | ◎ programs | |
| 選手：3連対率・平均ST・F/L数・コース別成績 | ◎ 期別成績（半期粒度） | ◎ 期別成績＋自前集計 | 自前DBからレース単位で再集計する方が精度が高い |
| モーター/ボート番号・2連率 | ◎ B | ◎ programs | 3連率は自前集計 |
| 展示タイム | ◎ K（事後記録） | ◎ previews | K の展示タイムはレース前情報として扱える |
| スタート展示（進入・展示ST）・チルト・部品交換 | △ previews が存在する期間のみ（〜2023/2024以降） | ◎ previews | 旧期間は NULL 扱い |
| 前検タイム・整備情報 | × | △ 一部previews | 取得不可は NULL |
| 天候・気温・水温・風向・風速・波高 | ◎ K（天候・風・波）／気温水温は× | ◎ previews | |
| 決まり手・場別傾向 | ◎ K から集計 | — | |

## 7. リスクと対策

1. Open API の停止・仕様変更 → アダプタ層で吸収、公式ファイル＋自前HTML取得へフォールバック、欠損はNULL。
2. 公式配布ファイルの書式変更 → パーサをレイアウト説明書ベースで実装し、行パターン検証・単体テストを持つ。取込失敗はログ＋アラート。
3. オッズの過去データ欠如 → 推定モデルで近似しつつ、実オッズ蓄積を最優先で開始。推定オッズ由来の指標は画面上で「推定」と明示。
4. 規約・アクセス負荷 → レート制限・キャッシュ・リトライ上限を取得層で強制。公式HTML取得は「オッズ」と「フォールバック」に限定。

## 参考リンク

* 公式ダウンロード：https://www.boatrace.jp/owpc/pc/extra/data/download.html
* 公式サイトポリシー：https://www.boatrace.jp/owpc/pc/extra/policy.html
* Boatrace Open API：https://github.com/BoatraceOpenAPI/ （programs / previews / results）
* team-nave BRDB-API：https://www.team-nave.com/system/jp/products/brdbapi/
* kyoteiodds.com：https://kyoteiodds.com/
# 01 要件整理・不足要件・懸念事項

## 1. 確定済み要件（指示書から。再質問しない）

| # | 項目 | 内容 |
|---|---|---|
| R1 | 目的 | 研究・予想システム。データ収集→統計分析→モデル→バックテスト→当日予想→結果照合→成績蓄積→改善の継続 |
| R2 | リーク防止 | 結果を予想に使わない。予想保存後に結果取得。過去予想の修正禁止。バックテストで未来情報を使わない |
| R3 | 記録 | 予想・見送り・的中・払戻・投資・回収率をすべて自動記録。モデルバージョン管理と前後比較 |
| R4 | 実戦投入条件 | 累計1,000R以上／累計的中率80%以上／累計回収率100%以上／直近100R的中率70%以上／目標回収率110%以上。自動購入はしない。判定表示＋警告（サンプル不足・場偏り） |
| R5 | 対象 | 当日全レースを分析。購入候補＝信頼度70%以上かつ期待値1.0以上（設定変更可）。それ以外は見送り（仮想採点あり） |
| R6 | 買い目 | 3連単15点固定、1点200円、計3,000円。本線10点＋穴5点。資金配分最適化は将来拡張 |
| R7 | 穴条件 | 最低オッズを固定せず 10/15/20/25/30/40/50倍以上をバックテストで比較（穴的中率・穴回収率・全体的中率・全体回収率・平均払戻・最大連敗・最大DD・期待値・サンプル数）。過学習注意 |
| R8 | データ | 過去約5年。レース・選手・モーター/ボート・展示・気象。自動取得を目標、CSV/手動補完、取得層とロジックの分離 |
| R9 | 時間重み | 均等／年単位／半減期などを比較して選択 |
| R10 | モデル | 統計モデル→確率推定→オッズ・期待値評価→AI補助→最終15点。120通りの確率推定 |
| R11 | 期待値 | 推定確率×オッズを基礎。オッズ変動・誤差・控除率・同条件実績を考慮可能に |
| R12 | 信頼度 | 統計的に定義しドキュメント化（安定性、モデル間一致、類似条件実績、データ量、オッズ整合、校正精度） |
| R13 | 分析軸 | 場別（コース別1着率・決まり手率・平均配当・万舟率、場×風×波）、選手×コース×場×モーター×天候 |
| R14 | バックテスト | ウォークフォワード。学習／検証／テスト期間の分離 |
| R15 | 保存項目 | 予想生成時に120通りの確率・オッズ・期待値、15点、根拠、信頼度、判定などを保存 |
| R16 | 結果照合 | 結果・払戻・的中区分（本線／穴／見送りだが的中／見送り正解） |
| R17 | 成績 | 日次・週次・月次・累計、多数の指標。的中率は累計／直近500/200/100/50／場別／グレード別／オッズ帯別／穴本線別／見送り別 |
| R18 | 画面 | ダッシュボード、レース一覧、レース詳細、結果、成績分析、実戦投入判定。根拠の可視化（統計値必須） |
| R19 | 改善 | 当たりやすい条件／外れやすい条件の分析、外れレース重点分析、モデル比較→採用判断 |
| R20 | 校正 | 予測確率の校正検証と必要に応じた校正 |
| R21 | 指標優先 | 回収率＞損益＞的中率＞期待値＞最大DD＞最大連敗＞穴回収率＞直近成績 |
| R22 | 運用 | 日次自動処理（レース前／後）。エラーログ・リトライ・過剰アクセス防止。欠損はNULL＋警告 |
| R23 | モバイル | スマホ最優先のレスポンシブWeb＋PWA。iPhone/Android/PC。サーバー側で一元管理。通知・ネイティブ化は将来 |

## 2. 不足していた要件と本設計での決め方

| # | 不足事項 | 本設計での扱い |
|---|---|---|
| G1 | **稼働環境**（どこで毎日動かすか、月額コスト） | ユーザー判断が必要 → レビュー時に確認（02_architecture に選択肢） |
| G2 | **過去の締切前オッズが公式に存在しない** | 回収率は実払戻で厳密計算。穴判定・期待値の過去検証は「市場オッズ推定モデル」で近似し、実オッズは運用開始日から蓄積して再検証。方針の了承を確認 |
| G3 | 当日オッズの取得経路（公式HTMLのみ） | 私的利用・低頻度に限定して公式HTMLから取得。可否をユーザーに確認 |
| G4 | 「AIによる補助分析」の具体 | v1.0では**確率に影響させない**（根拠文生成と整合性チェックのみ、統計値は必ず併記）。LLM利用はAPIコストが発生するため要否を確認 |
| G5 | 予想確定タイミング | 「直前情報公開後〜締切約5分前」に最終予想を確定・保存（それ以前は暫定版として別バージョン保存）。保存済み予想は不変 |
| G6 | オッズ取得時刻の定義 | 予想確定時点のオッズを「予想時オッズ」、レース後の払戻から逆算した値を「最終オッズ」として両方保存 |
| G7 | 「的中」の定義 | 3連単の的中組み合わせが15点に含まれること（1点の的中）。回収率は払戻÷3,000円 |
| G8 | 「直近100レース」の母集団 | 既定＝購入候補レースのみ（見送りは別集計）。累計的中率も購入候補が母集団。全レース仮想成績は別指標として併記 |
| G9 | 1日に同一レースへ複数回予想を出した場合 | 「確定版」1件のみを成績に算入。暫定版は分析用に保持 |
| G10 | 中止・欠場・不成立・特払い・返還 | 成績算入ルールを定義（レース中止＝対象外、欠場艇を含む買い目＝返還扱い、特払い＝規定通り） |
| G11 | 級別・勝率が「今節開始時点」の値であること | Bファイル値をそのまま使う（当時点の値なのでリークなし）。自前集計特徴量は必ず「レース日の前日まで」のレースのみで計算（同日先行レースは使わない。運用とバックテストの条件を揃えるため）。期別成績は公表日以降のレースにのみ適用 |
| G12 | タイムゾーン | すべてJST基準（DBはUTC保存＋JST変換） |
| G13 | 認証 | 個人利用だがインターネット公開するため、簡易ログイン（単一ユーザー・長期セッション）を必須にする |

## 3. 統計的な懸念（正直に記載）

### 3.1 「的中率80%」と「回収率100%以上」の同時達成は極めて困難

3連単は120通り。15点は全体の12.5%。平均的なレース（1号艇1着率55%程度）では、完全に正しい確率を知っていたとしても「確率上位15点」に含まれる確率質量は概ね55〜60%（Harville型近似では約57%）にとどまります。1号艇1着率80%級の鉄板レースに絞ると15点で約79%まで上がりますが、そのようなレースは配当が低く、控除率（約25%）を考慮すると15点買いで回収率100%以上を維持するのは一段と難しくなります。

つまり、購入候補を強く絞れば的中率80%は近づく可能性がありますが、そのとき回収率100%以上・110%以上を同時に満たすのは統計的に非常に厳しい、というのが事前の見立てです。

**本設計での扱い**：条件値はすべて設定パラメータとし、指示書の値を初期値として採用します。システムは結果を美化せず、条件未達なら未達と表示します。加えて、判定画面に「サンプル数に基づく信頼区間（Wilson区間）」を併記し、達成が偶然でないかを判断できるようにします。レビュー時に「条件値をこのままにするか」を1回だけ確認します。

### 3.2 見送りの有効性検証

購入候補（信頼度70%・期待値1.0）で絞ったときに、絞らなかった場合より回収率が改善するか（選別効果）を常時比較表示します。絞り込み条件も過学習しやすいため、検証期間で決めテスト期間で確認します。

### 3.3 オッズの不確実性

締切前オッズは締切直前に大きく動くため、「予想時オッズ」で計算した期待値と「最終オッズ」による実績は乖離します。両方を保存し乖離を統計的に把握します。

### 3.5 「信頼度70%以上」と「期待値1.0以上」は互いに逆方向に働く

信頼度が高い（堅い）レースほど本線は人気側になり、控除率25%の市場では期待値は0.75付近に張り付きます。期待値1.0以上を主張するには「市場より33%高い確率」を推定できている必要があり、両条件を同時に満たす購入候補はかなり少なくなる見込みです（04 §7）。バックテストで購入候補の発生率を必ず報告し、代替ゲート案（期待値0.9以上、本線と穴の期待値を分ける等）も参考表示します。閾値は設定値なので運用しながら変更できます。

### 3.4 サンプル数

1,000レースは購入候補ベースで概ね2〜4か月分。穴（5点）の回収率は分散が極端に大きく、1,000レースでも十分に収束しません。穴に関する指標は必ずサンプル数と信頼区間を併記します。

## 4. ユーザー判断が必要な事項（レビュー時に確認）

1. 稼働環境と月額コストの許容（G1）
2. 公式サイトからの低頻度オッズ取得の可否（G3）と過去オッズ推定の方針（G2）
3. 実戦投入条件（的中率80%等）と購入候補条件（信頼度70%かつ期待値1.0）を初期値のままとするか（3.1・3.5）
4. LLM（生成AI）による根拠文生成を初期版に含めるか（G4・APIコスト）
# 02 システム構成設計

## 1. 全体像

```
┌──────────────────────── 取得層（差し替え可能なアダプタ群） ────────────────────────┐
│ OfficialDownload(B/K/期別)  OpenAPI(programs/previews/results)  OfficialWebOdds  CsvImport │
│    └ レート制限・キャッシュ・リトライ・取得ログ（fetch_log）を共通ミドルウェアで強制        │
└───────────────────────────────────┬────────────────────────────────────────────┘
                                    ▼  正規化レコード（Pydantic dataclass）
┌──────────────────────────── 保存層（PostgreSQL） ────────────────────────────┐
│ raw_*（取得原本）  →  正規化テーブル（races / entries / results / odds_snapshots …）│
│ predictions（追記専用・改変不可）   scoring（照合結果）   model_registry / metrics     │
└───────────────────────────────────┬────────────────────────────────────────────┘
                                    ▼
┌──────────── 特徴量層（as-of 時点固定） ────────────┐  ┌───────── モデル層 ─────────┐
│ FeatureBuilder(asof=レース締切時刻)                │→ │ 艇別強さ → 120通り確率      │
│  過去レースのみ集計・欠損はNULL・完全性スコア      │  │ 校正 → 市場オッズ推定       │
└──────────────────────────────────────────────────┘  │ 期待値・信頼度 → 15点選定    │
                                                      └────────────┬───────────────┘
        ┌────────────── バックテスト層 ─────────────┐               │
        │ ウォークフォワード／パラメータ探索／指標   │◄──────────────┘（同一コードを共用）
        └──────────────────────────────────────────┘
┌──────────── 運用層 ────────────┐   ┌──────────── API/UI層 ────────────┐
│ スケジューラ（日次・レース前後）│   │ FastAPI(JSON API) + PWA(React)     │
│ 予想確定 → 保存 → 結果 → 採点  │   │ 認証・ダッシュボード・分析画面       │
└────────────────────────────────┘   └──────────────────────────────────┘
```

**設計原則**

1. 取得層と予想ロジックの分離：モデルは正規化テーブルのみを見る。取得元の変更はアダプタ差し替えで吸収。
2. as-of（時点固定）原則：特徴量は「そのレースの締切時刻より前に確定していた情報」だけから計算する関数 `build_features(race_id, asof_ts)` に一本化し、当日予想もバックテストも**同じ関数**を使う。
3. 予想は追記専用：`predictions` は DB トリガで UPDATE/DELETE を禁止。結果は別テーブル。採点は結合ビューで導出。
4. すべてに版を付ける：モデル版・特徴量定義版・選定ロジック版・パラメータ版を予想レコードに刻む。
5. 欠損は欠損のまま：推測値で埋めない。完全性スコアを予想に添付し、UIで警告表示。

## 2. 技術スタック（提案）

| 層 | 採用 | 理由 |
|---|---|---|
| 言語 | Python 3.12 | 統計・機械学習ライブラリ、ユーザーのPython経験 |
| DB | PostgreSQL 16（開発時はSQLite互換を意識せず、Docker で Postgres を使う） | 追記専用トリガ・ウィンドウ関数・JSONB。将来の同時アクセス |
| ORM/移行 | SQLAlchemy 2 + Alembic | スキーマ版管理 |
| モデル | numpy / pandas / scikit-learn / LightGBM | 校正・GBDT・実績豊富 |
| API | FastAPI + Uvicorn | 軽量・型付き・自動ドキュメント |
| スケジューラ | APScheduler（専用ワーカープロセス1つ。API は別プロセスで複数ワーカー可）＋ 手動実行 CLI | 二重実行を防ぐ。将来 Celery 等へ拡張可 |
| フロント | React + TypeScript + Vite、PWA（vite-plugin-pwa / Workbox）、グラフは Chart.js | モバイルファースト、ホーム画面追加、キャッシュ戦略を制御しやすい |
| 配信/TLS | Caddy（自動HTTPS）で API と静的PWA を同一オリジン配信 | 設定が最小 |
| コンテナ | Docker Compose（app / db / caddy） | どの稼働環境でも同じ構成 |
| 通知（将来） | `notifier` インターフェース（LINE Messaging API / Web Push） | 差し替え可能 |

## 3. 稼働環境の選択肢（ユーザー判断事項 G1）

| 案 | 構成 | 月額目安 | 長所 | 短所 |
|---|---|---|---|---|
| A（推奨） | 小型VPS 1台（例：さくらVPS/ConoHa/Lightsail 1〜2GB）＋Docker Compose | 約700〜1,500円 | 常時稼働・スケジュール確実・スマホからHTTPSで常時アクセス・構成が最も単純 | 少額の固定費、初回のサーバー設定 |
| B | GitHub Actions（定期バッチ）＋ Supabase/Neon（Postgres無料枠）＋ Netlify（PWA） | 0円〜 | 固定費ゼロ、Netlify経験あり | Actions の cron は数分〜数十分遅延し「締切5分前に確定」が不安定。無料枠の実行時間（2,000分/月）を超えやすい。構成が分散し障害切り分けが難しい |
| C | 購入予定の低予算PC（常時稼働）＋ Cloudflare Tunnel/Tailscale でスマホ公開 | 電気代のみ | 固定費ほぼゼロ、資源に余裕 | 停電・再起動・回線に依存、自宅外からのアクセス設定が必要 |

いずれもコンテナ構成は同一。本文書は案Aを前提に書くが、B/Cでもコード変更は不要（スケジューラの起動方法のみ差分）。

## 4. モジュール構成（Pythonパッケージ）

```
boatrace-lab/
├── app/
│   ├── ingest/          # 取得層
│   │   ├── base.py          # Source インターフェース、RateLimiter、Retry、FetchLog
│   │   ├── official_files.py# B/K/期別 LZH → 正規化（Shift_JIS固定書式パーサ＋検証）
│   │   ├── openapi.py       # programs / previews / results JSON
│   │   ├── official_web.py  # odds3t（オッズ）、beforeinfo/raceresult（フォールバック）
│   │   └── csv_import.py    # 手動CSV（過去オッズ・展示など）
│   ├── store/           # SQLAlchemy モデル、リポジトリ、Alembic
│   ├── features/        # as-of 特徴量（選手・モーター・場・コース・気象・展示）
│   ├── model/
│   │   ├── strength.py      # 艇別・コース別 1着/2着/3着 強さモデル
│   │   ├── trifecta.py      # 120通り確率（条件付き順位モデル）
│   │   ├── calibration.py   # 確率校正（isotonic / Platt）と校正評価
│   │   ├── market.py        # 市場オッズ推定モデル（過去オッズ欠如の補完）
│   │   ├── confidence.py    # 信頼度
│   │   ├── ev.py            # 期待値・期待回収率
│   │   ├── selection.py     # 本線10点＋穴5点の選定
│   │   └── registry.py      # モデル版・パラメータ・採用状態
│   ├── backtest/        # ウォークフォワード、パラメータ探索、指標
│   ├── scoring/         # 予想と結果の照合、的中区分、損益
│   ├── analytics/       # 成績集計（期間別・場別・オッズ帯別…）、外れ分析
│   ├── ops/             # 日次ジョブ、スケジューラ、アラート、通知インターフェース
│   ├── api/             # FastAPI ルータ、認証
│   └── cli.py           # `lab ingest`, `lab backtest`, `lab predict`, `lab score` 等
├── web/                 # React PWA
├── docs/
├── tests/               # パーサ・as-of・リーク検査・採点の単体テスト
└── docker-compose.yml
```

## 5. データフローと時刻の定義

| 時刻 | 定義 | 用途 |
|---|---|---|
| `post_time` | 締切予定時刻（B/programs） | as-of 基準、予想確定期限 |
| `preview_snapshots.fetched_at` | 直前情報を取得した時刻（追記専用） | `<= asof_ts` の最新行のみ特徴量に使う |
| `predictions.created_at` | 予想レコード保存時刻（DBサーバー時刻、改変不可） | リーク検査：`created_at < post_time` を採点時に必ず検証 |
| `results.fetched_at` | 結果取得時刻 | `> post_time` を検証 |

**当日フロー**

1. 朝（開催日 07:30）：programs 取得 → `races/entries` 登録（今節成績は初回取得値を固定）→ 出走表ベースの「暫定予想（stage=program）」を生成・保存。
2. 各レースの締切約10分前：公式 `beforeinfo`（直前情報）を1回取得（Open API previews は約30分更新のため補助扱い）→ 締切約6分前に `odds3t` を1回取得 → 取得完了イベントで「確定予想（stage=final）」を生成・保存（イベント駆動。締切5分前を過ぎても未取得なら、その時点の情報で欠損警告付きで確定）。
3. レース後（締切+約15分）：results 取得 → `results/payouts` 保存 → 採点 → 成績集計更新。
4. 夜（23:30）：公式 K/B ファイルで当日分を照合・補完（Open API との差異検査。K の展示タイム・気象は `source='official_k'` の事後スナップショットとして別行で保存し、当日予想には使わない）。
5. 週次：モデル再学習候補の作成とバックテスト比較（採用は手動承認）。

**リーク防止の技術的担保**

* `predictions` の INSERT 時トリガで `created_at` を `now()` に強制、UPDATE/DELETE を拒否。
* `scoring` ジョブは `prediction.created_at < post_time_at_pred` かつ `< races.post_time` かつ `results.fetched_at > post_time` の場合のみ算入。違反は `invalid_reason` を付けて集計から除外し、アラート。
* 特徴量ビルダーは「対象レースの `race_date` より前のレース」の `results/entries` しか読まないよう、リポジトリ層で強制フィルタ（未来行を注入しても出力が変わらないことを単体テストで検証）。
* 直前情報・気象・オッズは `fetched_at <= asof_ts` のスナップショットだけを使い、使用した行IDと特徴量値そのものを `predictions.features_used` に保存（再現性・監査）。
* バックテストは `predictions` に書かず `backtest_runs / backtest_predictions` に書く。レースごとに同じビルダーを `asof_ts = post_time` で呼び、学習データは `train_end < 検証開始` を強制。
* 進入コースは学習・推論とも「スタート展示の進入（無ければ艇番）」で統一し、実進入（結果）は特徴量に使わない。

## 6. セキュリティ・運用

* 単一ユーザー認証（パスワード＋長期セッションCookie、HTTPSのみ）。API はすべて認証必須。
* シークレット（DB・LLM API キー等）は環境変数。
* バックアップ：日次 `pg_dump` を保存（VPS内＋任意でオブジェクトストレージ）。
* ログ：取得ログ（fetch_log）、ジョブログ（job_run）、アプリログ（JSON）。
* 監視：ジョブ失敗・取得失敗率・予想未確定レース数をダッシュボードに表示（通知は将来）。
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
# 04 予想モデル設計（確率推定・校正・期待値・信頼度・15点選定）

## 1. パイプライン

```
特徴量(as-of) → [A] 艇別強さモデル → [B] 120通り確率 → [C] 校正 → [D] 市場オッズ(実 or 推定)
             → [E] 期待値 → [F] 信頼度 → [G] 15点選定(本線10+穴5) → [H] 購入候補/見送り判定 → 保存
```

すべての段はバージョン付きの純関数（入力：レース＋asof、出力：JSON）。当日予想とバックテストで同一コードを使う。

## 2. 特徴量セット v1（すべて asof_ts より前の情報のみ）

| 群 | 特徴量 | 出典 |
|---|---|---|
| レース | 場、R番号、レース種別（予選/準優/優勝戦）、グレード、節の日数、距離、進入固定、月、風向（8方位→場ごとの追い/向かい/横に変換）、風速、波高、天候、気温、水温 | B/K/previews |
| 選手（番組表値） | 級別、全国勝率/2連率/3連率、当地勝率/2連率/3連率、年齢、体重、支部、今節成績（平均着順・出走数・当節ST平均） | B/programs |
| 選手（自前集計・縮約付き） | 直近1年/3年の コース別1着率・2連率・3連率・平均ST、当該場×コース成績、場×風条件別成績、F/L回数、直近10走の平均着順、直近90日の出走数 | 自前DB |
| モーター/ボート | 番組表の2連率・3連率、当該モーターの当節・当期の自前集計（1着率・2連率・展示タイム順位平均）、ボート2連率 | B/自前DB |
| 展示（直前） | 展示タイム、レース内順位、レース平均との差、チルト、スタート展示の進入コース、展示ST、展示STのレース内順位、部品交換有無、前検タイム | previews（旧期間はK展示タイムのみ） |
| 相対 | 各数値の「レース内順位」「レース平均との差」「1号艇との差」 | 派生 |
| 進入予測 | 予想進入コース＝スタート展示の進入（無ければ艇番）。進入コースの各特徴量は予測コースで引く | 派生 |

縮約（Empirical Bayes）：率 = (成功数 + k·事前率) / (試行数 + k)、事前率＝場×コースの基準率、k は検証期間で 10〜50 から選択。少サンプル選手の暴れを防ぐ。

欠損：NULLのまま（LightGBM は欠損を扱える）。`completeness` = 重要特徴量の充足率（展示・気象・出走表の重み付き）。

## 3. [A] 艇別強さモデル

* 目的変数：艇ごとの 1着（binary）、2着以内、3着以内。
* 手法：LightGBM（クラス不均衡は重み調整）、ハイパーパラメータは粗いグリッドで検証期間により選択。
* ベースライン（必ず比較）：M0 = 場×コース基準率のみ、M0b = 番組表の勝率のみのロジスティック回帰。新モデルはこれらを検証期間で有意に上回ることが採用の前提。
* 出力：艇 i の強さスコア s_i（1着確率の校正前推定）と、2着・3着向けスコア。

## 4. [B] 120通り確率

**Model 1.0（初期）**：位置別割引付き Plackett–Luce

P(a→b→c) = s_a/Σs · s_b^λ₂/Σ_{j≠a}s_j^λ₂ · s_c^λ₃/Σ_{j≠a,b}s_j^λ₃

λ₂, λ₃ ∈ (0,1] は検証期間で3連単対数尤度を最大化して決める（Harville は λ=1 で人気側を過大評価する既知の傾向があるため）。

**Model 1.1（次段）**：条件付き順位モデル。「1着が a のとき b が2着」を、a・b の特徴と相互作用（コース差・ST差・展示差）で直接学習し、PL の 2着・3着項を置換。両者を対数線形で混合（重み検証期間決定）。

**共通制約**：120通りの確率は合計1に正規化。欠場艇を含む組み合わせは0。

## 5. [C] 校正

* 対象：①各組み合わせ確率 p_i（校正後にレース内で再正規化し合計1にする）、②「15点セットの的中確率」S = Σ_{i∈15} p_i。
* 学習データ：ウォークフォワードの各ステップで、学習窓の末尾3か月をホールドアウトして校正器を学習する（LightGBM の学習データそのものには当てない。検証期間で閾値を決めるため、校正器を検証期間で学習しない）。
* 手法：isotonic regression（単調）または Platt。折り返しで安定性確認。校正はパラメータ探索の最終段で1回学習し、探索途中は未校正の log-loss で比較する。
* 評価：信頼性曲線（10分位）、Brier、log-loss、ECE。**予測70%帯の実績が70%±誤差に収まるか**を毎月自動チェックし、乖離時は再校正候補を作成（採用は手動）。

## 6. [D] 市場オッズ

* 運用時：`odds_snapshots`（公式HTML）の予想確定時点の実オッズを使用。取得できなければ推定オッズ＋`odds_source='estimated'`。
* バックテスト時：過去の全通りオッズは存在しないため **市場オッズ推定モデル** を使う。
  * 公開情報のみ（級別・勝率・コース・場基準率・モーター2連率・展示）から「市場が付けるであろう確率 q_i」を PL 型で推定し、oddŝ_i = 0.75 / q_i を基礎とする。
  * 補正には K ファイルにある**全券種の払戻**（単勝・複勝・2連単・2連複・拡連複・3連複・3連単＝1レース約10個の観測価格）を使い、各券種の払戻が q から導かれる価格と整合するよう対数線形でフィットする。3連単の的中組み合わせだけを使う場合の選択バイアスを軽減する。
  * 精度は「観測価格 vs 推定価格」の相関・平均誤差で報告。運用開始後は実スナップショットで全120通りを検証し、推定モデルを更新。
* 回収率の計算は常に**実際の払戻金**を使う（推定オッズは穴判定・期待値のみに影響）。

## 7. [E] 期待値

* 買い目：EV_i = p̂_i × odds_i × d(odds_i)
  * p̂_i：校正後確率を市場確率 q_i 側へ縮約した値 p̂ = (1−β)·p + β·q（β は検証期間で決定、初期 0.3）。自モデルと市場の乖離が最大＝推定誤差が最大の買い目を選ばないための保守化（winner's curse 対策）。さらに複数seedモデルの分位点で下側推定を取る。
  * d(odds)：オッズ変動補正。初期値 1.0。実オッズ蓄積後は**全120通り**の「−15分→−6分」および「−6分→最終」の比のオッズ帯別中央値で推定する（的中買い目だけから推定すると、締切直前に資金が集まる側へ偏るため使わない）。
  * 控除率は市場オッズに内包済み。
* レース：期待回収率 ER = Σ_{i∈15} stake_i·EV_i / Σ stake_i。「期待値1.0以上」＝ ER ≥ 1.0（設定値）。
* 補助：同条件（場×風速帯×1号艇級別 等）での過去の実回収率も参考値として保存・表示。

**構造的な注意（レビュー確認事項）**：推定オッズは 0.75/q なので、ER ≥ 1.0 は「15点平均で市場より33%高い確率を主張している」状態を意味する。本線10点は人気側で EV≈0.75 付近に張り付きやすく、信頼度が高い（堅い）レースほど ER は 1.0 を下回る。したがって **「信頼度70%以上 かつ 期待値1.0以上」を同時に満たす購入候補は非常に少なくなる**見込みで、1,000レース到達に長期間を要する可能性がある。バックテストでは購入候補の発生率を必ず報告し、代替ゲート（例：ER ≥ 0.9、本線ER と穴ER の分離、信頼度と期待値のどちらかを主とする案）も参考値として並記する。閾値は設定値なので、実績を見て変更できる。

## 8. [F] 信頼度（定義）

信頼度 C は「この15点セットが的中する確率」の**校正済み推定値そのもの**とする。

C = Cal(S)、S = Σ_{i∈15} p_i

* Cal(S)：S を校正ホールドアウトの実績（セット的中／不的中）で校正した値。**C=70% は「同程度の予想が長期的に約70%当たる」ことを意味する。** ペナルティ係数を乗じると校正の意味が壊れるため乗じない。
* 不確実性は別の**ゲート／フラグ**として扱い、`predictions.flags` に記録する：
  * `completeness < completeness_min`（展示・気象など重要データ未取得）→ 見送り（skip_reason='incomplete'）
  * `low_agreement`：複数seed／ブートストラップ間の S の標準偏差が閾値超 → 見送り
  * `low_sample`：主要選手の当該場出走数が閾値未満 → フラグ表示（見送りにはしない、分析軸）
  * `odds_estimated`：実オッズ未取得 → フラグ表示
* 「類似条件での過去実績」と「オッズとの整合（p と q の乖離）」は信頼度の式には入れず、根拠表示と分析軸（外れ分析）で扱う。式に入れると校正の対象が二重になるため。
* 校正チェック：C∈[0.70,0.75) のレースの実セット的中率を毎月レポート。

「AIが自信がある」といった主観値は一切使わない。

## 9. [G] 15点選定

**本線10点**：校正後確率 p_i 上位10（オプション：p_i^(1−α)·EV_i^α、α は 0〜0.3 で検証。初期 α=0）。

**穴5点**：本線以外で以下を満たす候補から、**EV の信頼下限**（p̂ の下側分位点 × odds × d）降順に5点。
* odds_i ≥ hole_min_odds（バックテストで 10/15/20/25/30/40/50 を比較して決定。過学習防止のため検証期間で選び、テスト期間で確認。上位案が同程度なら中庸の値を採用）
* p̂_i ≥ p_min（例 0.3%。ゼロ確率の宝くじ買いを排除）
* 分散制約：同一1着艇の穴は最大2点、1号艇1着以外の穴を2点以上（レース荒れ度に応じて可変）。
* 候補が5点に満たない場合：閾値を1段下げて補充し、`hole_relaxed=true` を記録（穴5点は必ず確保）。穴オッズ比較表では relaxed 分を分離集計する。

**荒れ度**：1−p(1号艇1着) と p 分布のエントロピーを「荒れ度」として保存し、本線/穴の性格を分析軸に使う。

## 10. [H] 購入候補判定

decision = 'buy' ⇔ C ≥ confidence_min（0.70）AND ER ≥ ev_min（1.0）AND completeness ≥ 最低値 AND レースが中止でない。
それ以外は 'skip'（`skip_reason` を記録）。skip でも15点は生成・保存し、仮想採点する。

## 11. 時間減衰

学習サンプル重み w(t) = 0.5^{Δ年/h}。候補：均等（h=∞）／年単位ステップ／h=3,2,1,0.5年。検証期間 log-loss と ROI で比較し、差が1標準誤差以内なら単純な方（均等または長い h）を採用。

## 12. AI（LLM）補助分析

* v1.0：確率・選定には**影響させない**。役割は (a) 構造化根拠（統計値）を読みやすい文章に整える、(b) 入力データの異常（出走表と直前情報の不整合など）のフラグ出し。
* 統計値の表示は必須で、LLM文は任意。LLM が使えない／未設定なら定型テンプレート文で代替。
* 将来：LLMの出力を特徴量として加える場合も、必ず別モデル版としてバックテストで有意性を確認してから採用。

## 13. モデル版管理と昇格ルール

1. 変更は必ず新版（`model_versions`）。パラメータ・特徴量版・選定版・コードSHAを記録。
2. 候補版は同じ検証・テスト期間でバックテストし、現行版と比較する。**採用判断の指標順は指示書 R21（回収率＞損益＞的中率＞期待値＞最大DD＞最大連敗＞穴回収率＞直近成績）**に従い、すべて信頼区間付きで表示する。ハイパーパラメータ探索の主指標が log-loss なのは「確率の正しさが回収率の前提」であり、払戻の分散が大きい回収率では少数の万舟に引きずられて選択が不安定になるため。
3. 採用前に **shadow 運用**（role='shadow' で毎日予想保存・採点、最低300レース）を行う。300レースでは回収率の差は検出できない（区間が±数十%）ため、shadow の合否は log-loss・セット的中率・校正誤差で判定し、回収率は参考表示とする。
4. 採用（active 化）はユーザーの明示操作。旧版は retired としてデータを保持。ロールバック可能。
5. 運用中の active 版は固定パラメータのまま**月次で再学習**（学習窓の前進のみ。パラメータ変更は新版扱い）。バックテストが「常に前月末まで学習」した成績であることと整合させる。再学習ごとに `model_versions.artifact_path` を更新し、予想には学習時点（`trained_until`）を記録する。

## 15. 直前情報の期間差への対処

previews（展示ST・進入・チルト・部品交換）は 2023-05 以降しか存在しない。学習期間の大半で欠損する特徴量を LightGBM に入れると欠損パターン自体を学習してしまうため、
* **Model 1.0**：直前情報のうち展示タイム（K に全期間あり）のみ使用。
* **Model 1.1**：展示ST・進入・チルト・部品交換を追加し、学習期間を 2023-05 以降に限定した版と比較。
* バックテストでは運用時の直前情報取得率（実測）に合わせて特徴量をランダムに欠損させ、運用と同じ完全性分布で評価する（skew 補正）。
* バックテストで「欠場」は締切前に知り得ないため、欠場艇を含む買い目は返還として扱う（確率を0にしない）。K の風速・波高はレース時観測であり previews（−10分）と僅差だが未来情報であることを skew として記録する。

## 14. 過学習対策（まとめ）

* 学習／検証／テストの分離（05 参照）。テスト期間の評価回数を記録し、テストでの再調整は禁止。
* グリッドは粗く（穴オッズ7値、減衰6値、縮約k 3値、λ 0.1刻み）。
* 検証で僅差なら単純な設定を選ぶ。
* 指標には常にサンプル数と信頼区間（的中率：Wilson、回収率：ブートストラップ）を添える。
* 場別・条件別の「最高成績探し」を評価に使わない（分析表示のみ）。
# 05 バックテスト設計

## 1. 期間分割（データ：2021-09 〜 2026-08 の約5年）

| 区分 | 期間 | 用途 |
|---|---|---|
| 学習（初期） | 2021-09 〜 2023-12 | モデル学習 |
| 検証 | 2024-01 〜 2024-12 | ハイパーパラメータ・穴オッズ・時間減衰・閾値・校正の決定 |
| テスト（封印） | 2025-01 〜 2026-08 | 最終評価のみ。モデル版ごとに評価回数を記録し、テスト結果で再調整しない |

期間は設定ファイルで定義し、運用が進むごとに前方へスライドする。スライドして旧テスト期間が学習・検証側に入っても、**既に評価に使った旧テスト期間をパラメータ再探索の検証データとして使わない**（学習にのみ使う）。テスト期間の評価はモデル版ごとに1回だけ（Phase 5 で実施。Phase 6 はその結果をレポート化するのみで再評価しない）。

## 2. ウォークフォワード

```
for 月 m in 検証(またはテスト)期間:
    学習データ = m の前月末までの全レース（拡張ウィンドウ、時間減衰重み付き）
    学習窓の末尾3か月 = 校正ホールドアウト（LightGBM には使わず、校正器と市場オッズ推定器の補正に使う）
    モデル学習（LightGBM + PL λ）→ 校正器・市場オッズ推定器をホールドアウトで学習
    for レース r in 月 m（日付・締切順）:
        features = build_features(r, asof=r.post_time)     # 当日予想と同一関数
        pred = predict(features)                             # 120通り確率〜15点〜判定
        record(pred)                                         # 予想を先に記録
    結果照合・採点（月末に一括。採点は record 後）
```

* 学習データの `train_end < m の初日` を機械的に検証（assert）。
* `build_features` の as-of フィルタは単体テストで「未来レースが混入しない」ことを保証（意図的に未来行を注入し、結果が変わらないことを確認するテスト）。
* 校正器・市場オッズ推定器も同じウォークフォワードで学習（未来の払戻を使わない）。

## 3. 評価指標（各期間・各モデル版・各設定で算出）

**回収率系**：回収率（実払戻／投資）、損益、平均払戻、期待値（予測）と実現値の差、ブートストラップ95%区間。
**的中系**：的中率（Wilson 95%区間）、本線的中率、穴的中率、セット的中率の校正誤差。
**リスク系**：最大連敗、最大ドローダウン（累積損益の最大落込み）、月次回収率の標準偏差、最悪月。
**選別効果**：全レース仮想 vs 購入候補 vs 見送り の各指標、見送りだが的中の件数。
**モデル品質**：120通り log-loss、1着 log-loss、Brier、ECE、ベースライン M0/M0b に対する改善幅。
**分布**：場別・グレード別・オッズ帯別・風速帯別・穴/本線別（表示用。最適化には使わない）。

## 4. パラメータ探索（検証期間のみ）

| パラメータ | 候補 | 選定基準 |
|---|---|---|
| 穴最低オッズ | 10/15/20/25/30/40/50 | 比較表の列＝穴的中率・穴回収率・全体的中率・全体回収率・平均払戻・最大連敗・最大DD・期待値・サンプル数（relaxed 分は分離）。主指標は全体回収率、僅差なら中庸 |
| 時間減衰 h | 均等／年ステップ／3y/2y/1y/0.5y | log-loss（主）、回収率 |
| 縮約 k | 10/25/50 | log-loss |
| PL λ₂, λ₃ | 0.3〜1.0（0.1刻み） | 3連単 log-loss |
| 本線EV混合 α | 0/0.1/0.2/0.3 | 回収率、的中率 |
| 判定閾値 | 信頼度 0.60〜0.80、ER 0.9〜1.2 | 指示書初期値を既定。代替案は参考表示 |

探索は「粗いグリッド→検証で選定→テストで1回確認」。組み合わせ爆発を避け、逐次（減衰→縮約→λ→穴→α→β）で決める。探索途中は未校正確率の log-loss（穴・α・β は回収率）で比較し、校正器は最終段で1回学習してから閾値（信頼度0.70・ER1.0）の意味を確認する。
併せて、購入候補（信頼度≥0.70 かつ ER≥1.0）の**発生率**と、代替ゲート案の成績を参考表示する（04 §7 の構造的注意）。

## 5. 過去オッズ欠如への対処

* 回収率・損益は実払戻で厳密。
* 穴判定・期待値は推定オッズ（04 §6）。バックテスト結果には「推定オッズ依存」のラベルを付け、実オッズ蓄積（運用開始後）で再評価する二段構え。
* 感度分析：推定オッズを ±20% 変動させたときの穴選定と回収率の変化を報告（頑健性の確認）。

## 6. 出力

* `backtest_runs`（run_id, model_version, params, period, metrics jsonb, created_at）に保存。
* レポート：CLI で Markdown/HTML を生成し、UI「モデル比較」画面で閲覧。
* 予想単位の明細（レースごとの15点・的中・払戻）も保存し、外れ分析に利用。

## 7. ペーパートレードとの接続

ペーパートレード（Phase 7）は「テスト期間の前方延長」。同じ採点コードで、`predictions.created_at < post_time` を満たす実時間予想のみを算入する。バックテストの数値と実運用の数値を並べて表示し、乖離（例：オッズ変動・取得欠損）を監視する。
# 06 UI設計（スマートフォン最優先・PWA）

## 1. 方針

* モバイルファースト（幅360px基準）、下部タブナビ（ホーム／レース／成績／モデル／設定）、片手操作を想定しタップ領域は44px以上。
* 重要情報→詳細情報の順に縦に積み、統計詳細は「展開」で表示。
* 数値は色で意味を持たせる（推奨＝アクセント、見送り＝グレー、的中＝緑、外れ＝赤）。色以外にもラベル文字を必ず付ける。
* データはすべてサーバーAPIから取得。端末には表示キャッシュのみ。

## 2. 画面一覧

### 2.1 ホーム（ダッシュボード）
* 今日の日付／開催場数／レース数／予想対象数／購入候補数／見送り数。
* 「今日の勝負レース」カード（購入候補を信頼度×期待値の順に最大5件。場・R・締切時刻・信頼度・期待回収率・ステータス）。
* 本日の仮想収支（投資／払戻／損益／回収率、進行中は暫定表示）。
* 累計成績（的中率／回収率／損益、購入候補ベース）と直近100レース（的中率／回収率）。
* 実戦投入判定の要約（クリア／未達と不足量）。
* データ取得の健全性（直前情報未取得のレース数、最終取得時刻、エラー件数）。

### 2.2 本日のレース一覧
* フィルタ：推奨のみ／全件、場、ステータス（予想前／暫定／確定／結果待ち／確定済）。
* 行：場・R・締切時刻・信頼度バー・期待回収率・推奨/見送りバッジ・結果（的中🎯／外れ／未）。
* 締切順にソート、「次の締切」までのカウントダウン。

### 2.3 レース詳細
* ヘッダ：場・R・種別・締切・気象（天候/風向/風速/波高/水温）・データ完全性の警告（「一部データ未取得」）。
* 判定カード：信頼度、期待回収率、推奨/見送り（理由）。
* 買い目カード：本線10点／穴5点、各200円、合計3,000円。タップで展開→推定確率・予想時オッズ（実/推定の別）・期待値・穴判定理由・主要根拠。
* 6艇カード：選手名・級別・勝率・当地・平均ST・モーター・展示タイム（順位）・展示ST・進入・チルト・部品交換・自前集計（場×コース1着率など）。
* 「120通りを見る」：確率／オッズ／期待値の表（ソート可、横スクロール）。
* 根拠：構造化根拠（統計値の箇条）＋総合コメント（テンプレ/LLM）。暫定版と確定版の差分も閲覧可。
* 結果セクション（確定後）：結果・払戻・的中区分・投資・損益・回収率。

### 2.4 結果画面（レース後カード）
* 結果 1-3-2／予想15点内なら「🎯 的中（本線）」等、投資3,000円／払戻／損益／回収率。
* 見送りレースは「見送り：仮に購入していれば ○○」を表示。

### 2.5 成績ダッシュボード
* 期間切替：日／週／月／累計、直近 50/100/200/500。
* KPI：予想数・購入候補数・見送り数・的中数・的中率（区間付き）・投資・払戻・損益・回収率（区間付き）・平均配当・平均オッズ・本線/穴的中率・穴回収率・最大連敗・最大DD。
* グラフ：累積損益曲線、月別回収率、場別回収率、オッズ帯別的中率と回収率、校正曲線（予測確率 vs 実績）。
* 分析タブ：場別／グレード別／オッズ帯別／本線・穴別／見送り別／風速帯別／1号艇級別。
* 外れ分析：外れレースの一覧と傾向（荒れ度・風・場・1号艇級別）。

### 2.6 実戦投入判定
指示書の書式をそのまま採用（クリア／未達と不足量）。加えて：サンプル数に基づく信頼区間、場別偏り警告（特定場の寄与が損益の50%超）、直近成績悪化警告、校正乖離警告、推定オッズ依存の割合。
「最終的な購入判断はユーザーが行ってください」を常時表示。自動購入機能は持たない。

### 2.7 モデル
* モデル版一覧（active/shadow/candidate/retired）、各版の検証・テスト・shadow 成績比較表。
* 「採用」ボタン（確認ダイアログ付き）。ロールバック。
* バックテスト実行履歴と結果レポート。

### 2.8 設定
* 購入候補条件（信頼度・期待値）、穴条件、1日の上限（将来）、通知（将来）、データ取得の状態とログ、手動CSVインポート、手動再取得ボタン。

## 3. PWA

* `manifest.json`（名称・アイコン・スタンドアロン表示・テーマ色）、Service Worker（Workbox）。
* キャッシュ戦略：静的資産＝Cache First（バージョン付き）。API＝Network First、失敗時のみ古いキャッシュを「取得時刻付き」で表示（古い予想・オッズを最新として見せない）。予想確定後のレース詳細は Stale-While-Revalidate 可。
* オフライン：最後に取得したホーム／レース一覧／成績を「オフライン表示（HH:MM時点）」バッジ付きで表示。
* iOS：ホーム画面追加案内、`apple-touch-icon`、スプラッシュ用メタ。
* 通知：Web Push（将来）に備え Service Worker に受信ハンドラの拡張点を用意。LINE通知は `notifier` 実装で追加。

## 4. API（主要エンドポイント）

`GET /api/today`、`GET /api/races?date=`、`GET /api/races/{id}`（予想・120通り・根拠・結果）、`GET /api/stats?range=`、`GET /api/stats/breakdown?by=`、`GET /api/readiness`（実戦投入判定）、`GET /api/models`、`POST /api/models/{v}/activate`、`GET /api/backtests`、`GET /api/settings`、`PUT /api/settings`、`POST /api/import/csv`、`GET /api/health`。
# 07 日次自動処理・エラー対策・欠損処理

## 1. スケジュール（JST）

| 時刻 | ジョブ | 内容 | 失敗時 |
|---|---|---|---|
| 07:30 | `ingest_programs` | Open API programs（today.json）→ races/entries。失敗時は公式Bファイル（前日夜に配布される当日分）を代替 | 3回リトライ後アラート。出走表なしのレースは予想対象外 |
| 07:45 | `predict_program` | 全レースの暫定予想（stage=program）を生成・保存 | — |
| 各R 締切−10分 | `ingest_beforeinfo` | 公式 `beforeinfo`（直前情報）を1回取得 → `preview_snapshots`/`race_conditions` に追記。Open API previews（約30分更新）は補助で、取得できていれば先に使う | 取得不可は NULL で続行 |
| 各R 締切−6分 | `ingest_odds` | `odds3t` を1回取得 → `odds_snapshots`（将来 −15分の2回目を追加してオッズ変動 d(odds) を推定） | 取得不可は推定オッズ |
| 取得完了イベント（遅くとも締切−5分） | `predict_final` | 確定予想（stage=final）を生成・保存（active＋shadow、1トランザクション）。以後このレースの予想は変更不可 | 直前情報なしでも確定（警告付き）。`now() >= post_time` なら生成せず `skipped_late` |
| 各R 締切+15分〜 | `ingest_results` | results（today.json）→ results/result_entries → 採点 → 集計更新 | 未確定なら15分ごと再試行（最大6回） |
| 23:30 | `ingest_official_k_b` | 公式K/Bファイルで当日分を照合（結果・払戻・展示・気象の差異検査、欠損補完） | 差異はレポート |
| 日曜 03:00 | `weekly_model_check` | 校正チェック、候補モデルの再学習・バックテスト、shadow 成績集計 | レポートのみ |

スケジューラは APScheduler（アプリ内）。各ジョブは冪等（同じ入力で再実行しても二重登録しない）で、`job_run` に記録。手動実行は CLI/設定画面から。

## 2. 取得層の共通制御

* レート制限：ソースごとに最小間隔（公式HTML 3秒、Open API 1秒、公式ファイル 5秒）、同時1接続。
* リトライ：一時的エラー（タイムアウト・5xx・接続失敗）は指数バックオフで最大3回。4xx（404等）はリトライしない。
* キャッシュ：同一キーの再取得は ETag/更新時刻または内容ハッシュで抑止。previews/results の today.json は1回の取得で全レース分を得て差分更新。
* ログ：`fetch_log`（ソース・キー・時刻・所要・HTTPステータス・リトライ回数・エラー）。日次の失敗率をダッシュボードに表示。
* 過剰アクセス防止：公式HTMLは1レースあたり基本2リクエスト（beforeinfo・odds3t）。24場開催日でも約290R×2＝約580。リトライ込みの1日上限（既定 1,000）をハードリミットとし、超過時は推定オッズ／NULLにフォールバック。同時刻に多数の締切が重なる場合は締切の早い順に処理し、間に合わないレースは前倒し取得（−12分）で吸収する。

## 3. 欠損データの扱い

* 取得できない値は NULL。推測値で埋めない。
* 予想には `completeness` と `missing_fields` を保存。UIは「一部データ未取得（展示タイム・水温）」のように具体的に表示。
* 展示未取得：`w_complete` で信頼度を下げる。オッズ未取得：推定オッズを使用し `odds_source='estimated'` を表示。期待値は「推定」ラベル。
* 出走表未取得：そのレースは予想対象外（一覧に「未取得」として表示）。

## 4. 予想確定タイミングの厳格化

* `predict_final` は締切時刻の直前（既定5分前）に実行。実行時に `now() < post_time` を確認し、超過していれば確定予想を生成しない（`skipped_late` として記録）。
* 締切時刻の変更（遅延）を programs 再取得で検出した場合は `post_time` を更新するが、既に保存済みの予想はそのまま（`post_time_at_pred` と更新後 `post_time` の両方に対して `created_at` が前であることを採点時に確認）。programs 再取得で `entries.series_results`（今節成績）は上書きしない（初回取得値固定）。

## 5. 中止・返還

* レース中止：`status='cancelled'`、採点 `valid=false`。
* 返還：欠場艇・フライング（F）・出遅れ（L）の艇を含む買い目は公式ルール通り返還。`scoring.refunded_points/refunded_stake` に記録し、投資から除外して回収率を計算（指示書の3,000円固定は「有効買い目×200円」に読み替え。要件G10）。バックテストでは K の返還情報・`result_entries.abnormal` から機械的に判定。
* 特払い・不成立：公式払戻に従い記録、`is_irregular=true`。
* 払戻の単位：公式払戻は100円あたり。払戻額 = 払戻 × (200 / 100)。
* `skipped_late`（確定予想を生成できなかったレース）は成績母集団から除外し、件数を監視項目として表示する。

## 6. 監視項目（ホームに表示）

* 本日：出走表取得済レース数／直前情報取得率／オッズ取得率／確定予想数／結果取得数／採点済数。
* 直近7日：取得失敗率、ジョブ失敗数、`skipped_late` 件数、推定オッズ依存率。
# 08 開発ロードマップ

| Phase | 内容 | 完了条件（DoD） | 目安 |
|---|---|---|---|
| 0 | データ取得元・規約調査 | 00 文書 | 完了 |
| 1 | 設計（本文書群）とユーザーレビュー | レビュー承認、判断事項の確定 | 今回 |
| 2 | データ基盤：Docker構成、DBスキーマ、公式B/K/期別パーサ、Open API アダプタ、公式オッズ取得、CSVインポート、取得ログ | サンプル日（数日分）の取込がテスト付きで通る。5年分の一括取込を実行し件数検証（欠損率レポート） | 2〜3週 |
| 3 | 統計分析：場別・コース別・決まり手・配当分布・選手/モーター集計、as-of 特徴量ビルダー、リーク検査テスト | 分析ノート（Markdown）と UI の分析タブ雛形。as-of テストが通る | 1〜2週 |
| 4 | 予想モデル：M0/M0b ベースライン、LightGBM 強さモデル、PL-λ、校正、市場オッズ推定、信頼度、期待値 | 検証期間で M0/M0b を上回る。校正レポート | 2〜3週 |
| 5 | 15点選定：本線10+穴5、穴オッズ探索、時間減衰比較、閾値、判定 | 探索レポート（検証期間）。Model 1.0 の封印テスト評価（**この1回のみ**） | 1〜2週 |
| 6 | バックテスト基盤の整備：指標・モデル比較画面・`backtest_runs`・レポート生成 | Phase 5 の結果をレポート化（テストの再評価はしない）。実戦条件・購入候補発生率との比較表示 | 1〜2週 |
| 7 | ペーパートレード：日次自動処理、予想保存→結果照合→採点、PWA（ホーム／一覧／詳細／結果／成績／判定） | 連続7日間の無人稼働。取得率・確定率の監視 | 2〜3週 |
| 8 | 継続運用・改善：外れ分析、校正監視、候補モデル shadow、昇格フロー、通知拡張点 | 週次レポート自動生成、Model 1.1 の shadow 比較 | 継続 |

優先順位：リーク防止と記録の正しさ ＞ 取得の安定性 ＞ モデル精度 ＞ UI の見栄え。

将来拡張（構造のみ用意）：資金配分最適化（`stake` は既に買い目単位）、1日上限、場別専用モデル（`model_versions.params.scope`）、荒れ度予測、穴専用モデル、アンサンブル、自動校正、通知（notifier）、勝負レースランキング、自動レポート、長期収支シミュレーション、ネイティブアプリ（APIはそのまま利用可）。
