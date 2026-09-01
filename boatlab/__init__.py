"""boatlab — ボートレース研究・予想システム。

パッケージ構成（docs/02_architecture.md 参照）:
  ingest   取得層（外部ソース → 正規化レコード）
  store    保存層（SQLAlchemy モデル・書き込み）
  features 特徴量層（as-of 固定）
  model    予想モデル
  backtest バックテスト
  scoring  採点
  analytics 成績集計
  ops      日次ジョブ
  api      FastAPI
"""

__version__ = "0.1.0"
