# momentum_screener

日本株の初動候補を抽出し、20営業日以内に `+15%` へ先に到達し、`-10%` へ先に到達しない確率をニューラルネットで推定する個人用スクリーニングツールです。

投資助言ではありません。検証と運用補助のためのツールです。

## よく使う入口

最新候補を表示:

```bat
scripts\screen_latest.bat
```

株価データを更新してから表示:

```bat
scripts\update_and_screen.bat
```

モデルを再学習:

```bat
scripts\train_current.bat
```

銘柄ごとの過去シグナルを確認:

```bat
scripts\inspect_symbol_history.bat 285A
```

ローリング評価:

```bat
experiments\evaluation\rolling_evaluation.bat
```

リスク調整プリセットの比較:

```powershell
& '.venv\Scripts\python.exe' experiments\optimization\evaluate_risk_presets.py
```

## 主要ファイル

- `src/momentum_screener/data.py`: yfinance からの取得、OHLCV キャッシュ更新
- `src/momentum_screener/features.py`: 特徴量、初動ゲート、ラベル、出力列
- `src/momentum_screener/model.py`: NN、学習、評価、リスク調整スコア
- `src/momentum_screener/cli.py`: `train`, `screen`, `refresh-data`, `rolling-eval` などのCLI
- `src/momentum_screener/settings.py`: `config/screening_settings.json` を読み込む共通設定
- `scripts/screen_or_show.py`: 日常表示用の入口。CSVが新しければ再推論せず表示だけ行う
- `scripts/show_candidates.py`: 最新候補と直近履歴をコンソール表示
- `scripts/inspect_symbol_history.py`: 1銘柄の過去シグナルとその後の値動きを表示
- `experiments/optimization/evaluate_risk_presets.py`: 学習を増やさずリスク調整だけ比較
- `experiments/optimization/scoring_design_notes.txt`: 今後の目的関数・評価指標・スコア設計メモ

## 現在の学習仕様

- 初動候補日は、同一銘柄で20営業日のクールダウンを置いて学習に使う
- 推論時はクールダウン前の `raw_initial_momentum` もスコアリングする
- ラベルは `barrier`
- エントリー価格は候補日の翌営業日始値
- 20営業日以内に `+15%` に到達し、かつ `-10%` 到達より先なら正例
- 同じ日に利確ラインと損切りラインへ到達した場合は保守的に失敗扱い
- 学習設定
  - dropout: `0.15 / 0.10`
  - batch size: `256`
  - learning rate: `1e-3`
  - best epoch: validation loss

## 現在のスコア仕様

モデルの出力は `follow_through_prob` です。表示やランキングでは、リスク調整後の `final_score` を使います。

```text
final_score = follow_through_prob + upside_bonus - risk_penalty
```

デフォルトのリスク調整は `volatility_only` です。過去の探索では、純粋な上昇率だけなら `none` も強い一方、下落を抑える目的では `volatility_only` が扱いやすい結果でした。

## 出力

- `outputs/candidates_current.csv`: 最新候補
- `outputs/candidates_recent.csv`: 直近候補履歴
- `outputs/metrics_production.json`: 本番モデルの学習メトリクス
- `outputs/symbol_history/`: 銘柄別履歴確認のCSV
- `experiments/outputs/`: 評価・探索結果

最新候補の表示では、直近候補日のうち上位30位に入った回数を `回数` として表示します。最低表示数は1です。

## 設定

運用でよく触る値は `config/screening_settings.json` に集約しています。

- `gate`: 初動候補の抽出条件
- `label`: 学習ラベルの条件
- `screen.signal_count_days`: 直近何候補日で上位30入り回数を見るか

設定を変えた場合は、基本的に再学習とローリング評価で確認してください。

## 開発メモ

- Python は `.venv\Scripts\python.exe` を使う
- 軽い構文確認:

```powershell
& '.venv\Scripts\python.exe' -m compileall src scripts experiments\optimization
```

- 日常運用に不要な大規模探索スクリプトや一時比較スクリプトは削除済み
- 新しい実験を足す場合は、運用パスに混ぜず `experiments/` 配下へ置く
