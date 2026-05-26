# momentum_screener

日本株の初動モメンタム候補を抽出し、ニューラルネットワークで「翌営業日始値から20営業日以内に+15%へ先に到達し、-10%へ先に到達しない確率」を推定する個人用スクリーニングツールです。

これは投資助言ではなく、検証用の候補抽出ツールです。

## 通常の使い方

候補を表示する場合:

```bat
scripts\screen_latest.bat
```

動き:

- 既存の `outputs/candidates_current.csv` が最新なら、再推論せずに即表示
- CSVが無い、`outputs/candidates_recent.csv` が無い、またはモデル/データより古い場合だけ、既存モデルで再推論
- 最新候補には `銘柄コード`, `銘柄名`, `現在株価`, `回数`, `初回日`, `初回後騰落`, `最終スコア` を表示
- 直近履歴には、最新日を除く過去5候補日の上位候補も表示
- 最新候補CSVは `outputs/candidates_current.csv`、直近履歴CSVは `outputs/candidates_recent.csv` に保存

最新候補の見方:

- `最終スコア`: モデルが推定したフォロースルー確率
- `回数`: 最新日を含む直近6候補日のうち、初動ゲートに入り、かつ `final_score >= 0.55` だった回数
- `初回日`: 直近6候補日の中で、同じ銘柄が初めて `final_score >= 0.55` になった日
- `初回後騰落`: `初回日` の終値から現在株価までの騰落
- 回数が1の銘柄は、初回日と初回後騰落を空欄表示

大引け後など、株価データを更新してから既存モデルで推論する場合:

```bat
scripts\update_and_screen.bat
```

これはモデルを再学習しません。日々の運用ではこちらが「データ更新あり」のメインです。

## たまに使う操作

運用モデルを再学習する場合:

```bat
scripts\train_current.bat
```

再学習は毎日行う想定ではありません。銘柄リストを更新した時、ゲート条件や特徴量を変えた時、または月1回程度の見直し用です。

パラメータ変更を時系列分割で確認する場合:

```bat
scripts\train_evaluation.bat
```

パラメータと特徴量セットをまとめて探索する場合は、通常運用と混ざらないよう `experiments/optimization/` を使います。
実行方法、停止方法、今回の検証結果概要は [experiments/optimization/README.md](experiments/optimization/README.md) にまとめています。

JPXの `Issues_*.csv` から普通株リストを再作成する場合:

1. JPX Client Portalから上場銘柄一覧CSVをダウンロードします。
2. ダウンロードしたCSVをリポジトリ内の `config` フォルダへ置きます。
3. ファイル名は `Issues_*.csv` の形にします。例: `config/Issues_20260526020732.csv`
4. 次のコマンドを実行します。

```bat
scripts\build_listed_stocks.bat
```

`scripts\build_listed_stocks.bat` は、`config\Issues_*.csv` のうち更新日時が一番新しいファイルを自動で使い、普通株だけに絞った `config\listed_stocks.csv` を作成します。既存の `config\listed_stocks.csv` は上書きされます。

銘柄情報CSVはJPX Client Portalの上場銘柄一覧から取得します。

https://clientportal.jpx.co.jp/ClientPortal/s/Issue?language=ja

初動ゲートの広さや取りこぼしを確認する場合:

```bat
scripts\inspect_gate.bat
```

PowerShell版とGit Bash版もあります。

```powershell
.\scripts\screen_latest.ps1
.\scripts\update_and_screen.ps1
.\scripts\train_current.ps1
.\scripts\train_evaluation.ps1
```

```bash
bash scripts/screen_latest.sh
bash scripts/update_and_screen.sh
bash scripts/train_current.sh
bash scripts/train_evaluation.sh
```

## 現在の標準ファイル

- `config/Issues_20260526020732.csv`: JPXから取得した元の上場銘柄CSV
- `config/listed_stocks.csv`: 普通株に絞った銘柄リスト
- `data/ohlcv_current.csv`: 現在のOHLCVキャッシュ
- `models/momentum_nn_current.pt`: 画面表示・推論で使う現在モデル。通常は運用モデルと同じ
- `models/momentum_nn_production.pt`: 運用モデル
- `outputs/candidates_current.csv`: 現在の候補CSV
- `outputs/candidates_recent.csv`: 最新日を含む直近6候補日の履歴CSV
- `outputs/metrics_current.json`: 現在モデルの指標。通常は運用モデルの指標
- `outputs/metrics_production.json`: 運用モデルの指標
- `outputs/gate_current.json`: 現在ゲートの確認指標

評価用モデルや評価結果は必要な時だけ `scripts\train_evaluation.bat` で再生成します。
ローリング評価や探索は、通常運用と混ざらないよう `experiments/optimization/` 側で行います。

## 現在モデルの学習条件

学習データ:

- 対象: JPX銘柄CSVから抽出した国内普通株
- 銘柄リスト: `config/listed_stocks.csv`
- 元データ: yfinance日足OHLCV
- OHLCV期間: `2020-01-06` から `2026-05-25`
- OHLCV取得銘柄数: `3,728`
- OHLCV行数: `5,388,274`

運用モデル:

- ファイル: `models/momentum_nn_production.pt`
- 推論用エイリアス: `models/momentum_nn_current.pt`
- 学習範囲: ラベルが作れる最新イベントまで
- 直近20営業日程度のイベントは、20営業日先の判定がまだできないため学習対象外
- 運用モデルは未知期間を残さず、使えるデータをなるべく学習に回す本番用
- 指標の `valid` は早期停止確認用であり、厳密な将来評価ではありません

ラベル:

- 候補日の翌営業日始値をエントリー価格として使用
- 20営業日以内に `+15%` へ先に到達し、`-10%` へ先に到達しない場合を成功
- 同じ日に利確ラインと損切りラインへ両方到達した場合は保守的に失敗扱い
- `stop_first_rate_at_20/50` は上位20/50件のうち、損切りラインへ先に到達した比率

現在の初動ゲート:

- `turnover_5d_avg >= 100,000,000`
- `ret_5d > -0.01`
- `turnover_ratio_1d_20d >= 1.05` または `turnover_ratio_5d_20d >= 1.05`
- `close_ma25_ratio >= -0.01`

学習時と推論時の候補扱い:

- 学習時は、同じ上昇イベントを重複学習しにくくするため、同一銘柄の候補に20営業日クールダウンを適用
- 推論時は、クールダウン後の `initial_momentum` ではなく、クールダウン前の `raw_initial_momentum` をスコアリング
- そのため、同じ銘柄が別日で直近履歴に複数回表示される
- `screen_latest.bat` では、最新候補CSVは最新日だけを出力しつつ、回数・初回日・初回後騰落は直近6候補日で集計

推論CSVの主な追加列:

- `recent_signal_count`: 出力期間内でraw条件に入った回数
- `raw_recent_signal_count`: 集計期間内でraw条件に入った回数
- `score_recent_signal_count`: 集計期間内でraw条件に入り、かつ `final_score >= 0.55` だった回数
- `first_score_signal_date`: 集計期間内で最初に `final_score >= 0.55` になった日
- `first_score_signal_close`: `first_score_signal_date` の終値
- `return_since_first_score_signal`: `first_score_signal_close` から現在株価までの騰落

## 評価メモ

`+15% / -10%` ローリング評価:

- `valid_2023`: `valid_precision_at_20` 0.30
- `valid_2024`: `valid_precision_at_20` 0.45
- `valid_2025`: `valid_precision_at_20` 0.35

TOPIX proxy特徴量は試しましたが、full版は `valid_2024` が悪化し、相対リターン2本版も地合いなしを上回らなかったため、現時点では採用していません。

学習時のサンプル重み:

- 標準は `--sample-weight-mode future_max_ret`
- 従来どおり、将来最大上昇率を `0%` から `30%` の範囲で重みに反映します
- `--sample-weight-mode target_future_max_ret` を指定すると、成功ラベルのイベントだけ将来最大上昇率で重くします
- `--sample-weight-mode uniform` を指定すると、全イベントを同じ重みで学習します

重み付け比較メモ:

- `target_future_max_ret` は、失敗ラベルなのに一時的な高騰だけで重くなる問題を避ける案です
- ローリング評価では `test_precision_at_50` は改善しましたが、`valid_precision_at_20` と `stop_first_rate` が悪化しました
- 現時点では本モデルには適用せず、標準の `future_max_ret` を維持します

## セットアップ

初回だけ実行します。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## 注意

- yfinanceはリクエスト制限があります。通常確認は `screen_latest.bat` を使い、データ更新が必要な時だけ `update_and_screen.bat` を使ってください。
- yfinanceの日本株データには欠損や調整の癖があります。本運用ではJ-Quantsなどの安定したデータソースへの差し替えを推奨します。
- 長期間学習に広げる場合は、サンプル数が増える一方で、古い相場の癖や生存者バイアスも強くなります。比較実験してから採用してください。
