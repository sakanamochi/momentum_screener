# momentum_screener

日本株の「初動モメンタム候補」を広めのルールで抽出し、その後20営業日以内にさらに伸びる確率を小さなPyTorch NNで推定する試作ツールです。

## できること

- yfinanceから日足OHLCVを取得
- 候補日当日までに分かる情報だけで特徴量を作成
- ゆるい初動候補条件でイベントを抽出
- `future_max_ret_20d` から `target_20d` と `sample_weight` を作成
- `sample_weight` 付きでNNを学習
- 最新候補を `outputs/candidates.csv` に確率順で出力

## セットアップ

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## 使い方

通常は `scripts` だけ使えば十分です。

日々の通常運用。OHLCVだけ更新し、既存モデルで推論します。CSVが最新なら表示だけ行います。

```bat
scripts\screen_latest.bat
```

PowerShellから実行する場合:

```powershell
.\scripts\screen_latest.ps1
```

Git Bashなどから実行する場合:

```bash
bash scripts/screen_latest.sh
```

JPXの `Issues_*.csv` を `config/` に置いたあと、普通株リストを再作成する:

```bat
scripts\build_listed_stocks.bat
```

```powershell
.\scripts\build_listed_stocks.ps1
```

モデルを再学習する。毎日行う必要はなく、銘柄リストや方針を変えた時、または月1回程度の見直し用です。

```bat
scripts\train_current.bat
```

```powershell
.\scripts\train_current.ps1
```

初動ゲートの広さを確認する:

```bat
scripts\inspect_gate.bat
```

```powershell
.\scripts\inspect_gate.ps1
```

標準の出力先:

- `config/listed_stocks.csv`: 普通株リスト
- `data/ohlcv_current.csv`: 現在のOHLCVキャッシュ
- `models/momentum_nn_current.pt`: 現在の学習済みモデル
- `outputs/candidates_current.csv`: 現在の候補
- `outputs/metrics_current.json`: 現在の評価指標
- `outputs/gate_current.json`: 現在の初動ゲート指標

## 直接CLIを使う場合

サンプル銘柄で学習から出力まで実行します。

```powershell
momentum-screener run --refresh
```

出力:

- `data/ohlcv.csv`: 取得した日足データのキャッシュ
- `models/momentum_nn.pt`: 学習済みNN
- `outputs/metrics.json`: 評価指標
- `outputs/candidates.csv`: 最新候補

銘柄を増やす場合は `config/tickers_sample.txt` を編集するか、別ファイルを指定してください。

```powershell
momentum-screener run --tickers-file config/my_tickers.txt --refresh
```

東証銘柄を広く試す場合は、4桁コードから `1300.T` 〜 `9999.T` を生成できます。欠番やyfinanceで取れない銘柄は自動的に空データとして落ちます。

まずは件数を絞った試運転を推奨します。

```powershell
momentum-screener run --ticker-universe tse-all --max-tickers 300 --no-sample-tickers --refresh
```

問題なく動いたら、範囲を広げます。全範囲は時間がかかるため、最初はコード帯で分けるのが扱いやすいです。

```powershell
momentum-screener run --ticker-universe tse-all --code-start 1300 --code-end 3999 --no-sample-tickers --cache data/ohlcv_1300_3999.csv --refresh
momentum-screener run --ticker-universe tse-all --code-start 4000 --code-end 6999 --no-sample-tickers --cache data/ohlcv_4000_6999.csv --refresh
momentum-screener run --ticker-universe tse-all --code-start 7000 --code-end 9999 --no-sample-tickers --cache data/ohlcv_7000_9999.csv --refresh
```

既存モデルで最新候補だけ出し直す場合:

```powershell
momentum-screener screen
```

## 改善用コマンド

初動ゲートを緩めた時の取りこぼしを、NN学習なしで確認できます。

```powershell
momentum-screener inspect-gate --cache-only --cache data/ohlcv_tse_3000_3999.csv `
  --gate-min-turnover-5d 50000000 `
  --gate-min-ret-5d -0.01 `
  --gate-min-turnover-ratio-1d-20d 1.05 `
  --gate-min-turnover-ratio-5d-20d 1.05 `
  --gate-min-close-ma25-ratio -0.01
```

複数の取得済みキャッシュを結合して、再学習できます。

```powershell
momentum-screener merge-caches data/ohlcv_tse_1800_2799.csv data/ohlcv_tse_3000_3999.csv --output data/ohlcv_common_like_1800_3999.csv
momentum-screener run --cache-only --cache data/ohlcv_common_like_1800_3999.csv --model-path models/momentum_nn_common_like.pt --output outputs/candidates_common_like.csv
```

普通株中心で試す場合は、ETF/ETNが多い `1300` 番台や一部 `2000` 番台を避け、まず `3000-3999` などから広げると候補が見やすくなります。より正確にはJPXやJ-Quantsの銘柄マスタから普通株だけのCSVを作り、`--ticker-csv` を使ってください。

```powershell
momentum-screener run --ticker-csv config/listed_stocks.csv --ticker-csv-code-column code --no-sample-tickers --refresh
```

## 注意

これは投資助言ではなく、検証用のスクリーニング基盤です。yfinanceの日本株データは欠損や調整の癖があるため、本運用ではJ-Quantsなどの安定したデータソースへの差し替えを推奨します。
