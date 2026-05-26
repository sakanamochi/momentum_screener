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
- CSVが無い、またはモデル/データより古い場合だけ、既存モデルで再推論
- 画面には `銘柄コード`, `銘柄名`, `現在株価`, `最終スコア` だけ表示
- CSV自体は `outputs/candidates_current.csv` に保存

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

複数年で安定性を見るローリング評価を行う場合:

```bat
scripts\rolling_evaluation_barrier15_10.bat
```

JPXの `Issues_*.csv` から普通株リストを再作成する場合:

```bat
scripts\build_listed_stocks.bat
```

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
- `outputs/metrics_current.json`: 現在モデルの指標。通常は運用モデルの指標
- `outputs/metrics_production.json`: 運用モデルの指標
- `outputs/gate_current.json`: 現在ゲートの確認指標

評価用モデルや評価結果は必要な時だけ `scripts\train_evaluation.bat` や `scripts\rolling_evaluation_barrier15_10.bat` で再生成します。

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

- `turnover_5d_avg >= 50,000,000`
- `ret_5d > -0.01`
- `turnover_ratio_1d_20d >= 1.05` または `turnover_ratio_5d_20d >= 1.05`
- `close_ma25_ratio >= -0.01`
- 同一銘柄の候補は20営業日クールダウン

## 評価メモ

`+15% / -10%` ローリング評価:

- `valid_2023`: `valid_precision_at_20` 0.30
- `valid_2024`: `valid_precision_at_20` 0.45
- `valid_2025`: `valid_precision_at_20` 0.35

TOPIX proxy特徴量は試しましたが、full版は `valid_2024` が悪化し、相対リターン2本版も地合いなしを上回らなかったため、現時点では採用していません。

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
