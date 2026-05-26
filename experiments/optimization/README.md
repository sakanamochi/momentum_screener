# Optimization Experiments

このディレクトリは、パラメータ・特徴量探索用の隔離領域です。
本番モデル、通常の候補CSV、日々のスクリーニング運用と混同しないよう、探索スクリプトと検証メモをここにまとめています。

## 目的

- 初動ゲート条件、学習時クールダウン、サンプル重み、特徴量セットの影響を検証する
- 本番モデルや `outputs/candidates_current.csv` を上書きせずに、何十回から何百回の学習結果を比較する
- 一度の探索結果をそのまま採用せず、valid/test の両方で崩れにくい変更だけを本体に反映する

## 実行方法

標準の小さめ探索:

```powershell
experiments\optimization\optimize_experiments.bat
```

まず動作確認だけする場合:

```powershell
.\.venv\Scripts\python.exe experiments\optimization\optimize_experiments.py --study-name smoke --search-size smoke --max-trials 2 --epochs 1 --patience 1
```

座標探索で中規模に見る場合:

```powershell
.\.venv\Scripts\python.exe experiments\optimization\optimize_experiments.py --study-name medium_coord --algorithm coordinate --search-size medium --coordinate-passes 2 --epochs 35 --patience 6
```

現在の共通設定を基準に、周辺パラメータだけを軽く再検証する場合:

```powershell
.\.venv\Scripts\python.exe experiments\optimization\optimize_experiments.py --study-name focused_coord --algorithm coordinate --search-size focused --coordinate-passes 1 --epochs 25 --patience 5
```

MA25条件とクールダウンだけを確認する場合:

```powershell
.\.venv\Scripts\python.exe experiments\optimization\optimize_experiments.py --study-name focused_ma_cooldown --algorithm coordinate --search-size focused_ma_cooldown --coordinate-passes 1 --epochs 20 --patience 4
```

全組み合わせ候補からランダムに試す場合:

```powershell
.\.venv\Scripts\python.exe experiments\optimization\optimize_experiments.py --study-name medium_random_200 --algorithm random --search-size medium --max-trials 200 --epochs 35 --patience 6
```

## 出力

探索結果は `experiments/outputs/optimization/<study名>/` に保存されます。この配下は生成物なので、必要な概要を残したら削除して構いません。

- `config.json`: 探索条件
- `progress.json`: 現在の進捗、経過時間、推定残り時間、暫定ベスト
- `trials.csv`: trialごとの平均指標とスコア
- `folds.csv`: trialごとの各fold詳細
- `best_config.json`: 探索スコアが最も高かった設定

進捗確認:

```powershell
Get-Content experiments\outputs\optimization\medium_coord\progress.json
```

CSV確認:

```powershell
start experiments\outputs\optimization\medium_coord\trials.csv
```

途中停止:

```powershell
New-Item experiments\outputs\optimization\medium_coord\STOP -ItemType File
```

`STOP` ファイルを見つけると、実行中のfoldを保存してから次へ進まず終了します。強制終了よりもCSV/JSONが壊れにくい止め方です。

## 探索方針

座標探索は、他の条件を固定したまま1つのパラメータだけを動かし、その軸で一番良かった値を採用して次の軸へ進みます。
ランダムに全組み合わせを混ぜるより、どの変更が効いたかを追いやすく、trial数も抑えやすいです。

主な探索対象:

- 初動ゲート条件: 売買代金、5日騰落率、売買代金倍率、25日線乖離、クールダウン日数
- サンプル重み: `future_max_ret`, `target_future_max_ret`, `uniform`
- 特徴量セット: 全特徴量、簡素化セット、日中足形状系を除いたセット、株数回転率系を除いたセットなど

特徴量は基本的に「追加」よりも「悪影響がありそうなものを外す」方向で探索します。
地合い指標などの外部特徴量は、過去に性能悪化が大きかったため標準探索には含めていません。

## 2026-05-26 結果概要

### medium_coord

- 完走: 52/52 trials
- ベストスコア: trial 36, score 0.5058
- ベスト設定:
  - `feature_set=all`
  - `gate_min_turnover_5d=100000000`
  - `gate_min_ret_5d=-0.02`
  - `gate_ratio_profile=current`
  - `gate_min_close_ma25_ratio=-0.03`
  - `cooldown_days=20`
  - `sample_weight_mode=future_max_ret`

ただし trial 36 は valid では改善した一方、test 側が初期設定より悪化しました。
そのため、このベスト設定を丸ごと本モデルへ適用するのは見送りました。

初期設定 trial 1 と trial 36 の比較:

| 指標 | trial 1 | trial 36 |
| --- | ---: | ---: |
| score | 0.4650 | 0.5058 |
| valid p20 | 36.7% | 40.0% |
| valid stop-first p20 | 36.7% | 31.7% |
| test p20 | 35.0% | 31.7% |
| test p50 | 36.0% | 32.0% |
| test stop-first p20 | 43.3% | 51.7% |

### 一度採用を検討した変更

`gate_min_turnover_5d` を `50,000,000` から `100,000,000` に変更しました。
複数条件で、上位候補のヒット率と損切り先行率が安定して改善していたためです。

同じ初期寄り条件での比較:

| 指標 | 50,000,000 | 100,000,000 |
| --- | ---: | ---: |
| score | 0.4650 | 0.4708 |
| valid p20 | 36.7% | 36.7% |
| valid p50 | 30.7% | 31.3% |
| valid stop-first p20 | 36.7% | 35.0% |
| test p20 | 35.0% | 40.0% |
| test p50 | 36.0% | 43.3% |
| test stop-first p20 | 43.3% | 35.0% |

### 追加検証

出来高下限 `100,000,000` を固定して、次の周辺パラメータを検証しました。

- `gate_min_ret_5d`: `-0.02`, `-0.01`, `0.0`
- `gate_ratio_profile`: `current`, `one_day_strict`
- `gate_min_close_ma25_ratio`: `-0.03`, `-0.01`, `0.0`
- `cooldown_days`: `10`, `20`, `30`

結果:

- `gate_min_ret_5d=-0.01` 据え置き
- `gate_ratio_profile=current` 据え置き
- `gate_min_close_ma25_ratio=-0.01` 据え置き
- `cooldown_days=20` 据え置き

`cooldown_days=10` は test 側で良い場面がありましたが、valid 側が崩れやすいため採用しませんでした。
その後、実際の候補が大型・急騰済み銘柄に寄りすぎる副作用が見えたため、標準運用では `50,000,000` に戻しました。
この検証結果は、再検討用の記録として残しています。
