# Evaluation

このフォルダは、通常運用のモデルを評価するための入口だけを残しています。

## 単発評価

```bat
experiments\evaluation\train_evaluation.bat
```

出力:

```text
experiments/outputs/train_evaluation/metrics.json
experiments/outputs/train_evaluation/candidates.csv
experiments/outputs/train_evaluation/models/momentum_nn.pt
```

## ローリング評価

```bat
experiments\evaluation\rolling_evaluation.bat
```

出力:

```text
experiments/outputs/rolling_evaluation/folds.csv
experiments/outputs/rolling_evaluation/metrics.json
```

別名で保存したい場合:

```bat
experiments\evaluation\rolling_evaluation.bat my_study_name
```

## 主に見る指標

- `precision_at_20`, `precision_at_50`: スコア上位20/50件の成功率
- `stop_first_rate_at_20`, `stop_first_rate_at_50`: 利確より先に損切りラインへ到達した割合。低いほどよい
- `avg_future_max_ret_at_20`, `avg_future_max_ret_at_50`: 上位候補の平均最大上昇率
- `avg_future_min_ret_at_20`, `avg_future_min_ret_at_50`: 上位候補の平均最大下落率。浅いほどよい
- `avg_days_to_profit_at_20`, `avg_days_to_profit_at_50`: 利確到達までの平均営業日数

採用判断では validation だけでなく test 側の悪化を必ず確認してください。
