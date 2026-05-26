# Evaluation Experiments

このディレクトリは、通常運用とは分けて評価用モデルやローリング評価を実行するための場所です。
生成物は `experiments/outputs/<検証名>/` に保存します。

## 単発の評価用学習

```bat
experiments\evaluation\train_evaluation.bat
```

検証名を指定する場合:

```bat
experiments\evaluation\train_evaluation.bat trial_ret_filter
```

出力:

- `experiments/outputs/<検証名>/models/momentum_nn.pt`
- `experiments/outputs/<検証名>/metrics.json`
- `experiments/outputs/<検証名>/candidates.csv`

## ローリング評価

```bat
experiments\evaluation\rolling_evaluation.bat
```

検証名を指定する場合:

```bat
experiments\evaluation\rolling_evaluation.bat rolling_baseline
```

出力:

- `experiments/outputs/<検証名>/metrics.json`
- `experiments/outputs/<検証名>/folds.csv`

## メモ

標準の初動ゲート、利確/損切り、スコア閾値は `config/screening_settings.json` を参照します。
検証ごとに結果フォルダが分かれるので、不要になった検証は `experiments/outputs/<検証名>/` ごと削除できます。

## Metrics

`metrics.json` と `folds.csv` に出る主な項目です。

### 評価区間

- `train_*`: 学習期間内の上位候補に対する指標。過学習の影響を受けるので参考値。
- `valid_*`: 早期停止や設定比較に使う検証期間の指標。
- `test_*`: `valid_end` より後の期間がある場合の未知期間指標。運用モデルのように `valid_end=2099-12-31` の場合は `test_events=0` になる。

### 件数

- `train_events`, `valid_events`, `test_events`: 各区間で評価対象になった候補イベント数。
- `all_events`: 学習用データセット全体の候補イベント数。
- `raw_gate_rows`: クールダウン前の初動ゲートに入った行数。
- `initial_momentum_events`: 学習用クールダウン適用後の候補イベント数。
- `symbols_with_ohlcv`: OHLCVデータがある銘柄数。

### 上位候補の成績

- `precision_at_20`, `precision_at_50`: スコア上位20件/50件のうち、成功ラベルだった割合。
- `stop_first_rate_at_20`, `stop_first_rate_at_50`: スコア上位20件/50件のうち、利確より先に損切りラインへ到達した割合。低いほどよい。
- `avg_days_to_profit_at_20`, `avg_days_to_profit_at_50`: 成功した上位候補が利確ラインへ到達するまでの平均営業日数。
- `avg_future_max_ret_at_20`, `avg_future_max_ret_at_50`: スコア上位20件/50件の、翌営業日始値から20営業日内高値までの平均最大上昇率。

### 学習状態

- `best_valid_loss`: 学習中に最も良かった検証損失。小さいほどモデルの検証誤差は小さいが、投資成績とは必ずしも一致しない。
- `epochs_trained`: 実際に学習したエポック数。早期停止した場合は指定 `epochs` より小さくなる。

### ゲートとラベル条件

- `training_config`: 学習・評価に使った設定のスナップショット。`data`, `gate`, `label`, `sample_weight`, `model` に分けて保存する。
- `gate_recall`: `target_threshold` 以上の将来上昇があった全行のうち、raw初動ゲートで拾えた割合。
- `label_mode`: ラベル作成方法。通常は `barrier`。
- `profit_barrier`: 成功判定の利確ライン。
- `stop_barrier`: 失敗判定の損切りライン。
- `target_threshold`: `max_ret` ラベルや `gate_recall` で使う上昇率閾値。
- `sample_weight_mode`: 学習サンプルの重み付け方法。
- `sample_weight_cap`: 重み付けに使う将来最大上昇率の上限。
- `sample_weight_scale`: 将来最大上昇率を重みに反映する倍率。
