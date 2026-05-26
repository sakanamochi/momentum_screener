# Optimization Notes

このフォルダは、日常運用ではなく検証用です。

## 残しているもの

- `evaluate_risk_presets.py`
  - 1回の学習結果に対して、複数の `risk_adjustment` プリセットを比較します。
  - 学習をプリセットごとにやり直さないので比較的軽いです。

- `scoring_design_notes.txt`
  - 今後の目的関数、評価指標、最終スコア設計のメモです。

## 実行例

```powershell
& '.venv\Scripts\python.exe' experiments\optimization\evaluate_risk_presets.py
```

出力:

```text
experiments/outputs/risk_presets/folds.csv
experiments/outputs/risk_presets/summary.csv
experiments/outputs/risk_presets/metrics.json
```

## 方針

大きな探索スクリプトは削除済みです。今後も、必要になった比較だけを小さいスクリプトとして追加してください。
