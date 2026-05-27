# Codex Handoff

このリポジトリで作業するCodex向けの短い引継ぎメモです。

## Python実行

- Windows環境では基本的に `.\.venv\Scripts\python.exe` を使う。
- Codexのサンドボックスでは、このPython実行に承認が必要になることがある。
- 既に許可されている場合でも、失敗したら同じコマンドを `require_escalated` で再実行する。
- 任意の `python` ではなく、できるだけ `.venv\Scripts\python.exe` を明示する。

## UTF-8と文字化け

- READMEや設定ファイルはUTF-8前提。
- PowerShellの `Get-Content` は環境によって日本語が文字化けすることがある。
- 日本語を確認する時は以下のどちらかを使う。

```powershell
Get-Content README.md -Encoding UTF8
```

または:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe -c "from pathlib import Path; print(Path('README.md').read_text(encoding='utf-8'))"
```

## 設定

- 初動ゲート、ラベル条件、スコア閾値は `config/screening_settings.json` に集約されている。
- 値の意味は同ファイルの `_description` を見る。
- 値を変えたら、productionモデルの再学習と再推論が必要。

## 運用と検証

- 日常運用のモデルは `models/momentum_nn_production.pt`。
- 通常の出力は `outputs/` に上書き保存する。
- 検証用のモデルや結果は `experiments/outputs/<検証名>/` に保存する。
- 単発評価とローリング評価は `experiments/evaluation/`。
- パラメータ探索は `experiments/optimization/`。
