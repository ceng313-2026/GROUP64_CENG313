# Dataset Directory (`data/`)

Raw competition data files (`train/`, `test.csv`, `test_input.csv`) are not tracked in Git due to GitHub file size limits.

To download the dataset automatically in Google Colab or locally via Kaggle API:
```bash
kaggle competitions download -c nfl-big-data-bowl-2026-prediction -p ./data
unzip -q ./data/nfl-big-data-bowl-2026-prediction.zip -d ./data
