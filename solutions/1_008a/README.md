# 008a — the submission behind the final private score

The exact program submitted. `main.py` reads `assets/config.json` and the weights that
`bash scripts/pack.sh 008a` stages into `assets/`; train them with `bash recipes/008a.sh`.

```bash
DATPARK_DATA_DIR=/path/with/niftis uv run python solutions/1_008a/main.py
```
