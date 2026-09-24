# 011 — 008a with sm0 retrained under count noise

The exact program submitted. `main.py` reads `assets/config.json` and the weights that
`bash scripts/pack.sh 011` stages into `assets/`; train them with `bash recipes/011.sh`.

```bash
DATPARK_DATA_DIR=/path/with/niftis uv run python solutions/2_011/main.py
```
