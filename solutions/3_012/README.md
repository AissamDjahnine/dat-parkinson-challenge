# 012 — 011 with a 50-feature SBR head and an RAS guard

The exact program submitted. `main.py` reads `assets/config.json` and the weights that
`bash scripts/pack.sh 012` stages into `assets/`; train them with `bash recipes/012.sh`.

```bash
DATPARK_DATA_DIR=/path/with/niftis uv run python solutions/3_012/main.py
```
