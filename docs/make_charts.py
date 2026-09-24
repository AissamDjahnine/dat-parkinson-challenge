"""Render the aggregate README charts (scores only, no scan data).

    uv run python docs/make_charts.py
    dot -Tsvg docs/diagrams/pipeline.dot -o docs/figures/pipeline.svg
    dot -Tsvg docs/diagrams/lineage.dot  -o docs/figures/lineage.svg
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).parent / "figures"
BG, PANEL, FG, DIM, TOP, REST = "#0d1117", "#161b22", "#c9d1d9", "#8b949e", "#f78166", "#58a6ff"

# (name, grouped-CV log loss, public log loss) for every scored submission with a recorded CV
SCORED = [("003", .2793, .2814), ("008a", .2472, .2653), ("011", .2448, .2665),
          ("012", .2426, .2702), ("SIAM25", .2423, .2806), ("014", .2407, .2798),
          ("PENTA1", .2364, .2708), ("SILOSIA", .2355, .2732), ("SILO-V3", .2337, .2770)]


def cv_vs_public() -> None:
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL, "text.color": FG,
                         "axes.labelcolor": FG, "xtick.color": DIM, "ytick.color": DIM,
                         "axes.edgecolor": "#30363d", "font.size": 9})
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for name, cv, pub in SCORED:
        top = name in ("008a", "011", "012")
        ax.scatter(cv, pub, s=48 if top else 28, color=TOP if top else REST, zorder=3,
                   edgecolor=BG, lw=0.6)
        ax.annotate(name, (cv, pub), xytext=(5, 4), textcoords="offset points", fontsize=8,
                    color=FG if top else DIM)
    ax.set_xlim(.2815, .2325)                     # better CV to the right
    ax.set_ylim(.2605, .2835)
    ax.set_xlabel("grouped CV log loss   (better ->)")
    ax.set_ylabel("public log loss   (lower is better)")
    ax.grid(color="#21262d", lw=0.6)
    ax.set_title("Past CV 0.247, a better CV score bought a worse public score", color=FG,
                 fontsize=10)
    fig.savefig(OUT / "cv_vs_public.svg", facecolor=BG, bbox_inches="tight", pad_inches=0.12,
                metadata={"Date": None})


if __name__ == "__main__":
    cv_vs_public()
