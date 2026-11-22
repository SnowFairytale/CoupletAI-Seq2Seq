"""Draw the encoder-decoder architecture used in the report."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x, y, w, h, text, color="#EAF2FB", edge="#4C72B0", fontsize=10):
    patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.08",
                           linewidth=1.4, edgecolor=edge, facecolor=color)
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, start, end, style="-|>", color="#555555", lw=1.4, ls="-",
          connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=2, shrinkB=2,
                                 connectionstyle=connectionstyle))


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    # ----------------------------- encoder ----------------------------- #
    box(ax, 1.1, 4.4, 1.8, 0.9, "Upper couplet\n$x_1,\\dots,x_L$")
    box(ax, 3.3, 4.4, 1.6, 0.9, "Char\nembedding")
    box(ax, 5.5, 4.4, 1.9, 0.95, "BiGRU encoder\n(2 layers)", "#FFF3E0", "#DD8452")
    box(ax, 8.1, 4.4, 1.9, 0.9, "Encoder states\n$h_1,\\dots,h_L$", "#E8F5E9", "#55A868")

    arrow(ax, (2.0, 4.4), (2.5, 4.4))
    arrow(ax, (4.1, 4.4), (4.55, 4.4))
    arrow(ax, (6.45, 4.4), (7.15, 4.4))

    # ----------------------------- decoder ----------------------------- #
    box(ax, 0.95, 1.55, 1.7, 0.9, "Previous char\n$y_{t-1}$")
    box(ax, 2.95, 1.55, 1.55, 0.9, "Char\nembedding")
    box(ax, 4.45, 1.55, 1.35, 0.9, "Concat\n$[e_t;c_t]$", "#F3E5F5", "#8172B3")
    box(ax, 6.6, 1.55, 1.8, 0.95, "GRU decoder\n(2 layers)", "#FFF3E0", "#DD8452")
    box(ax, 8.75, 1.55, 1.6, 0.9, "Linear +\nsoftmax")
    box(ax, 10.85, 1.55, 1.55, 0.9, "Next char\n$y_t$", "#E8F5E9", "#55A868")

    # attention module above the concat node
    box(ax, 4.45, 3.15, 1.55, 0.85, "Additive\nattention", "#EDE7F6", "#8172B3")

    # embedding -> concat -> decoder -> output
    arrow(ax, (1.8, 1.55), (2.18, 1.55))
    arrow(ax, (3.73, 1.55), (3.78, 1.55))
    arrow(ax, (5.13, 1.55), (5.7, 1.55))
    arrow(ax, (7.5, 1.55), (7.95, 1.55))
    arrow(ax, (9.55, 1.55), (10.08, 1.55))

    # attention: encoder states -> attention -> concat
    arrow(ax, (8.1, 3.95), (5.25, 3.3), color="#8172B3", ls="--", lw=1.2)
    ax.text(6.45, 3.38, "all encoder states", fontsize=8.2, color="#8172B3",
            ha="center")
    arrow(ax, (6.6, 2.03), (5.2, 2.72), color="#8172B3", ls=":", lw=1.2,
          connectionstyle="arc3,rad=-0.2")
    ax.text(6.05, 2.38, "decoder state $s_{t-1}$", fontsize=8.5, color="#8172B3",
            ha="center")
    arrow(ax, (4.45, 2.72), (4.45, 2.02), color="#8172B3", lw=1.4)
    ax.text(4.62, 2.37, "context $c_t$", fontsize=8.5, color="#8172B3")

    # encoder final state -> decoder initial state
    arrow(ax, (8.9, 3.95), (7.05, 2.05), color="#DD8452", ls=":", lw=1.2,
          connectionstyle="arc3,rad=0.15")
    ax.text(8.55, 2.55, "initial hidden", fontsize=8.5, color="#DD8452")

    # autoregressive loop
    ax.plot([10.85, 10.85, 0.95, 0.95], [1.08, 0.55, 0.55, 1.08],
            color="#999999", lw=1.1)
    arrow(ax, (0.95, 0.62), (0.95, 1.06), color="#999999", lw=1.1)
    ax.text(5.9, 0.28, "autoregressive loop: teacher forcing during training, "
                       "greedy / beam search at inference",
            fontsize=9, color="#666666", ha="center")

    

    output = Path("results/figures")
    output.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output / "model_architecture.png", dpi=220)
    plt.close(fig)
    print(f"saved {output / 'model_architecture.png'}")


if __name__ == "__main__":
    main()


