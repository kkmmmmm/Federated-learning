"""Figure generation.

Figures are rendered at their *final* physical size (figsize in inches) and high
DPI, so they can be placed in Word at native size without any resizing/stretching.

* ``panel_figure``      -- 4x4 grid (one panel per validation region) of point
  estimates with 95% CI error bars (Figures 1-3 and S1-S9). y-axis range and
  tick spacing match the original Excel charts.
* ``pca_figure``        -- Figure 4: PC1/PC2 scatter of model coefficients with
  shared (aligned) axes across penalties and non-overlapping labels.
* ``convergence_figure``-- FL global training log-loss / AUROC vs round.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import config as C

PEN_TITLE = C.PENALTY_LABEL


def _yticks(ylim, major):
    return np.round(np.arange(ylim[0], ylim[1] + major / 2, major), 4)


def panel_figure(between_pen: pd.DataFrame, metric: str, red_region: int,
                 ylabel: str, suptitle: str, outpath: str,
                 ref_line: float | None = None, ylim=None, major: float | None = None,
                 figsize=(9.4, 6.1), dpi: int = 200):
    # NB: ``suptitle`` is intentionally not drawn — the model (penalty) name is
    # already given in the figure caption in the manuscript/supplement, so the
    # freed vertical space is given back to the panels for readability.
    fig, axes = plt.subplots(4, 4, figsize=figsize, sharey=True)
    red = str(red_region)
    yticks = _yticks(ylim, major) if (ylim and major) else None
    for ax, r in zip(axes.ravel(), C.REGIONS):
        sub = between_pen[between_pen["ValidationRegion"] == r].copy()
        sub["Source"] = sub["Source"].astype(str)
        sub = sub.set_index("Source")
        order = [str(i) for i in C.REGIONS if i != r] + ["Centralized", "FL"]
        xs = np.arange(len(order))
        for x, s in zip(xs, order):
            p = sub.loc[s, metric]
            lo = sub.loc[s, f"{metric}_lower"]; up = sub.loc[s, f"{metric}_upper"]
            yerr = [[max(p - lo, 0)], [max(up - p, 0)]]
            if s == "Centralized":
                col, mk, ms = "#1f77b4", "s", 4.5
            elif s == "FL":
                col, mk, ms = "#2ca02c", "D", 4.5
            elif s == red:
                col, mk, ms = "red", "o", 3.5
            else:
                col, mk, ms = "black", "o", 3
            ax.errorbar(x, p, yerr=yerr, fmt=mk, color=col, ms=ms,
                        ecolor=col, elinewidth=0.6, capsize=1.0, alpha=0.85)
        if ref_line is not None:
            ax.axhline(ref_line, color="grey", ls="--", lw=0.6)
        ax.set_title(f"Region {r}", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"R{s}" if s.isdigit() else s[:4] for s in order],
                           rotation=90, fontsize=5)
        ax.tick_params(axis="y", labelsize=7)
        if ylim:
            ax.set_ylim(*ylim)
        if yticks is not None:
            ax.set_yticks(yticks)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=6)
    handles = [
        Line2D([], [], color="black", marker="o", ls="", label="Local (other regions)"),
        Line2D([], [], color="red", marker="o", ls="", label=f"Local Region {red_region}"),
        Line2D([], [], color="#1f77b4", marker="s", ls="", label="Centralized (excl. region)"),
        Line2D([], [], color="#2ca02c", marker="D", ls="", label="FL (excl. region)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.04, 1, 1.0])
    fig.savefig(outpath, dpi=dpi)
    fig.savefig(outpath.replace(".png", ".pdf"))
    plt.close(fig)


def pca_figure(pca: dict[str, pd.DataFrame], outpath: str,
               figsize=(9.4, 6.0), dpi: int = 200):
    try:
        from adjustText import adjust_text
        have_adjust = True
    except Exception:
        have_adjust = False

    present = [p for p in C.PENALTIES if p in pca]
    allx = np.concatenate([pca[p]["PC1"].to_numpy() for p in present])
    ally = np.concatenate([pca[p]["PC2"].to_numpy() for p in present])
    xpad = 0.12 * (allx.max() - allx.min()); ypad = 0.12 * (ally.max() - ally.min())
    xlim = (allx.min() - xpad, allx.max() + xpad)
    ylim = (ally.min() - ypad, ally.max() + ypad)

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axl = axes.ravel()
    for ax in axl[len(present):]:
        ax.axis("off")
    for ax, pen in zip(axl, present):
        df = pca[pen]; texts = []
        for _, rr in df.iterrows():
            idx = rr["Index"]
            if idx == "FL":
                ax.scatter(rr.PC1, rr.PC2, marker="s", color="red", s=50, zorder=4)
                texts.append(ax.text(rr.PC1, rr.PC2, "FL", color="red",
                                     fontsize=7, fontweight="bold"))
            elif idx == "Centralized":
                ax.scatter(rr.PC1, rr.PC2, marker="^", color="#1f77b4", s=50, zorder=4)
                texts.append(ax.text(rr.PC1, rr.PC2, "Cen", color="#1f77b4",
                                     fontsize=7, fontweight="bold"))
            else:
                ax.scatter(rr.PC1, rr.PC2, marker="o", color="black", s=20, zorder=3)
                texts.append(ax.text(rr.PC1, rr.PC2, str(idx), fontsize=6))
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.axhline(0, color="grey", lw=0.4); ax.axvline(0, color="grey", lw=0.4)
        ax.set_title(PEN_TITLE[pen], fontsize=10)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("PC1", fontsize=7); ax.set_ylabel("PC2", fontsize=7)
        if have_adjust:
            adjust_text(texts, ax=ax, expand=(1.25, 1.5),
                        arrowprops=dict(arrowstyle="-", color="grey", lw=0.4))
    handles = [
        Line2D([], [], color="black", marker="o", ls="", label="Region models"),
        Line2D([], [], color="#1f77b4", marker="^", ls="", label="Centralized"),
        Line2D([], [], color="red", marker="s", ls="", label="FL"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    # Overall title omitted — described in the figure caption. Per-panel penalty
    # labels are kept so each quadrant remains identifiable.
    fig.tight_layout(rect=[0, 0.05, 1, 1.0])
    fig.savefig(outpath, dpi=dpi)
    fig.savefig(outpath.replace(".png", ".pdf"))
    plt.close(fig)


def convergence_figure(conv: dict, outpath: str, figsize=(9.4, 4.6), dpi: int = 200):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    colors = {"none": "black", "l1": "#1f77b4", "l2": "#2ca02c", "elasticnet": "#d62728"}
    for pen, h in conv.items():
        ax1.plot(h.rounds, h.train_logloss, marker="o", ms=3, color=colors[pen],
                 label=PEN_TITLE[pen])
        ax2.plot(h.rounds, h.train_auroc, marker="o", ms=3, color=colors[pen],
                 label=PEN_TITLE[pen])
        if h.converged_round:
            ax1.axvline(h.converged_round, color=colors[pen], ls=":", lw=0.8)
    ax1.set_xlabel("Communication round"); ax1.set_ylabel("Global training log-loss")
    ax1.legend(fontsize=8)
    ax2.set_xlabel("Communication round"); ax2.set_ylabel("Global training AUROC")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    fig.savefig(outpath.replace(".png", ".pdf"))
    plt.close(fig)
