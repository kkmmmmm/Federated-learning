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


_SRC_LABEL = {"Centralized": "Cent", "FL": "FL", "FL_recal": "FLrc", "FedAvg": "FedAvg"}


def panel_figure(between_pen: pd.DataFrame, metric: str, red_region: int,
                 ylabel: str, suptitle: str, outpath: str,
                 ref_line: float | None = None, ylim=None, major: float | None = None,
                 extra_sources: list | None = None,
                 figsize=(12.5, 7.2), dpi: int = 300):
    # NB: ``suptitle`` is intentionally not drawn — the model (penalty) name is
    # already given in the figure caption in the manuscript/supplement, so the
    # freed vertical space is given back to the panels for readability.
    # ``extra_sources`` (e.g. ["FL_recal"]) appends the recalibrated FL model;
    # used only for the calibration-intercept figure (it is identical to FL for
    # AUROC and the calibration slope, an intercept-only correction).
    extra = extra_sources or []
    fig, axes = plt.subplots(4, 4, figsize=figsize, sharey=True)
    red = str(red_region)
    yticks = _yticks(ylim, major) if (ylim and major) else None
    for ax, r in zip(axes.ravel(), C.REGIONS):
        sub = between_pen[between_pen["ValidationRegion"] == r].copy()
        sub["Source"] = sub["Source"].astype(str)
        sub = sub.set_index("Source")
        order = [str(i) for i in C.REGIONS if i != r] + ["Centralized", "FL"] + extra
        xs = np.arange(len(order))
        for x, s in zip(xs, order):
            if s not in sub.index:
                continue
            p = sub.loc[s, metric]
            lo = sub.loc[s, f"{metric}_lower"]; up = sub.loc[s, f"{metric}_upper"]
            yerr = [[max(p - lo, 0)], [max(up - p, 0)]]
            mfc = None
            if s == "Centralized":
                col, mk, ms = "#0072B2", "s", 5.5
            elif s == "FL":
                col, mk, ms = "#009E73", "D", 5.5
            elif s == "FL_recal":
                col, mk, ms, mfc = "#009E73", "D", 6.0, "none"   # hollow = recalibrated FL
            elif s == "FedAvg":
                col, mk, ms, mfc = "#ff7f0e", "D", 6.0, "none"   # uncorrected FedAvg
            elif s == red:
                col, mk, ms = "#D55E00", "o", 4.0
            else:
                col, mk, ms = "#333333", "o", 3.5
            ax.errorbar(x, p, yerr=yerr, fmt=mk, color=col, ms=ms,
                        ecolor=col, elinewidth=0.6, capsize=1.0, alpha=0.85,
                        markerfacecolor=(mfc if mfc else col))
        if ref_line is not None:
            ax.axhline(ref_line, color="grey", ls="--", lw=0.6)
        ax.set_title(f"Region {r}", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"R{s}" if s.isdigit() else _SRC_LABEL.get(s, s[:4])
                            for s in order], rotation=90, fontsize=6)
        ax.tick_params(axis="y", labelsize=8)
        if ylim:
            ax.set_ylim(*ylim)
        if yticks is not None:
            ax.set_yticks(yticks)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=8)
    handles = [
        Line2D([], [], color="#333333", marker="o", ls="", label="Local (other regions)"),
        Line2D([], [], color="#D55E00", marker="o", ls="", label=f"Local Region {red_region}"),
        Line2D([], [], color="#0072B2", marker="s", ls="", label="Centralized (excl. region)"),
        Line2D([], [], color="#009E73", marker="D", ls="", label="FL (excl. region)"),
    ]
    if "FL_recal" in extra:
        handles.append(Line2D([], [], color="#009E73", marker="D", ls="",
                              markerfacecolor="none", label="FL recalibrated"))
    if "FedAvg" in extra:
        handles.append(Line2D([], [], color="#ff7f0e", marker="D", ls="",
                              markerfacecolor="none", label="FedAvg (uncorrected)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.04, 1, 1.0])
    fig.savefig(outpath, dpi=dpi)
    fig.savefig(outpath.replace(".png", ".pdf"))
    plt.close(fig)


def pca_figure(pca: dict[str, pd.DataFrame], outpath: str,
               figsize=(9.4, 6.0), dpi: int = 300):
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

    red = str(C.RED_REGION_AUROC)
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    axl = axes.ravel()
    for ax in axl[len(present):]:
        ax.axis("off")
    for ax, pen in zip(axl, present):
        df = pca[pen]; texts = []
        for _, rr in df.iterrows():
            idx = str(rr["Index"])
            if idx == "FL":
                ax.scatter(rr.PC1, rr.PC2, marker="D", color="#009E73", s=48, zorder=4)
                texts.append(ax.text(rr.PC1, rr.PC2, "FL", color="#009E73",
                                     fontsize=8.5, fontweight="bold"))
            elif idx == "FL_recal":
                ax.scatter(rr.PC1, rr.PC2, marker="D", facecolors="none",
                           edgecolors="#009E73", s=60, linewidths=1.4, zorder=5)
                texts.append(ax.text(rr.PC1, rr.PC2, "FLrc", color="#009E73",
                                     fontsize=8.5, fontweight="bold"))
            elif idx == "Centralized":
                ax.scatter(rr.PC1, rr.PC2, marker="^", color="#0072B2", s=55, zorder=4)
                texts.append(ax.text(rr.PC1, rr.PC2, "Cent", color="#0072B2",
                                     fontsize=8.5, fontweight="bold"))
            elif idx == red:
                ax.scatter(rr.PC1, rr.PC2, marker="o", color="#D55E00", s=42, zorder=4)
                texts.append(ax.text(rr.PC1, rr.PC2, idx, color="#D55E00",
                                     fontsize=8.5, fontweight="bold"))
            else:
                ax.scatter(rr.PC1, rr.PC2, marker="o", color="#333333", s=20, zorder=3)
                texts.append(ax.text(rr.PC1, rr.PC2, idx, fontsize=7.5))
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.axhline(0, color="grey", lw=0.4); ax.axvline(0, color="grey", lw=0.4)
        ax.set_title(PEN_TITLE[pen], fontsize=12)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("PC1", fontsize=10); ax.set_ylabel("PC2", fontsize=10)
        if have_adjust:
            adjust_text(texts, ax=ax, expand=(1.25, 1.5),
                        arrowprops=dict(arrowstyle="-", color="grey", lw=0.4))
    handles = [
        Line2D([], [], color="#333333", marker="o", ls="", label="Region models"),
        Line2D([], [], color="#D55E00", marker="o", ls="", label=f"Region {red}"),
        Line2D([], [], color="#0072B2", marker="^", ls="", label="Centralized"),
        Line2D([], [], color="#009E73", marker="D", ls="", label="FL"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10, frameon=False)
    # Overall title omitted — described in the figure caption. Per-panel penalty
    # labels are kept so each quadrant remains identifiable.
    fig.tight_layout(rect=[0, 0.05, 1, 1.0])
    fig.savefig(outpath, dpi=dpi)
    fig.savefig(outpath.replace(".png", ".pdf"))
    plt.close(fig)


def convergence_figure(conv: dict, outpath: str, figsize=(12.5, 4.6), dpi: int = 300):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    colors = {"none": "#333333", "l1": "#0072B2", "l2": "#009E73", "elasticnet": "#D55E00"}
    have_recal = False
    for pen, h in conv.items():
        ax1.plot(h.rounds, h.train_logloss, marker="o", ms=3, color=colors[pen],
                 label=PEN_TITLE[pen])
        ax2.plot(h.rounds, h.train_auroc, marker="o", ms=3, color=colors[pen],
                 label=PEN_TITLE[pen])
        if h.converged_round:
            ax1.axvline(h.converged_round, color=colors[pen], ls=":", lw=0.8)
        # Final federated intercept-recalibration step (a single extra round).
        rll = getattr(h, "recal_logloss", None)
        rau = getattr(h, "recal_auroc", None)
        if rll is not None and h.rounds:
            xr = h.rounds[-1] + 1
            ax1.plot([h.rounds[-1], xr], [h.train_logloss[-1], rll], color=colors[pen],
                     ls="-", lw=0.8)
            ax1.plot(xr, rll, marker="*", ms=11, color=colors[pen],
                     markeredgecolor="k", markeredgewidth=0.4, zorder=5)
            ax2.plot(xr, rau, marker="*", ms=11, color=colors[pen],
                     markeredgecolor="k", markeredgewidth=0.4, zorder=5)
            have_recal = True
    ax1.set_xlabel("Communication round"); ax1.set_ylabel("Global training log-loss")
    ax2.set_xlabel("Communication round"); ax2.set_ylabel("Global training AUROC")
    handles, labels = ax1.get_legend_handles_labels()
    if have_recal:
        handles.append(Line2D([], [], color="grey", marker="*", ls="", ms=11,
                              markeredgecolor="k", markeredgewidth=0.4,
                              label="after intercept recalibration"))
        labels.append("after intercept recalibration")
    ax1.legend(handles, labels, fontsize=9)
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi)
    fig.savefig(outpath.replace(".png", ".pdf"))
    plt.close(fig)
