#!/usr/bin/env python3
"""Generate publication-ready overview figures from recomputed CSV metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ["jev-1.13", "kev-0.8b", "kev-4b", "kev-9b"]
DISPLAY = {"jev-1.13": "JEV 1.13", "kev-0.8b": "KEV-0.8B", "kev-4b": "KEV-4B", "kev-9b": "KEV-9B"}
COLORS = {"jev-1.13": "#F94144", "kev-0.8b": "#277DA1", "kev-4b": "#43AA8B", "kev-9b": "#F8961E"}


def save(fig, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main_heatmap(metrics: Path, output: Path) -> None:
    frame = pd.read_csv(metrics / "main_by_dataset.csv")
    matrix = frame.pivot(index="source", columns="model", values="support_deviation")[MODEL_ORDER] * 100
    ordinal = frame.drop_duplicates("source").set_index("source")["task"]
    matrix = matrix.loc[sorted(matrix.index, key=lambda s: (ordinal[s] != "ordinal", matrix.loc[s].mean()))]
    fig, ax = plt.subplots(figsize=(7.2, 10.5))
    sns.heatmap(matrix.rename(columns=DISPLAY), cmap="magma_r", vmin=0,
                annot=True, fmt=".0f", annot_kws={"fontsize": 6.5}, linewidths=.3,
                cbar_kws={"label": "Support deviation $D_R$ (pp; lower is better)"}, ax=ax)
    ax.set(xlabel="", ylabel="", title="Decision-support bias across 40 datasets")
    ax.tick_params(axis="y", labelsize=7)
    save(fig, output, "main40_dr_heatmap")


def position_figure(metrics: Path, output: Path) -> None:
    frame = pd.read_csv(metrics / "position_by_dataset.csv")
    aggregate = frame.groupby(["model", "task"])[["original_d_r", "randomized_d_r", "pairwise_agreement"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    width = .18; x = np.arange(2)
    for index, model in enumerate(MODEL_ORDER):
        subset = aggregate[aggregate.model == model].set_index("task")
        axes[0].bar(x + (index - 1.5) * width, [subset.loc[t, "original_d_r"] * 100 for t in ["ordinal", "nominal"]],
                    width, color=COLORS[model], alpha=.35, edgecolor=COLORS[model])
        axes[0].bar(x + (index - 1.5) * width, [subset.loc[t, "randomized_d_r"] * 100 for t in ["ordinal", "nominal"]],
                    width, facecolor="none", edgecolor=COLORS[model], linewidth=1.8, label=DISPLAY[model])
        axes[1].bar(x + (index - 1.5) * width, [subset.loc[t, "pairwise_agreement"] * 100 for t in ["ordinal", "nominal"]],
                    width, color=COLORS[model], label=DISPLAY[model])
    axes[0].set(title="Bias before/after randomization", ylabel="$D_R$ (pp; lower is better)", xticks=x,
                xticklabels=["Ordinal", "Nominal"])
    axes[1].set(title="Agreement across two random orders", ylabel="Pairwise agreement (%)", xticks=x,
                xticklabels=["Ordinal", "Nominal"], ylim=(0, 105))
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    save(fig, output, "position_intervention")


def kcurve_figure(metrics: Path, output: Path) -> None:
    frame = pd.read_csv(metrics / "kcurve_by_condition.csv")
    frame = frame[(frame.source != "dbpedia14") & (frame.binning == "equal_frequency")]
    aggregate = frame.groupby(["model", "k"])[["support_deviation", "u_soft", "margin"]].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.45), constrained_layout=True)
    specs = [("support_deviation", "$D_R$ (pp; lower is better)", "Decision-support bias"),
             ("u_soft", "Soft utilization (%)", "Probability-space coverage"),
             ("margin", "Top-1 margin (pp)", "Decision margin")]
    for ax, (metric, ylabel, title) in zip(axes, specs):
        for model in MODEL_ORDER:
            subset = aggregate[aggregate.model == model].sort_values("k")
            ax.plot(subset.k, subset[metric] * 100, marker="o", ms=4.2, lw=2.1,
                    color=COLORS[model], label=DISPLAY[model])
        ax.set(title=title, xlabel="Number of candidates K", ylabel=ylabel,
               xticks=[2, 4, 6, 8, 10, 12, 14])
        ax.grid(alpha=.22)
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, output, "controlled_k_curve")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=ROOT / "outputs" / "metrics")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "figures")
    args = parser.parse_args()
    sns.set_theme(style="whitegrid", context="paper")
    main_heatmap(args.metrics, args.output)
    position_figure(args.metrics, args.output)
    kcurve_figure(args.metrics, args.output)
    print(f"Wrote figures to {args.output}")


if __name__ == "__main__":
    main()
