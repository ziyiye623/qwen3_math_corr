#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

TESTS = ["Z_L", "A_Z", "A_L", "A_Z_given_L_linear"]
LABELS = {
    "Z_L": "Corr(Z,L|X)",
    "A_Z": "Corr(A,Z|X)",
    "A_L": "Corr(A,L|X)",
    "A_Z_given_L_linear": "Corr(A,Z|X,L) linear",
}


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    args = ap.parse_args()
    root = Path(args.exp_dir)

    f = pd.read_csv(root / "fisher_summary_all_steps.csv").sort_values("checkpoint_step")
    fn = pd.read_csv(root / "fisher_summary_nontrunc_all_steps.csv").sort_values("checkpoint_step")
    s = pd.read_csv(root / "length_reward_sanity_all_steps.csv").sort_values("checkpoint_step")
    c = pd.read_csv(root / "prompt_level_corr_all_steps.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    for test in TESTS:
        q = f[f.test == test]
        ax.plot(q.checkpoint_step, q.median_pearson_corr, marker="o", label=LABELS[test])
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set(xlabel="Checkpoint step", ylabel="Median prompt-level Pearson correlation")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    save(fig, root / "corr_vs_step.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for test in TESTS:
        q = f[f.test == test]
        y = -np.log10(np.clip(q.fisher_pvalue.astype(float), 1e-300, 1))
        ax.plot(q.checkpoint_step, y, marker="o", label=LABELS[test])
    ax.axhline(-np.log10(.05), color="black", ls="--", lw=1, label="p=0.05")
    ax.set(xlabel="Checkpoint step", ylabel="-log10(Fisher combined p)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    save(fig, root / "fisher_pvalue_vs_step.png")

    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    metrics = [
        ("mean_response_length", "Mean response length"),
        ("median_response_length", "Median response length"),
        ("clip_ratio", "Clip ratio"),
        ("mean_reward", "Mean reward"),
        (None, "Valid prompt count (A_Z)"),
    ]
    for ax, (metric, title) in zip(axes.flat, metrics):
        if metric is None:
            q = f[f.test == "A_Z"]
            ax.plot(q.checkpoint_step, q.valid_prompt_count, marker="o")
        else:
            ax.plot(s.checkpoint_step, s[metric], marker="o")
        ax.set_title(title)
        ax.set_xlabel("Checkpoint step")
        ax.grid(alpha=.25)
    axes.flat[-1].axis("off")
    save(fig, root / "length_reward_sanity_vs_step.png")

    # New: all samples vs non-truncated samples correlation trajectories.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, test in zip(axes.flat, TESTS):
        qa = f[f.test == test].sort_values("checkpoint_step")
        qn = fn[fn.test == test].sort_values("checkpoint_step")
        ax.plot(qa.checkpoint_step, qa.median_pearson_corr, marker="o", label="all samples")
        ax.plot(qn.checkpoint_step, qn.median_pearson_corr, marker="o", label="non-truncated")
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.set_title(LABELS[test])
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel("Median Pearson correlation")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    save(fig, root / "corr_all_vs_nontrunc.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, test in zip(axes.flat, TESTS):
        qa = f[f.test == test].sort_values("checkpoint_step")
        qn = fn[fn.test == test].sort_values("checkpoint_step")
        ya = -np.log10(np.clip(qa.fisher_pvalue.astype(float), 1e-300, 1))
        yn = -np.log10(np.clip(qn.fisher_pvalue.astype(float), 1e-300, 1))
        ax.plot(qa.checkpoint_step, ya, marker="o", label="all samples")
        ax.plot(qn.checkpoint_step, yn, marker="o", label="non-truncated")
        ax.axhline(-np.log10(.05), color="black", ls="--", lw=1, label="p=0.05")
        ax.set_title(LABELS[test])
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel("-log10(Fisher combined p)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    save(fig, root / "fisher_all_vs_nontrunc.png")

    for step in sorted(c.checkpoint_step.unique()):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        for ax, test in zip(axes.flat, TESTS):
            q = c[(c.checkpoint_step == step) & (c.test == test)].copy()
            valid = q.is_valid.astype(str).str.lower().eq("true") & np.isfinite(q.pearson_corr)
            v = q[valid]
            ax.scatter(v.prompt_value, v.pearson_corr, alpha=.75)
            ax.axhline(0, color="black", ls="--", lw=1)
            value_corr = np.nan
            if len(v) >= 3 and v.prompt_value.nunique() > 1 and v.pearson_corr.nunique() > 1:
                value_corr = pearsonr(v.prompt_value, v.pearson_corr).statistic
                z = np.polyfit(v.prompt_value, v.pearson_corr, 1)
                x = np.linspace(v.prompt_value.min(), v.prompt_value.max(), 100)
                ax.plot(x, np.polyval(z, x), color="tab:red")
            fs = f[(f.checkpoint_step == step) & (f.test == test)].iloc[0]
            ax.set_title(
                f"{LABELS[test]}\nmedian={fs.median_pearson_corr:.3g}, "
                f"Fisher p={fs.fisher_pvalue:.3g}, n={len(v)}, corr(value,r)={value_corr:.3g}"
            )
            ax.set(xlabel="Prompt value (mean reward)", ylabel="Prompt-level correlation")
            ax.grid(alpha=.2)
        save(fig, root / f"prompt_value_step_{int(step)}.png")


if __name__ == "__main__":
    main()
