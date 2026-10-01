#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

STEPS = [0, 25, 50]
TESTS = ["Z_L", "A_Z", "A_L", "A_Z_given_L_linear"]


def flatten_summary(fisher_df, prefix=""):
    out = {}
    for test in TESTS:
        q = fisher_df[fisher_df["test"] == test]
        if q.empty:
            continue
        r = q.iloc[0]
        stem = f"{prefix}{test}"
        out[f"median_corr__{stem}"] = r["median_pearson_corr"]
        out[f"fisher_pvalue__{stem}"] = r["fisher_pvalue"]
        out[f"valid_prompt_count__{stem}"] = r["valid_prompt_count"]
        out[f"positive_prompt_count__{stem}"] = r["positive_prompt_count"]
        out[f"negative_prompt_count__{stem}"] = r["negative_prompt_count"]
        out[f"p_lt_005_prompt_count__{stem}"] = r["p_lt_005_prompt_count"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    root = Path(args.exp_dir)

    corr_all, corr_nontrunc = [], []
    fisher_all, fisher_nontrunc = [], []
    sanity_all, reward_var_all = [], []
    rows_all_summary, rows_nontrunc_summary = [], []

    for step in STEPS:
        d = root / f"step_{step}" / "diagnostics"
        if not (d / ".complete").exists():
            continue
        required = [
            "prompt_level_corr.csv", "prompt_level_corr_nontrunc.csv",
            "correlation_summary.csv", "correlation_summary_nontrunc.csv",
            "length_reward_sanity.csv", "prompt_reward_variance.csv",
        ]
        missing = [name for name in required if not (d / name).exists()]
        if missing:
            raise SystemExit(f"step {step} marked complete but missing: {missing}")

        ca = pd.read_csv(d / "prompt_level_corr.csv"); ca["checkpoint_step"] = step; corr_all.append(ca)
        cn = pd.read_csv(d / "prompt_level_corr_nontrunc.csv"); cn["checkpoint_step"] = step; corr_nontrunc.append(cn)
        fa = pd.read_csv(d / "correlation_summary.csv"); fa["checkpoint_step"] = step; fisher_all.append(fa)
        fn = pd.read_csv(d / "correlation_summary_nontrunc.csv"); fn["checkpoint_step"] = step; fisher_nontrunc.append(fn)
        sa = pd.read_csv(d / "length_reward_sanity.csv"); sa["checkpoint_step"] = step; sanity_all.append(sa)
        rv = pd.read_csv(d / "prompt_reward_variance.csv"); rv["checkpoint_step"] = step; reward_var_all.append(rv)

        s = sa.iloc[0].to_dict()
        all_row = {
            "Step": step,
            "Mean L": s.get("mean_response_length", np.nan),
            "Median L": s.get("median_response_length", np.nan),
            "Clip ratio": s.get("clip_ratio", np.nan),
            "Mean reward": s.get("mean_reward", np.nan),
            "Mean reward | truncated": s.get("mean_reward_truncated", np.nan),
            "Mean reward | non-truncated": s.get("mean_reward_nontruncated", np.nan),
            "Zero-reward-var prompt count": s.get("zero_reward_variance_prompt_count", np.nan),
            "Zero-reward-var prompt ratio": s.get("zero_reward_variance_prompt_ratio", np.nan),
            "Total N": s.get("total_sample_count", np.nan),
        }
        all_row.update(flatten_summary(fa))
        # User-facing shorthand: use A_Z valid prompts as the main valid-prompt diagnostic.
        all_row["Valid prompts"] = all_row.get("valid_prompt_count__A_Z", np.nan)
        rows_all_summary.append(all_row)

        non_row = {
            "Step": step,
            "N non-truncated": s.get("nontruncated_sample_count", np.nan),
            "Mean reward non-truncated": s.get("mean_reward_nontruncated", np.nan),
        }
        non_row.update(flatten_summary(fn, prefix="nontrunc__"))
        non_row["Valid prompts"] = non_row.get("valid_prompt_count__nontrunc__A_Z", np.nan)
        rows_nontrunc_summary.append(non_row)

    if not corr_all:
        raise SystemExit("no completed checkpoint diagnostics")

    corr_all_df = pd.concat(corr_all, ignore_index=True)
    corr_nontrunc_df = pd.concat(corr_nontrunc, ignore_index=True)
    fisher_all_df = pd.concat(fisher_all, ignore_index=True)
    fisher_nontrunc_df = pd.concat(fisher_nontrunc, ignore_index=True)
    sanity_df = pd.concat(sanity_all, ignore_index=True)
    reward_var_df = pd.concat(reward_var_all, ignore_index=True)

    corr_all_df.to_csv(root / "prompt_level_corr_all_steps.csv", index=False)
    corr_nontrunc_df.to_csv(root / "prompt_level_corr_nontrunc_all_steps.csv", index=False)
    fisher_all_df.to_csv(root / "fisher_summary_all_steps.csv", index=False)
    fisher_nontrunc_df.to_csv(root / "fisher_summary_nontrunc_all_steps.csv", index=False)
    sanity_df.to_csv(root / "length_reward_sanity_all_steps.csv", index=False)
    reward_var_df.to_csv(root / "prompt_reward_variance_all_steps.csv", index=False)

    pv = corr_all_df[["checkpoint_step", "prompt_id", "source_index", "prompt_value"]].drop_duplicates()
    pv.to_csv(root / "prompt_value_all_steps.csv", index=False)

    all_summary = pd.DataFrame(rows_all_summary).sort_values("Step")
    non_summary = pd.DataFrame(rows_nontrunc_summary).sort_values("Step")
    all_summary.to_csv(root / "summary_all_samples.csv", index=False)
    non_summary.to_csv(root / "summary_nontruncated.csv", index=False)

    wide_all = fisher_all_df.pivot(
        index="checkpoint_step", columns="test",
        values=["median_pearson_corr", "fisher_pvalue", "valid_prompt_count"]
    )
    wide_all.columns = [f"{metric}__{test}" for metric, test in wide_all.columns]
    wide_non = fisher_nontrunc_df.pivot(
        index="checkpoint_step", columns="test",
        values=["median_pearson_corr", "fisher_pvalue", "valid_prompt_count"]
    )
    wide_non.columns = [f"{metric}__{test}__nontrunc" for metric, test in wide_non.columns]
    master = sanity_df.merge(wide_all.reset_index(), on="checkpoint_step", how="left")
    master = master.merge(wide_non.reset_index(), on="checkpoint_step", how="left")
    master.to_csv(root / "checkpoint_summary_master.csv", index=False)
    sanity_df[["checkpoint_step", "runtime_seconds", "peak_memory_gib"]].to_csv(
        root / "runtime_memory_summary.csv", index=False
    )

    if args.wandb:
        import wandb
        run = wandb.init(
            project="qwen3_math_correlation",
            group="alltrainable_np32_g32_seed42",
            name="across_checkpoint_summary",
            job_type="summary",
            config={"completed_steps": sorted(master.checkpoint_step.astype(int).tolist())},
        )
        artifact = wandb.Artifact("across_checkpoint_summary", type="summary")
        names = [
            "checkpoint_summary_master.csv", "summary_all_samples.csv", "summary_nontruncated.csv",
            "prompt_level_corr_all_steps.csv", "prompt_level_corr_nontrunc_all_steps.csv",
            "prompt_value_all_steps.csv", "length_reward_sanity_all_steps.csv",
            "prompt_reward_variance_all_steps.csv", "fisher_summary_all_steps.csv",
            "fisher_summary_nontrunc_all_steps.csv", "runtime_memory_summary.csv",
            "corr_vs_step.png", "fisher_pvalue_vs_step.png", "length_reward_sanity_vs_step.png",
            "corr_all_vs_nontrunc.png", "fisher_all_vs_nontrunc.png",
        ] + [f"prompt_value_step_{s}.png" for s in STEPS]
        for name in names:
            p = root / name
            if p.exists():
                artifact.add_file(str(p), name=name)
        run.log({"completed_checkpoint_count": len(master)})
        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
