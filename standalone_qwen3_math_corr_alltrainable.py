#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint as checkpoint_utils
from scipy.stats import chi2, pearsonr
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from verl.utils.reward_score import default_compute_score as verl_default_compute_score


RAW_FIELDS = [
    "checkpoint_step", "prompt_id", "source_index", "generation_id", "data_source", "prompt", "response",
    "prompt_token_ids", "response_token_ids", "reward", "response_length", "generated_tensor_len",
    "has_eos", "eos_pos", "stop_token_id", "hit_length_cap", "score_norm_sq", "backward_seconds",
]
TESTS = ("Z_L", "A_Z", "A_L", "A_Z_given_L_linear")


def json_dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def append_csv_row(path, row, fields):
    """Crash-safe append: write one completed response at a time without rewriting prior rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if need_header:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finite_float(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def load_selected_prompts(data_path, selection_path, num_prompts, expected_data_source=None):
    df = pd.read_parquet(data_path)
    selected = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    indices = selected["selected_indices"][:num_prompts]
    if len(indices) < num_prompts:
        raise ValueError(f"selection file only has {len(indices)} indices, need {num_prompts}")
    rows = []
    for pid, idx in enumerate(indices):
        row = df.iloc[int(idx)]
        reward_model = row["reward_model"]
        ground_truth = reward_model["ground_truth"] if isinstance(reward_model, dict) else reward_model
        data_source = str(row["data_source"])
        if expected_data_source and data_source != expected_data_source:
            raise ValueError(f"unexpected data_source at index {idx}: {data_source!r} != {expected_data_source!r}")
        messages = row["prompt"].tolist() if hasattr(row["prompt"], "tolist") else row["prompt"]
        rows.append({
            "prompt_id": pid,
            "source_index": int(idx),
            "messages": messages,
            "ground_truth": str(ground_truth),
            "data_source": data_source,
            "level": str(row["level"]) if "level" in row.index else "",
            "problem_type": str(row["type"]) if "type" in row.index else "",
        })
    return rows


def generation_eos_ids(model_path, tokenizer, eos_token_ids_json=None):
    if eos_token_ids_json:
        value = json.loads(eos_token_ids_json)
        if not isinstance(value, list):
            value = [value]
        return sorted({int(x) for x in value if x is not None})
    ids = []
    try:
        value = GenerationConfig.from_pretrained(model_path).eos_token_id
        ids.extend(value if isinstance(value, list) else [value])
    except Exception:
        pass
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    return sorted({int(x) for x in ids if x is not None})


def truncate_generated(ids, eos_ids, pad_id):
    stop_ids = set(eos_ids)
    if pad_id is not None:
        stop_ids.add(int(pad_id))
    first = None
    for pos, token in enumerate(ids):
        if int(token) in stop_ids:
            first = (pos, int(token))
            break
    if first is None:
        return list(map(int, ids)), False, -1, None
    pos, token = first
    return list(map(int, ids[:pos])), token in set(eos_ids), pos, token


def generate_many(model, prompt_ids, tokenizer, args, eos_ids, prompt_id):
    """Generate in small chunks to avoid 32 x long-response KV-cache OOM.

    The sampling distribution/config is unchanged; generation_batch_size is only an execution/memory knob.
    Seeds are deterministic by prompt and chunk and therefore identical across checkpoints.
    """
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    batch_size = max(1, min(args.generation_batch_size, args.num_generations))
    results = []
    for start_gid in range(0, args.num_generations, batch_size):
        n = min(batch_size, args.num_generations - start_gid)
        response_seed = args.seed + prompt_id * 100000 + start_gid
        torch.manual_seed(response_seed)
        torch.cuda.manual_seed_all(response_seed)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                # VERL/vLLM uses -1 to disable top-k; Transformers uses 0.
                top_k=0 if args.top_k < 0 else args.top_k,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=n,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        for output in outputs:
            generated = output[len(prompt_ids):].detach().cpu().tolist()
            active, has_eos, eos_pos, stop_id = truncate_generated(generated, eos_ids, tokenizer.pad_token_id)
            results.append({
                "response_ids": active,
                "generated_tensor_len": len(generated),
                "has_eos": bool(has_eos),
                "eos_pos": int(eos_pos),
                "stop_token_id": stop_id,
                "hit_length_cap": len(generated) >= args.max_new_tokens and eos_pos < 0,
            })
        del outputs
        torch.cuda.empty_cache()
    return results


def score_norm_sq(
    model,
    prompt_ids,
    response_ids,
    ce_chunk_tokens=128,
    grad_chunk_elems=4_000_000,
):
    """
    Exact:
        || grad_theta sum_t log pi_theta(y_t | x, y_<t) ||_2^2

    Memory optimizations only:
      1. compute FP32 cross entropy in token chunks;
      2. checkpoint CE chunks so their large FP32 intermediates are not all
         retained simultaneously;
      3. accumulate each parameter gradient norm immediately after that
         gradient is produced, then clear param.grad;
      4. accumulate very large parameter gradients in element chunks.

    The objective and gradient definition are unchanged.
    """
    if response_ids is None or len(response_ids) == 0:
        return float("nan")

    prompt_ids = [int(x) for x in prompt_ids]
    response_ids = [int(x) for x in response_ids]

    if len(prompt_ids) == 0:
        raise ValueError("prompt_ids must be non-empty")

    device = next(model.parameters()).device

    prompt_len = len(prompt_ids)
    response_len = len(response_ids)

    full_ids = torch.tensor(
        [prompt_ids + response_ids],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones_like(full_ids)

    # The response tokens themselves are the CE targets.
    targets = full_ids[
        :,
        prompt_len : prompt_len + response_len
    ]

    model.zero_grad(set_to_none=True)

    # Qwen3 GradientCheckpointingLayer only checkpoints when its own
    # training flag is True. The diagnostic model stays globally in eval()
    # so generation behavior is unchanged; only these checkpointing wrapper
    # layers are temporarily activated during the score-norm backward.
    model.gradient_checkpointing_enable()

    gc_layers = []
    gc_training_states = []

    for module in model.modules():
        if isinstance(module, GradientCheckpointingLayer):
            gc_layers.append(module)
            gc_training_states.append(module.training)
            module.training = True

    # GPU scalar only.
    grad_norm_sq = torch.zeros(
        (),
        dtype=torch.float32,
        device=device,
    )

    handles = []

    def _accumulate_grad_and_clear(param):
        grad = param.grad

        if grad is None:
            return

        g = grad.detach().reshape(-1)

        # Avoid a full FP32 copy for giant matrices such as embeddings.
        for j in range(0, g.numel(), grad_chunk_elems):
            chunk = g[j : j + grad_chunk_elems]
            grad_norm_sq.add_(
                chunk.float().square().sum()
            )

        # We only need ||grad||^2, not the gradient itself.
        param.grad = None

    for param in model.parameters():
        if param.requires_grad:
            handles.append(
                param.register_post_accumulate_grad_hook(
                    _accumulate_grad_and_clear
                )
            )

    try:
        # IMPORTANT:
        # No labels=... here. Transformers' built-in CausalLM loss casts the
        # entire [sequence, vocabulary] logits tensor to FP32, which is the
        # source of the 4+ GiB allocation seen for long MATH responses.
        out = model(
            input_ids=full_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        # Causal alignment:
        # response token at index prompt_len+i is predicted by
        # logits at index prompt_len+i-1.
        response_logits = out.logits[
            :,
            prompt_len - 1 :
            prompt_len - 1 + response_len,
            :
        ]

        def _chunk_logprob_sum(logits_chunk, target_chunk):
            vocab_size = logits_chunk.shape[-1]

            # Same FP32 CE convention as HF CausalLM loss, but only for
            # ce_chunk_tokens positions at a time.
            return -torch.nn.functional.cross_entropy(
                logits_chunk.float().reshape(-1, vocab_size),
                target_chunk.reshape(-1),
                reduction="sum",
            )

        pieces = []

        for st in range(0, response_len, ce_chunk_tokens):
            ed = min(st + ce_chunk_tokens, response_len)

            piece = checkpoint_utils.checkpoint(
                _chunk_logprob_sum,
                response_logits[:, st:ed, :],
                targets[:, st:ed],
                use_reentrant=False,
            )

            pieces.append(piece)

        logprob_sum = torch.stack(pieces).sum()

        # Produces exactly grad(sum response log-probabilities).
        logprob_sum.backward()

        value = float(grad_norm_sq.item())

        # Release large references before empty_cache().
        del logprob_sum
        del pieces
        del response_logits
        del out

    finally:
        for handle in handles:
            handle.remove()

        for module, old_training in zip(
            gc_layers,
            gc_training_states,
        ):
            module.training = old_training

        model.zero_grad(set_to_none=True)

    del full_ids
    del attention_mask
    del targets
    del grad_norm_sq

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return value


def invalid_stats(reason, n, reward, length, score):
    return {
        "pearson_corr": float("nan"), "pearson_pvalue": float("nan"), "num_samples": n,
        "reward_mean": finite_float(np.mean(reward)) if n else float("nan"),
        "reward_std": finite_float(np.std(reward)) if n else float("nan"),
        "length_mean": finite_float(np.mean(length)) if n else float("nan"),
        "length_std": finite_float(np.std(length)) if n else float("nan"),
        "score_mean": finite_float(np.mean(score)) if n else float("nan"),
        "score_std": finite_float(np.std(score)) if n else float("nan"),
        "is_valid": False, "invalid_reason": reason,
    }


def corr_for_test(test, reward, length, score):
    z = np.asarray(reward, float)
    l = np.asarray(length, float)
    a = np.asarray(score, float)
    n = len(z)
    base = invalid_stats("", n, z, l, a)
    if n < 3:
        return invalid_stats("too_few_samples", n, z, l, a)
    if not (np.isfinite(z).all() and np.isfinite(l).all() and np.isfinite(a).all()):
        return invalid_stats("nan_or_inf", n, z, l, a)
    if test == "Z_L":
        if np.ptp(z) == 0:
            return invalid_stats("constant_reward", n, z, l, a)
        if np.ptp(l) == 0:
            return invalid_stats("constant_length", n, z, l, a)
        x, y = z, l
    elif test == "A_Z":
        if np.ptp(a) == 0:
            return invalid_stats("constant_score", n, z, l, a)
        if np.ptp(z) == 0:
            return invalid_stats("constant_reward", n, z, l, a)
        x, y = a, z
    elif test == "A_L":
        if np.ptp(a) == 0:
            return invalid_stats("constant_score", n, z, l, a)
        if np.ptp(l) == 0:
            return invalid_stats("constant_length", n, z, l, a)
        x, y = a, l
    elif test == "A_Z_given_L_linear":
        if np.ptp(a) == 0:
            return invalid_stats("constant_score", n, z, l, a)
        if np.ptp(z) == 0:
            return invalid_stats("constant_reward", n, z, l, a)
        if np.ptp(l) == 0:
            return invalid_stats("constant_length", n, z, l, a)
        X = np.column_stack([np.ones(n), l])
        x = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
        y = z - X @ np.linalg.lstsq(X, z, rcond=None)[0]
    else:
        raise ValueError(test)
    if np.std(x) == 0:
        return invalid_stats("constant_x", n, z, l, a)
    if np.std(y) == 0:
        return invalid_stats("constant_y", n, z, l, a)
    result = pearsonr(x, y)
    base.update({
        "pearson_corr": float(result.statistic), "pearson_pvalue": float(result.pvalue),
        "is_valid": True, "invalid_reason": "",
    })
    return base


def summarize_prompt_correlations(rows):
    output = []
    for test in TESTS:
        subset = [x for x in rows if x["test"] == test]
        valid = [x for x in subset if x["is_valid"] and math.isfinite(float(x["pearson_corr"]))]
        corr = np.asarray([x["pearson_corr"] for x in valid], float)
        pvals = np.asarray([x["pearson_pvalue"] for x in valid if 0 < x["pearson_pvalue"] <= 1], float)
        stat = float(-2 * np.log(pvals).sum()) if len(pvals) else float("nan")
        fisher = float(chi2.sf(stat, 2 * len(pvals))) if len(pvals) else float("nan")
        output.append({
            "test": test,
            "median_pearson_corr": float(np.median(corr)) if len(corr) else float("nan"),
            "fisher_pvalue": fisher,
            "fisher_stat": stat,
            "valid_prompt_count": len(valid),
            "positive_prompt_count": int((corr > 0).sum()),
            "negative_prompt_count": int((corr < 0).sum()),
            "p_lt_005_prompt_count": int(sum(x["pearson_pvalue"] < 0.05 for x in valid)),
        })
    return output


def build_corr_rows(raw, prompts, checkpoint_step, nontruncated=False):
    rows = []
    for prompt in prompts:
        subset = [x for x in raw if int(x["prompt_id"]) == prompt["prompt_id"]]
        if nontruncated:
            subset = [x for x in subset if not bool(x["hit_length_cap"])]
        z = [float(x["reward"]) for x in subset]
        l = [float(x["response_length"]) for x in subset]
        a = [float(x["score_norm_sq"]) for x in subset]
        prompt_value = float(np.mean(z)) if z else float("nan")
        for test in TESTS:
            rows.append({
                "checkpoint_step": checkpoint_step,
                "prompt_id": prompt["prompt_id"],
                "source_index": prompt["source_index"],
                "test": test,
                "prompt_value": prompt_value,
                "sample_scope": "nontruncated" if nontruncated else "all",
                **corr_for_test(test, z, l, a),
            })
    return rows


def prompt_reward_variance_rows(raw, prompts, checkpoint_step):
    rows = []
    for prompt in prompts:
        subset = [x for x in raw if int(x["prompt_id"]) == prompt["prompt_id"]]
        z = np.asarray([float(x["reward"]) for x in subset], float)
        variance = float(np.var(z)) if len(z) else float("nan")
        rows.append({
            "checkpoint_step": checkpoint_step,
            "prompt_id": prompt["prompt_id"],
            "source_index": prompt["source_index"],
            "num_samples": len(z),
            "reward_mean": float(np.mean(z)) if len(z) else float("nan"),
            "reward_variance": variance,
            "zero_reward_variance": bool(len(z) > 0 and np.ptp(z) == 0),
        })
    return rows


def load_existing_raw(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    df = pd.read_csv(path)
    rows = df.to_dict("records")
    for row in rows:
        for key in ("has_eos", "hit_length_cap"):
            value = row.get(key)
            if isinstance(value, str):
                row[key] = value.strip().lower() == "true"
            else:
                row[key] = bool(value)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--checkpoint_step", type=int, required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--selection_path", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--num_prompts", type=int, required=True)
    ap.add_argument("--num_generations", type=int, required=True)
    ap.add_argument("--generation_batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--top_p", type=float, required=True)
    ap.add_argument("--top_k", type=int, required=True)
    ap.add_argument("--do_sample", action=argparse.BooleanOptionalAction, required=True)
    ap.add_argument("--chat_template_kwargs_json", default="{}")
    ap.add_argument("--eos_token_ids_json", default=None)
    ap.add_argument("--expected_data_source", default="DigitalLearningGmbH/MATH-lighteval")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb_project", default="qwen3_math_correlation")
    ap.add_argument("--wandb_group", default="alltrainable_np32_g32_seed42")
    ap.add_argument("--wandb_mode", default="online")
    args = ap.parse_args()

    start = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    chat_template_kwargs = json.loads(args.chat_template_kwargs_json)
    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_ids = generation_eos_ids(tokenizer_path, tokenizer, args.eos_token_ids_json)
    prompts = load_selected_prompts(
        args.data_path, args.selection_path, args.num_prompts, expected_data_source=args.expected_data_source
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).cuda()
    model.eval()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(True)
    torch.cuda.reset_peak_memory_stats()

    config = {
        **vars(args),
        "base_model": AutoConfig.from_pretrained(args.model_path).name_or_path,
        "region": "all_trainable",
        "dataset_split": "train",
        "selected_prompt_indices": [p["source_index"] for p in prompts],
        "selected_prompt_data_sources": sorted({p["data_source"] for p in prompts}),
        "eos_token_id": eos_ids,
        "pad_token_id": tokenizer.pad_token_id,
        "chat_template_kwargs": chat_template_kwargs,
        "reward_implementation": "verl.utils.reward_score.default_compute_score(data_source='DigitalLearningGmbH/MATH-lighteval') -> math_reward.compute_score",
        "score_norm_definition": "FP32 sum over all trainable parameter gradient squares of sum(response token log-probs); first EOS/PAD excluded",
        "script_hash": sha256_file(__file__),
        "gpu_model": torch.cuda.get_device_name(0),
    }
    json_dump(outdir / "resolved_config.json", config)
    json_dump(outdir / "generation_config.json", {
        k: config[k] for k in [
            "max_new_tokens", "temperature", "top_p", "top_k", "do_sample", "seed",
            "eos_token_id", "pad_token_id", "chat_template_kwargs", "generation_batch_size",
        ]
    })
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    json_dump(outdir / "diag_param_groups.json", {
        "all_trainable": {
            "num_tensors": len(params),
            "num_scalar_params": sum(p.numel() for _, p in params),
            "parameter_names": [n for n, _ in params],
        }
    })

    wb = None
    if args.wandb_mode != "disabled":
        import wandb
        wb = wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=f"step_{args.checkpoint_step}",
            config=config,
            mode=args.wandb_mode,
        )

    raw_path = outdir / "raw_score_norm_samples.csv"
    raw = load_existing_raw(raw_path)
    completed = {(int(x["prompt_id"]), int(x["generation_id"])) for x in raw}
    if raw:
        print(json.dumps({"status": "resume", "existing_raw_rows": len(raw)}), flush=True)

    for prompt in prompts:
        needed_gids = [gid for gid in range(args.num_generations) if (prompt["prompt_id"], gid) not in completed]
        if not needed_gids:
            print(json.dumps({"prompt_id": prompt["prompt_id"], "status": "already_complete"}), flush=True)
            continue
        prompt_ids = tokenizer.apply_chat_template(
            prompt["messages"], tokenize=True, add_generation_prompt=True, **chat_template_kwargs
        )
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        generated_batch = generate_many(model, prompt_ids, tokenizer, args, eos_ids, prompt["prompt_id"])
        for gid, generated in enumerate(generated_batch):
            if gid not in needed_gids:
                continue
            response_ids = generated["response_ids"]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            reward = float(verl_default_compute_score(
                data_source=prompt["data_source"],
                solution_str=response,
                ground_truth=prompt["ground_truth"],
            ))
            t0 = time.time()
            norm = score_norm_sq(model, prompt_ids, response_ids)
            backward_seconds = time.time() - t0
            row = {
                "checkpoint_step": args.checkpoint_step,
                "prompt_id": prompt["prompt_id"],
                "source_index": prompt["source_index"],
                "generation_id": gid,
                "data_source": prompt["data_source"],
                "prompt": prompt_text,
                "response": response,
                "prompt_token_ids": json.dumps(prompt_ids),
                "response_token_ids": json.dumps(response_ids),
                "reward": reward,
                "response_length": len(response_ids),
                "generated_tensor_len": generated["generated_tensor_len"],
                "has_eos": generated["has_eos"],
                "eos_pos": generated["eos_pos"],
                "stop_token_id": generated["stop_token_id"],
                "hit_length_cap": generated["hit_length_cap"],
                "score_norm_sq": norm,
                "backward_seconds": backward_seconds,
            }
            append_csv_row(raw_path, row, RAW_FIELDS)
            raw.append(row)
            completed.add((prompt["prompt_id"], gid))
            print(json.dumps({k: row[k] for k in [
                "prompt_id", "generation_id", "reward", "response_length", "has_eos",
                "hit_length_cap", "score_norm_sq", "backward_seconds",
            ]}), flush=True)

    # Keep deterministic row order after crash/resume appends.
    raw = sorted(raw, key=lambda x: (int(x["prompt_id"]), int(x["generation_id"])))

    corr_rows = build_corr_rows(raw, prompts, args.checkpoint_step, nontruncated=False)
    corr_nontrunc_rows = build_corr_rows(raw, prompts, args.checkpoint_step, nontruncated=True)
    write_csv(outdir / "prompt_level_corr.csv", corr_rows)
    write_csv(outdir / "prompt_level_corr_nontrunc.csv", corr_nontrunc_rows)
    summary = summarize_prompt_correlations(corr_rows)
    summary_nontrunc = summarize_prompt_correlations(corr_nontrunc_rows)
    write_csv(outdir / "correlation_summary.csv", summary)
    write_csv(outdir / "correlation_summary_nontrunc.csv", summary_nontrunc)

    reward_var_rows = prompt_reward_variance_rows(raw, prompts, args.checkpoint_step)
    write_csv(outdir / "prompt_reward_variance.csv", reward_var_rows)
    zero_var_count = sum(bool(x["zero_reward_variance"]) for x in reward_var_rows)

    lengths = np.asarray([float(x["response_length"]) for x in raw], float)
    rewards = np.asarray([float(x["reward"]) for x in raw], float)
    hit_cap = np.asarray([bool(x["hit_length_cap"]) for x in raw], bool)
    nontrunc_rewards = rewards[~hit_cap]
    trunc_rewards = rewards[hit_cap]
    sanity = [{
        "checkpoint_step": args.checkpoint_step,
        "total_sample_count": len(raw),
        "nontruncated_sample_count": int((~hit_cap).sum()),
        "truncated_sample_count": int(hit_cap.sum()),
        "mean_response_length": float(lengths.mean()),
        "median_response_length": float(np.median(lengths)),
        "max_response_length": int(lengths.max()),
        "clip_ratio": float(hit_cap.mean()),
        "eos_stop_ratio": float(np.mean([bool(x["has_eos"]) for x in raw])),
        "mean_reward": float(rewards.mean()),
        "mean_reward_truncated": float(trunc_rewards.mean()) if len(trunc_rewards) else float("nan"),
        "mean_reward_nontruncated": float(nontrunc_rewards.mean()) if len(nontrunc_rewards) else float("nan"),
        "reward_variance": float(rewards.var()),
        "zero_reward_variance_prompt_count": int(zero_var_count),
        "zero_reward_variance_prompt_ratio": float(zero_var_count / len(prompts)) if prompts else float("nan"),
        "runtime_seconds": time.time() - start,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }]
    write_csv(outdir / "length_reward_sanity.csv", sanity)

    if wb:
        metrics = {**sanity[0]}
        metrics.update({f"all/{x['test']}/median_corr": x["median_pearson_corr"] for x in summary})
        metrics.update({f"nontrunc/{x['test']}/median_corr": x["median_pearson_corr"] for x in summary_nontrunc})
        wb.log(metrics)
        artifact = __import__("wandb").Artifact(f"step_{args.checkpoint_step}_diagnostics", type="diagnostics")
        artifact.add_dir(str(outdir))
        wb.log_artifact(artifact)
        wb.finish()

    print(json.dumps({
        "status": "complete",
        "sanity": sanity[0],
        "summary": summary,
        "summary_nontrunc": summary_nontrunc,
    }), flush=True)


if __name__ == "__main__":
    main()
