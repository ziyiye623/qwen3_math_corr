#!/usr/bin/env bash
set -euo pipefail

# New MATH pipeline; does not modify or reuse GSM8K output directories.
REPO=/mdr5/aifin/users/zhouhongyi/projects/verl_grpo_math500_custom
PY="$REPO/.venv/bin/python"
EXP=/mdr5/aifin/users/zhouhongyi/projects/qwen3_math_corr
BASE_MODEL=/ky200t/models/Qwen3-8B
DATA=/mdr5/aifin/users/zhouhongyi/projects/verl_grpo_math_artifacts/data/math/train.parquet
CKROOT=/ky200t/aifin/users/zhouhongyi/checkpoints/qwen3_8b_math_grpo_8096_n32_actormicro4_4gpu_a800_2000step
SELECTION="$EXP/selected_prompt_indices_seed42.json"
GEN_ENV="$EXP/math_rollout_generation.env"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export WANDB_DIR="$EXP/wandb"
mkdir -p "$WANDB_DIR"

# Refuse to guess generation settings. Create this file only after inspecting the actual MATH training Hydra config.
if [[ ! -f "$GEN_ENV" ]]; then
  cat >&2 <<EOF
Missing $GEN_ENV
Create it from the ACTUAL MATH training rollout config before smoke/formal runs. Required variables:
  MAX_NEW_TOKENS
  TEMPERATURE
  TOP_P
  TOP_K
  DO_SAMPLE=true|false
  CHAT_TEMPLATE_KWARGS_JSON='{}' (or exact training kwargs)
Optional:
  EOS_TOKEN_IDS_JSON='[...]' (only if training overrides model/tokenizer EOS ids)
EOF
  exit 2
fi
# shellcheck disable=SC1090
source "$GEN_ENV"
: "${MAX_NEW_TOKENS:?missing MAX_NEW_TOKENS}"
: "${TEMPERATURE:?missing TEMPERATURE}"
: "${TOP_P:?missing TOP_P}"
: "${TOP_K:?missing TOP_K}"
: "${DO_SAMPLE:?missing DO_SAMPLE}"
: "${CHAT_TEMPLATE_KWARGS_JSON:?missing CHAT_TEMPLATE_KWARGS_JSON}"
EOS_TOKEN_IDS_JSON=${EOS_TOKEN_IDS_JSON:-}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-4}

case "$DO_SAMPLE" in
  true|True|1) SAMPLE_FLAG=--do_sample ;;
  false|False|0) SAMPLE_FLAG=--no-do_sample ;;
  *) echo "DO_SAMPLE must be true or false, got: $DO_SAMPLE" >&2; exit 2 ;;
esac

mkdir -p "$EXP/merged_checkpoints"

ensure_selection() {
  if [[ -f "$SELECTION" ]]; then
    echo "[$(date -Is)] using existing selection: $SELECTION"
    return
  fi
  echo "[$(date -Is)] creating fixed MATH prompt selection seed=42: $SELECTION"
  "$PY" - "$DATA" "$SELECTION" <<'PY'
import json, random, sys
from pathlib import Path
import pandas as pd

data_path, out_path = sys.argv[1], sys.argv[2]
df = pd.read_parquet(data_path, columns=["data_source"])
rng = random.Random(42)
indices = rng.sample(range(len(df)), 32)
obj = {
    "seed": 42,
    "num_prompts": 32,
    "dataset_path": data_path,
    "selected_indices": indices,
    "data_source_counts": df.iloc[indices]["data_source"].astype(str).value_counts().to_dict(),
}
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
Path(out_path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
print(json.dumps(obj, indent=2))
PY
}

merge_step() {
  local gpu=$1 step=$2
  local src="$CKROOT/global_step_${step}/actor"
  local merged="$EXP/merged_checkpoints/step_${step}"
  if [[ -f "$merged/model.safetensors.index.json" ]]; then
    echo "[$(date -Is)] step $step already merged: $merged"
    return
  fi
  test -d "$src"
  echo "[$(date -Is)] GPU $gpu merging step $step from $src"
  rm -rf "$merged"; mkdir -p "$merged"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m verl.model_merger merge --backend fsdp \
    --local_dir "$src" --target_dir "$merged" --use_cpu_initialization \
    > "$EXP/merge_step_${step}.log" 2>&1
  test -f "$merged/config.json"
  test -f "$merged/model.safetensors.index.json"
}

model_path_for_step() {
  local step=$1
  if [[ "$step" == "0" ]]; then
    printf '%s\n' "$BASE_MODEL"
  else
    printf '%s\n' "$EXP/merged_checkpoints/step_${step}"
  fi
}

run_diag() {
  local gpu=$1 step=$2 np=$3 ng=$4 out=$5 wbmode=$6 gen_batch=$7
  local model_path
  if [[ "$step" != "0" ]]; then merge_step "$gpu" "$step"; fi
  model_path=$(model_path_for_step "$step")
  mkdir -p "$out"

  local eos_args=()
  if [[ -n "$EOS_TOKEN_IDS_JSON" ]]; then
    eos_args+=(--eos_token_ids_json "$EOS_TOKEN_IDS_JSON")
  fi

  echo "[$(date -Is)] GPU $gpu diagnostic step=$step np=$np ng=$ng model=$model_path"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EXP/standalone_qwen3_math_corr_alltrainable.py" \
    --model_path "$model_path" --tokenizer_path "$BASE_MODEL" --checkpoint_step "$step" \
    --data_path "$DATA" --selection_path "$SELECTION" --outdir "$out" \
    --num_prompts "$np" --num_generations "$ng" --generation_batch_size "$gen_batch" \
    --max_new_tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top_p "$TOP_P" --top_k "$TOP_K" \
    "$SAMPLE_FLAG" --chat_template_kwargs_json "$CHAT_TEMPLATE_KWARGS_JSON" "${eos_args[@]}" \
    --expected_data_source DigitalLearningGmbH/MATH-lighteval --seed 42 \
    --wandb_project qwen3_math_correlation --wandb_group alltrainable_np32_g32_seed42 --wandb_mode "$wbmode" \
    > "$out/run.log" 2>&1
  touch "$out/.complete"
  echo "[$(date -Is)] GPU $gpu completed step $step -> $out"
}

smoke() {
  ensure_selection
  local gpu=${SMOKE_GPU:-0}
  local out="$EXP/smoke_step_50/diagnostics"
  rm -f "$out/.complete"
  run_diag "$gpu" 50 2 2 "$out" disabled 2
  echo "===== smoke outputs ====="
  ls -lh "$out"
  "$PY" - "$out" <<'PY'
import sys, pandas as pd
from pathlib import Path
p=Path(sys.argv[1])
raw=pd.read_csv(p/'raw_score_norm_samples.csv')
print(raw[['prompt_id','generation_id','reward','response_length','hit_length_cap','score_norm_sq']])
print('\nall-sample correlation summary:')
print(pd.read_csv(p/'correlation_summary.csv'))
print('\nnon-truncated correlation summary:')
print(pd.read_csv(p/'correlation_summary_nontrunc.csv'))
print('\nsanity:')
print(pd.read_csv(p/'length_reward_sanity.csv').T)
PY
}

formal() {
  ensure_selection
  # Exactly one checkpoint per GPU: GPU0->Step0, GPU1->Step25, GPU2->Step50.
  local gpu0=${GPU_STEP0:-0} gpu25=${GPU_STEP25:-1} gpu50=${GPU_STEP50:-2}
  worker() {
    local gpu=$1 step=$2
    local out="$EXP/step_${step}/diagnostics"
    if [[ -f "$out/.complete" ]]; then
      echo "[$(date -Is)] GPU $gpu step $step complete; skipping"
      return
    fi
    run_diag "$gpu" "$step" 32 32 "$out" offline "$GENERATION_BATCH_SIZE"
  }
  worker "$gpu0" 0  > "$EXP/worker_gpu${gpu0}_step0.log"  2>&1 & p0=$!
  worker "$gpu25" 25 > "$EXP/worker_gpu${gpu25}_step25.log" 2>&1 & p25=$!
  worker "$gpu50" 50 > "$EXP/worker_gpu${gpu50}_step50.log" 2>&1 & p50=$!
  wait "$p0" "$p25" "$p50"
  "$PY" "$EXP/summarize_qwen3_math_corr.py" --exp_dir "$EXP"
  "$PY" "$EXP/plot_qwen3_math_corr.py" --exp_dir "$EXP"
  echo "[$(date -Is)] W&B upload deferred; offline runs kept under $WANDB_DIR"
  echo "[$(date -Is)] MATH step0/25/50 correlation pipeline complete"
}

case "${1:-}" in
  smoke) smoke ;;
  formal) formal ;;
  *) echo "usage: $0 {smoke|formal}" >&2; exit 2 ;;
esac
