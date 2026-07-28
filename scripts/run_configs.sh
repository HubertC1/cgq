#!/bin/bash
# General-purpose launcher for --exp_config-driven runs (train + value-map render). Two modes:
#
#   chain      -- run configs one at a time, each gets the full GPU. For a long sequence of runs
#                 where each needs the whole card (e.g. large batch size / big networks).
#   concurrent -- run configs N at a time on one GPU, sharing it via XLA_PYTHON_CLIENT_MEM_FRACTION
#                 (each process capped to that fraction of GPU memory). For many small/cheap runs
#                 that individually don't need the whole card.
#
# Usage:
#   scripts/run_configs.sh chain <gpu_id> <config1> [config2 ...]
#   scripts/run_configs.sh concurrent <gpu_id> <batch_size> <mem_fraction> <config1> [config2 ...]
#
# Examples:
#   scripts/run_configs.sh chain 0 \
#     giant_stitch_sigma0.0_task4_iql1step giant_stitch_sigma0.0_task4_iql5step
#
#   scripts/run_configs.sh concurrent 1 3 0.3 \
#     giant_explore_sigma0.0_task4_iql1step giant_explore_sigma0.0_task4_iql5step \
#     giant_explore_sigma0.0_task4_aciql5step
#
# Config names are configs/exp/<name>.py, without the .py. Each run's checkpoint + value map are
# logged into the same wandb run as training (recovered from the training log, not hand-computed,
# so this stays correct even if main.py's auto-naming logic changes). Stops (does not continue to
# the next batch/config) on the first failure -- fix it and rerun just the remaining configs.

set -o pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs renders

MODE="$1"; shift
if [ "$MODE" != "chain" ] && [ "$MODE" != "concurrent" ]; then
  echo "Usage: $0 chain <gpu_id> <config...>" >&2
  echo "       $0 concurrent <gpu_id> <batch_size> <mem_fraction> <config...>" >&2
  exit 1
fi

GPU="$1"; shift
if [ "$MODE" == "concurrent" ]; then
  BATCH_SIZE="$1"; shift
  MEM_FRACTION="$1"; shift
else
  BATCH_SIZE=1
  MEM_FRACTION=1.0
fi
CONFIGS=("$@")

if [ ${#CONFIGS[@]} -eq 0 ]; then
  echo "No configs given." >&2
  exit 1
fi

PYTHON=/home/hubertchang/miniconda3/envs/cgq/bin/python

run_one () {
  local cfg="$1"
  local log="logs/${cfg}.log"

  echo "--- launching $cfg (gpu=$GPU, mem_fraction=$MEM_FRACTION) ---"
  CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION \
    $PYTHON main.py --exp_config="configs/exp/${cfg}.py" \
    > "$log" 2>&1
  local status=$?
  if [ $status -ne 0 ]; then
    echo "!!! $cfg training FAILED (exit $status) -- see $log" >&2
    return $status
  fi

  # Checkpoint path is exp/cgq/<run_group>/<env_name>/<exp_name>/params_N.pkl -- parse run_group/
  # exp_name straight from it rather than re-deriving them, so this can't drift out of sync with
  # main.py's own path-building logic.
  local ckpt
  ckpt=$(grep "^Saved to " "$log" | tail -1 | sed 's/^Saved to //')
  if [ -z "$ckpt" ]; then
    echo "!!! $cfg: no checkpoint found in $log (offline_steps=0 or save_interval<=0?), skipping render" >&2
    return 0
  fi
  local exp_name run_group
  exp_name=$(basename "$(dirname "$ckpt")")
  run_group=$(basename "$(dirname "$(dirname "$(dirname "$ckpt")")")")
  echo "=== $cfg -> exp_name=$exp_name, run_group=$run_group, checkpoint=$ckpt ==="

  if [ -n "$SKIP_RENDER" ]; then
    echo "--- SKIP_RENDER set, not rendering a value map for $cfg (e.g. antmaze: the value network takes a full high-dim observation, not a 2D xy grid -- render_value_map.py only makes sense for pointmaze-family envs) ---"
    return 0
  fi

  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/render_value_map.py \
    --exp_config="configs/exp/${cfg}.py" \
    --checkpoint="$ckpt" \
    --save_path="renders/${exp_name}.png" \
    --run_group="$run_group" \
    --exp_name="$exp_name" \
    >> "$log" 2>&1
  local render_status=$?
  if [ $render_status -ne 0 ]; then
    echo "!!! $cfg: value-map render FAILED (exit $render_status) -- see $log" >&2
    return $render_status
  fi
}

n=${#CONFIGS[@]}
i=0
while [ $i -lt $n ]; do
  batch=("${CONFIGS[@]:$i:$BATCH_SIZE}")
  echo "=== batch: ${batch[*]} ==="
  pids=()
  for cfg in "${batch[@]}"; do
    run_one "$cfg" &
    pids+=($!)
  done
  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  if [ $fail -ne 0 ]; then
    echo "!!! a run in this batch failed -- stopping" >&2
    exit 1
  fi
  i=$((i + BATCH_SIZE))
done

echo "=== all ${#CONFIGS[@]} configs complete ==="
