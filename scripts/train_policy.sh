#!/usr/bin/env bash
# Usage: bash scripts/train_policy.sh omp_jvp|omp_dde|dp3|flowpolicy|mp1 TASK TAG SEED GPU [Hydra overrides...]
set -euo pipefail

if [[ $# -lt 5 ]]; then
    echo "Usage: $0 omp_jvp|omp_dde|dp3|flowpolicy|mp1 TASK TAG SEED GPU [Hydra overrides...]" >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
case "$1" in
    omp_jvp|omp-jvp|OMP-JVP) method=omp_jvp ;;
    omp_dde|omp-dde|OMP-DDE) method=omp_dde ;;
    dp3|DP3) method=dp3 ;;
    flowpolicy|FlowPolicy) method=flowpolicy ;;
    mp1|MP1) method=mp1 ;;
    *) echo "Unknown method: $1. Choose omp_jvp, omp_dde, dp3, flowpolicy, or mp1." >&2; exit 2 ;;
esac
task_name=$2
tag=$3
seed=$4
gpu_id=$5
shift 5
exp_name="${task_name}-${method}-${tag}"
run_dir="$repo_root/data/outputs/${exp_name}_seed${seed}"
cd -- "$repo_root"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="$gpu_id"
exec "${PYTHON:-python}" "$repo_root/OMP/train.py" --config-name "$method" \
    "task=$task_name" \
    "hydra.run.dir=$run_dir" \
    "training.seed=$seed" \
    "training.device=cuda:0" \
    "exp_name=$exp_name" \
    "logging.mode=${WANDB_MODE:-offline}" \
    "checkpoint.save_ckpt=true" \
    "$@"
