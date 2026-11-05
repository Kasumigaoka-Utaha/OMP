#!/usr/bin/env bash
# Usage: bash scripts/gen_demonstration_adroit.sh hammer [EPISODES=10] [GPU=0] [EXPERT_CHECKPOINT]
set -euo pipefail

task_name=${1:?Specify an Adroit task: door, hammer, or pen}
num_episodes=${2:-10}
gpu_id=${3:-0}
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
generator_dir="$repo_root/third_party/VRL3/src"
expert_checkpoint=${4:-"$repo_root/third_party/VRL3/ckpts/vrl3_${task_name}.pt"}
if [[ $# -lt 4 && ! -f "$expert_checkpoint" ]]; then
    expert_checkpoint="$repo_root/third_party/VRL3/vrl3_ckpts/vrl3_${task_name}.pt"
fi
if [[ ! -f "$generator_dir/gen_demonstration_expert.py" ]]; then
    echo "Missing $generator_dir/gen_demonstration_expert.py. See install.md for third-party setup." >&2
    exit 1
fi
if [[ ! -f "$expert_checkpoint" ]]; then
    echo "Missing expert checkpoint: $expert_checkpoint. Pass its absolute path as the fourth argument." >&2
    exit 1
fi
mkdir -p "$repo_root/data"
cd -- "$generator_dir"
export CUDA_VISIBLE_DEVICES="$gpu_id"
exec "${PYTHON:-python}" gen_demonstration_expert.py \
    --env_name "$task_name" \
    --num_episodes "$num_episodes" \
    --root_dir "$repo_root/data/" \
    --expert_ckpt_path "$expert_checkpoint" \
    --img_size 84 \
    --not_use_multi_view \
    --use_point_crop
