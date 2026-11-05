#!/usr/bin/env bash
# Usage: bash scripts/gen_demonstration_metaworld.sh basketball [EPISODES=10] [GPU=0]
set -euo pipefail

task_name=${1:?Specify a Meta-World task, for example basketball}
num_episodes=${2:-10}
gpu_id=${3:-0}
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
generator_dir="$repo_root/third_party/Metaworld"
if [[ ! -f "$generator_dir/gen_demonstration_expert.py" ]]; then
    echo "Missing $generator_dir/gen_demonstration_expert.py. See install.md for third-party setup." >&2
    exit 1
fi
mkdir -p "$repo_root/data"
cd -- "$generator_dir"
export CUDA_VISIBLE_DEVICES="$gpu_id"
exec "${PYTHON:-python}" gen_demonstration_expert.py \
    "--env_name=$task_name" \
    --num_episodes "$num_episodes" \
    --root_dir "$repo_root/data/"
