#!/usr/bin/env bash
# Train the included Adroit and Meta-World tasks sequentially.
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 omp_jvp|omp_dde|dp3|flowpolicy|mp1 TAG SEED GPU [Hydra overrides...]" >&2
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
tag=$2
seed=$3
gpu_id=$4
shift 4
for config in "$repo_root"/OMP/omp/config/task/{adroit_,metaworld_}*.yaml; do
    [[ -f "$config" ]] || continue
    task_name="$(basename -- "$config" .yaml)"
    bash "$script_dir/train_policy.sh" "$method" "$task_name" "$tag" "$seed" "$gpu_id" "$@"
done
