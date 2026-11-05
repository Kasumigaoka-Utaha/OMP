# OMP: One-step Meanflow Policy with Directional Alignment

Code for **OMP**, by Han Fang, Yize Huang, Yuheng Zhao, Paul Weng, Xiao Li, and Yutong Ban.

[Paper (ICML 2026 / PMLR)](https://proceedings.mlr.press/v306/fang26h.html) · [arXiv](https://arxiv.org/abs/2512.19347) · [Project page](https://kasumigaoka-utaha.github.io/OMP_web/)

OMP learns a policy conditioned on 3D point clouds and robot state. It adds directional alignment to MeanFlow training and generates an action sequence with one network evaluation at inference. This repository exposes both variants described in the paper:

| Variant | Derivative during training | Hydra configuration | Policy implementation |
| --- | --- | --- | --- |
| **OMP-JVP** | Automatic differentiation with a Jacobian-vector product | `omp_jvp` | [`OMPJVPPolicy`](OMP/omp/policy/omp_jvp_policy.py) |
| **OMP-DDE** | Central finite differences using the Differential Derivation Equation | `omp_dde` | [`OMPDDEPolicy`](OMP/omp/policy/omp_dde_policy.py) |

Both variants use the same one-step inference interface. DDE changes the training derivative calculation.

For comparison, the repository retains three baseline implementations:

| Baseline | Hydra configuration | Policy implementation |
| --- | --- | --- |
| **DP3** | `dp3` | [`DP3`](OMP/omp/policy/dp3.py), inherited simplified implementation |
| **FlowPolicy** | `flowpolicy` | [`FlowPolicy`](OMP/omp/policy/flowpolicy.py) |
| **MP1** | `mp1` | [`MP1`](OMP/omp/policy/mp1.py), the basic MeanFlow policy |

These baseline sources retain the implementations and defaults inherited by this project. Their benchmark results have not been reproduced as part of this cleanup.

## Method and configuration

For noise `z_T`, a demonstration action sequence `z_0`, and predicted interval-averaged velocity `u`, directional alignment supervises `u` against `v_0 = z_T - z_0`:

```text
L_DA = -log((cos(u, v_0) + 1) / 2)
L = L_mse + lambda_Disp * L_Disp + lambda_DA * L_DA
```

See [Section 4 of the paper](https://arxiv.org/html/2512.19347v3#S4) for the MeanFlow identity and derivation. The public configurations use `policy.directional_weight=1.0`, `policy.dispersive_weight=0.5`, `policy.epsilon_dir=1e-6`, and `policy.guidance_scale=1.0` (the straight-line training velocity). OMP-DDE uses `policy.dde_epsilon=0.005`, following the paper's sensitivity study. Loss weights retain the values in the supplied implementation and can be overridden for experiments.

## Repository layout

```text
OMP/
  train.py                     # Shared training workspace
  eval.py                      # Evaluate a saved checkpoint
  setup.py                     # Install the omp Python package
  omp/
    policy/
      omp_jvp_policy.py        # OMP-JVP
      omp_dde_policy.py        # OMP-DDE
      dp3.py                   # DP3 baseline
      flowpolicy.py            # FlowPolicy baseline
      mp1.py                   # MP1 baseline
      base_policy.py           # Shared policy interface
    config/                    # Five method configurations and simulator tasks
    model/                     # Shared encoder and required policy backbones
    dataset/                   # Zarr demonstration loaders
    env_runner/                # Adroit and Meta-World evaluation
scripts/                       # Training, evaluation, and data generation
tests/                         # Policy and source-layout checks
third_party/                   # External simulator/expert dependencies
```

## Installation

Follow [install.md](install.md) for the Linux/CUDA simulation stack. The example environment uses PyTorch 2.0.1; OMP-JVP evaluates derivatives through `torch.autograd.functional.jvp`. The simulators use legacy MuJoCo/Gym dependencies and the modified third-party environments inherited from DP3. External sources, expert weights, demonstration datasets, and trained OMP checkpoints are **not included** in this checkout.

Run the following commands from the repository root after installing the simulator dependencies and PyTorch:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ./OMP
```

## Demonstration data

Place datasets under the repository root's `data/` directory:

```text
data/adroit_hammer_expert.zarr
data/adroit_door_expert.zarr
data/adroit_pen_expert.zarr
data/metaworld_basketball_expert.zarr
```

Task names correspond to YAML filenames in [`OMP/omp/config/task/`](OMP/omp/config/task/). Each dataset is a Zarr replay buffer with `data/state`, `data/action`, `data/point_cloud`, and cumulative `meta/episode_ends`; Adroit data also includes `data/img` for the inherited loader. Arrays share the same leading time dimension. The loader exposes `state` as the policy's `agent_pos` observation. Point and action dimensions must match the task's `shape_meta`.

After obtaining the third-party sources and Adroit expert checkpoints described in [third_party/README.md](third_party/README.md), generate demonstrations with:

```bash
# TASK, NUMBER_OF_EPISODES, GPU
bash scripts/gen_demonstration_adroit.sh hammer 10 0
bash scripts/gen_demonstration_metaworld.sh basketball 10 0
```

The generators write to this repository's `data/` directory. To use another dataset location, pass `task.dataset.zarr_path=/absolute/path/to/dataset.zarr` to training. Public task configurations select at most 10 training demonstrations and use no held-out demonstration split; evaluation uses simulator rollouts.

## Training

```bash
# METHOD, TASK, EXPERIMENT_TAG, RANDOM_SEED, GPU
bash scripts/train_policy.sh omp_jvp adroit_hammer paper 0 0
bash scripts/train_policy.sh omp_dde adroit_hammer paper 0 0
bash scripts/train_policy.sh omp_jvp metaworld_basketball paper 0 0
```

The same launcher supports the retained baselines:

```bash
bash scripts/train_policy.sh dp3 adroit_hammer baseline 0 0
bash scripts/train_policy.sh flowpolicy adroit_hammer baseline 0 0
bash scripts/train_policy.sh mp1 adroit_hammer baseline 0 0
```

Extra arguments are Hydra overrides, for example:

```bash
bash scripts/train_policy.sh omp_dde adroit_pen ablation 10 0 \
    policy.dde_epsilon=0.01 logging.mode=online
```

Logging defaults to offline WandB. Online logging requires your own WandB credentials. Each run saves its configuration, logs, and checkpoints under:

```text
data/outputs/<TASK>-<METHOD>-<EXPERIMENT_TAG>_seed<SEED>/
```

The OMP configurations follow the paper's simulation protocol: 10 demonstrations, seeds `0`, `10`, and `20`, batch size `128`, AdamW with learning rate `1e-4`, observation/prediction/execution horizons `2/4/3`, and evaluation every `200` epochs. Task configurations use `3000` epochs for Adroit and `1000` for Meta-World. For benchmark comparisons, aggregate the five highest evaluation success rates in each training run, then report the mean and standard deviation over the three seeds. These settings are documented in [Section 5 and Appendix D](https://arxiv.org/html/2512.19347v3#S5).

## Evaluation

Use the same method, task, tag, and seed as training. Accepted methods are `omp_jvp`, `omp_dde`, `dp3`, `flowpolicy`, and `mp1`:

```bash
bash scripts/eval_policy.sh omp_jvp adroit_hammer paper 0 0
bash scripts/eval_policy.sh omp_dde adroit_hammer paper 0 0
```

To choose a checkpoint explicitly:

```bash
bash scripts/eval_policy.sh omp_dde adroit_hammer paper 0 0 \
    evaluation.checkpoint=/absolute/path/to/checkpoint.ckpt
```

Evaluation writes results to the run's `evaluation/eval_results.log` and records rollout videos. Evaluating one checkpoint is useful for inspection; the paper's reported metric aggregates evaluations throughout training.

## Scope and validation

This checkout contains the OMP simulation training/evaluation pipeline and the three baseline implementations listed above. The paper also reports real-robot experiments; a complete real-robot deployment stack and its data are not supplied here. The release reorganization and method alignment do not constitute a fresh reproduction of the paper's numerical results. Full training requires the external environments, demonstrations, and a suitable GPU.

Run the repository checks with:

```bash
python -m unittest discover -s tests -v
```

## Citation

```bibtex
@InProceedings{pmlr-v306-fang26h,
  title     = {{OMP}: One-step Meanflow Policy with Directional Alignment},
  author    = {Fang, Han and Huang, Yize and Zhao, Yuheng and Weng, Paul and Li, Xiao and Ban, Yutong},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  pages     = {29147--29166},
  year      = {2026},
  volume    = {306},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v306/fang26h.html}
}
```

## Acknowledgments and license

This codebase builds on [FlowPolicy](https://github.com/zql-kk/FlowPolicy), [3D Diffusion Policy (DP3)](https://github.com/YanjieZe/3D-Diffusion-Policy), and the MeanFlow/MP1 formulation discussed in the paper. We thank the upstream authors and simulator maintainers. The existing [MIT license](LICENSE) and upstream copyright notice are preserved. Third-party components retain their respective licenses.
