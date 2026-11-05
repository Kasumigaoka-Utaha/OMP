# Installing OMP

The simulation pipeline uses Linux, an NVIDIA CUDA GPU, MuJoCo 2.1.0, and the modified legacy environments from [DP3](https://github.com/YanjieZe/3D-Diffusion-Policy). The paper uses an RTX 4090. The commands below provide a compatibility-oriented Python 3.8 / PyTorch 2.0.1 setup based on the inherited environment; they are not a newly validated environment lock. Native Windows simulation is not covered by these instructions.

Run commands from the OMP repository root unless a different directory is shown. Obtain this OMP repository from its published location; the upstream FlowPolicy repository is an acknowledgment, not the OMP source repository.

## 1. Create the environment and install PyTorch

```bash
conda create -n omp python=3.8 -y
conda activate omp

# Example CUDA 11.7 wheels; choose a matching CUDA/PyTorch combination for your GPU.
python -m pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu117
```

Both OMP variants and the retained DP3, FlowPolicy, and MP1 baselines use this environment. OMP-JVP uses `torch.autograd.functional.jvp` with a differentiable output graph. The example versions above are also specified in `requirements.txt`; update them together when changing the environment. See the [official PyTorch version matrix](https://pytorch.org/get-started/previous-versions/) for alternative wheel combinations.

## 2. Obtain third-party sources

This checkout contains instructions in `third_party/`, but does not bundle the environments or RL expert weights. Follow [third_party/README.md](third_party/README.md) to obtain the modified DP3 sources. The required directory layout is:

```text
third_party/
  gym-0.21.0/
  Metaworld/
  mujoco-py-2.1.2.14/
  pytorch3d_simplified/
  rrl-dependencies/mj_envs/
  rrl-dependencies/mjrl/
  VRL3/src/
  VRL3/vrl3_ckpts/            # Adroit experts; VRL3/ckpts/ is also supported
```

Use the modified sources rather than replacing them with current upstream Gym/Meta-World releases; their APIs and point-cloud data generators differ.

## 3. Install MuJoCo 2.1.0

```bash
mkdir -p ~/.mujoco
curl -L https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
    -o /tmp/omp-mujoco210.tar.gz
tar -xzf /tmp/omp-mujoco210.tar.gz -C ~/.mujoco
```

Add the following environment variables to your shell configuration, adapting the CUDA and driver library locations to your machine:

```bash
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda/lib64"
export MUJOCO_GL=egl
```

Reload the shell configuration before continuing. Compiling `mujoco-py` and the point-cloud extensions requires a C/C++ toolchain and the corresponding system OpenGL/EGL development libraries. Refer to the [DP3 installation guide](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/master/INSTALL.md) for its original simulator setup.

## 4. Install Python and simulator dependencies

```bash
# Legacy build tools for the vendored Gym and mujoco-py sources.
python -m pip install setuptools==59.5.0 wheel==0.38.4 Cython==0.29.35 patchelf==0.17.2.0

python -m pip install -r requirements.txt
python -m pip install -e third_party/gym-0.21.0
python -m pip install -e third_party/mujoco-py-2.1.2.14
python -m pip install -e third_party/Metaworld
python -m pip install -e third_party/rrl-dependencies/mj_envs
python -m pip install -e third_party/rrl-dependencies/mjrl
python -m pip install -e third_party/pytorch3d_simplified
python -m pip install -e ./OMP
```

The simplified PyTorch3D extension must be built against the selected PyTorch/CUDA installation. This project uses the Gym 0.21 and Zarr 2 APIs; upgrading to Gymnasium or Zarr 3 requires code changes.

## 5. Check the installation

```bash
python -c "import torch, omp; from torch.autograd.functional import jvp; print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
python -c "from omp.policy.omp_jvp_policy import OMPJVPPolicy; from omp.policy.omp_dde_policy import OMPDDEPolicy; print('OMP policy imports OK')"
python -c "from omp.policy.dp3 import DP3; from omp.policy.flowpolicy import FlowPolicy; from omp.policy.mp1 import MP1; print('Baseline policy imports OK')"
python -c "from omp.env import AdroitEnv, MetaWorldEnv; print('Simulator imports OK')"
python -m unittest discover -s tests -v
```

Check the five resolved Hydra configurations without starting training:

```bash
for method in omp_jvp omp_dde dp3 flowpolicy mp1; do
    python OMP/train.py --config-name="$method" --cfg job --resolve
done
```

These checks do not replace a complete simulator rollout or benchmark reproduction. Prepare demonstrations and use the training/evaluation commands in [README.md](README.md).

Logging defaults to offline WandB. Run `wandb login` and pass `logging.mode=online` when you want to upload logs to your own account.
