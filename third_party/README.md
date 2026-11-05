# Third-party simulation dependencies

The OMP source tree does not bundle simulator sources, expert policies, or demonstration datasets. Its Adroit and Meta-World wrappers use the modified environment sources maintained with [3D Diffusion Policy (DP3)](https://github.com/YanjieZe/3D-Diffusion-Policy/tree/master/third_party). Use that collection rather than a current, unmodified Meta-World release.

From the OMP repository root, an example way to obtain the sources is:

```bash
git clone --depth 1 https://github.com/YanjieZe/3D-Diffusion-Policy.git /tmp/omp-dp3-sources
cp -R /tmp/omp-dp3-sources/third_party/. ./third_party/
```

Keep the upstream license and copyright files when copying these sources. Record the upstream commit you use if you need a reproducible environment. The current OMP checkout does not specify a validated dependency-source revision.

Required directories for OMP's simulation pipeline:

| Directory | Purpose |
| --- | --- |
| `gym-0.21.0/` | Legacy Gym API |
| `mujoco-py-2.1.2.14/` | MuJoCo 2.1.0 bindings |
| `Metaworld/` | Modified Meta-World environments and expert demonstration generator |
| `rrl-dependencies/mj_envs/` | Adroit tasks |
| `rrl-dependencies/mjrl/` | Adroit environment utilities |
| `pytorch3d_simplified/` | Point-cloud sampling extension |
| `VRL3/src/` | Adroit expert demonstration generator |
| `VRL3/vrl3_ckpts/` or `VRL3/ckpts/` | Adroit expert weights |

For Adroit demonstration generation, the launcher accepts existing files such as `VRL3/vrl3_ckpts/vrl3_hammer.pt` and `VRL3/ckpts/vrl3_hammer.pt`, or an explicit checkpoint path as its fourth argument. If these weights are missing, obtain the expert checkpoint archive linked in the [DP3 data instructions](https://github.com/YanjieZe/3D-Diffusion-Policy#-data) and place it under `third_party/VRL3/`. These expert policies collect data; they are not OMP checkpoints.

Install the dependencies using [install.md](../install.md), then run `scripts/gen_demonstration_adroit.sh` or `scripts/gen_demonstration_metaworld.sh` from the repository root. Generated Zarr datasets are written to the root `data/` directory. DexArt assets in the upstream collection are not required for the Adroit/Meta-World OMP pipeline documented here.
