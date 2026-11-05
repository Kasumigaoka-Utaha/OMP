"""Evaluate OMP-JVP or OMP-DDE using a saved training checkpoint."""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    os.chdir(ROOT_DIR)

import os
import hydra
from omegaconf import OmegaConf
import pathlib
from train import TrainOMPWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)
    

@hydra.main(
    version_base=None,
    config_name='omp_jvp',
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'omp', 'config'))
)
def main(cfg):
    workspace = TrainOMPWorkspace(cfg)
    workspace.eval()

if __name__ == "__main__":
    main()
