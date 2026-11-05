"""Shared training workspace for OMP-JVP and OMP-DDE."""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import time
import threading
from hydra.core.hydra_config import HydraConfig
from omp.policy.base_policy import BasePolicy
from omp.dataset.base_dataset import BaseDataset
from omp.env_runner.base_runner import BaseRunner
from omp.common.checkpoint_util import TopKCheckpointManager
from omp.common.pytorch_util import dict_apply, optimizer_to
from omp.model.mean.ema_model import EMAModel
from omp.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _copy_to_cpu(value):
    """Copy checkpoint state before handing it to a background writer."""
    if isinstance(value, torch.Tensor):
        return value.detach().to('cpu').clone()
    if isinstance(value, dict):
        return {key: _copy_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_to_cpu(item) for item in value)
    return copy.deepcopy(value)


class TrainOMPWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: BasePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: BasePolicy = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except:  # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.debug:
            cfg.training.num_epochs = min(cfg.training.num_epochs, 100)
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False

        RUN_VALIDATION = False  # reduce time cost

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        num_train_batches = len(train_dataloader)
        if cfg.training.max_train_steps is not None:
            num_train_batches = min(num_train_batches, cfg.training.max_train_steps)
        if num_train_batches < 1:
            raise ValueError("The training dataset produced no batches.")
        accumulate_every = cfg.training.gradient_accumulate_every
        if accumulate_every < 1:
            raise ValueError("training.gradient_accumulate_every must be at least 1.")
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=((num_train_batches + accumulate_every - 1)
                                // accumulate_every) * cfg.training.num_epochs,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)

        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)

        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        os.makedirs(self.output_dir, exist_ok=True)
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **OmegaConf.to_container(cfg.logging, resolve=True)
        )
        wandb_run.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        # training loop
        test_scores = []
        self.optimizer.zero_grad()
        for local_epoch_idx in range(self.epoch, cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                           leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # compute loss
                    t1_1 = time.time()
                    raw_loss, loss_dict = self.model.compute_loss(batch)
                    group_start = (batch_idx // accumulate_every) * accumulate_every
                    group_size = min(accumulate_every, num_train_batches - group_start)
                    loss = raw_loss / group_size
                    loss.backward()

                    t1_2 = time.time()

                    # Flush the final batch as well as complete accumulation groups.
                    is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                    if cfg.training.max_train_steps is not None:
                        is_last_batch = is_last_batch or batch_idx >= cfg.training.max_train_steps - 1
                    optimizer_step = ((batch_idx + 1) % cfg.training.gradient_accumulate_every == 0
                                      or is_last_batch)
                    if optimizer_step:
                        if cfg.training.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.training.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema and optimizer_step:
                        ema.step(self.model)
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()

                    if verbose:
                        print(f"total one step time: {t2 - t1:.3f}")
                        print(f" compute loss time: {t1_2 - t1_1:.3f}")
                        print(f" step optimizer time: {t1_3 - t1_2:.3f}")
                        print(f" update ema time: {t1_4 - t1_3:.3f}")
                        print(f" logging time: {t1_5 - t1_4:.3f}")

                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps - 1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run rollout
            if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
                t3 = time.time()
                # runner_log = env_runner.run(policy, dataset=dataset)
                runner_log = env_runner.run(policy)
                t4 = time.time()
                # print(f"rollout time: {t4-t3:.3f}")
                # log all
                step_log.update(runner_log)
                test_scores.append(runner_log['test_mean_score'])

            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                   leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            loss, loss_dict = self.model.compute_loss(batch)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps - 1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

            # Run one-step action generation on a training batch.
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    gt_action = batch['action']

                    result = policy.predict_action(obs_dict)
                    pred_action = result['action_pred']
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log['train_action_mse_error'] = mse.item()
                    del batch
                    del obs_dict
                    del gt_action
                    del result
                    del pred_action
                    del mse

            if env_runner is None:
                step_log['test_mean_score'] = - train_loss
                test_scores.append(step_log['test_mean_score'])

            # checkpoint
            if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint(next_epoch=self.epoch + 1)
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value

                # We can't copy the last checkpoint here
                # since save_checkpoint uses threads.
                # therefore at this point the file might have been empty!
                topk_ckpt_path = None
                if topk_manager.monitor_key in metric_dict:
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                if topk_ckpt_path is not None:
                    self.save_checkpoint(path=topk_ckpt_path, next_epoch=self.epoch + 1)
            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

        if cfg.checkpoint.save_ckpt and cfg.checkpoint.save_last_ckpt:
            self.save_checkpoint()

        last_data = dict()
        if len(test_scores) > 0:
            top5 = sorted(test_scores, reverse=True)[:5]
            top5_mean = 100 * sum(top5) / len(top5)
            print("top 5 mean scores:", top5_mean)
            last_data["top_5_mean_scores"] = top5_mean
            wandb_run.log(last_data, step=self.global_step)
        else:
            print("No rollout or evaluation scores were collected; skipping top-5 summary.")
        wandb_run.finish()

    def eval(self):
        """Evaluate trained weights; never fall back to an untrained model."""
        cfg = copy.deepcopy(self.cfg)
        if cfg.task.env_runner is None:
            raise ValueError("Evaluation requires a task.env_runner.")

        checkpoint = OmegaConf.select(cfg, "evaluation.checkpoint")
        checkpoint_path = (pathlib.Path(checkpoint).expanduser() if checkpoint
                           else self.get_checkpoint_path(tag="latest"))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. Train this run first or "
                "set evaluation.checkpoint=/absolute/path/to/checkpoint.ckpt."
            )
        cprint(f"Loading checkpoint {checkpoint_path}", 'magenta')
        # Keep the current output directory when evaluating a checkpoint elsewhere.
        self.load_checkpoint(path=checkpoint_path, include_keys=self.include_keys)
        eval_output_dir = pathlib.Path(
            OmegaConf.select(cfg, "evaluation.output_dir")
            or pathlib.Path(self.output_dir).joinpath('evaluation')
        ).expanduser()
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=str(eval_output_dir))
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.to(torch.device(cfg.training.device))

        runner_log = env_runner.run_eval(policy)

        success_log_path = eval_output_dir.joinpath('eval_results.log')
        with success_log_path.open('a', encoding='utf-8') as f:
            f.write(f"For Environment {cfg.task.name}, "
                    f"mean_success_rates: {runner_log['mean_success_rates']:.4f}\n")
        cprint(f"Success rates logged to {success_log_path}", 'green')

        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')
        return runner_log

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def save_checkpoint(self, path=None, tag='latest',
                        exclude_keys=None,
                        include_keys=None,
                        use_thread=False,
                        next_epoch=None):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        }

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                # Periodic checkpoints are written before the epoch counter advances.
                # Final/manual checkpoints already hold the next epoch to train.
                saved_value = next_epoch if key == 'epoch' and next_epoch is not None else value
                payload['pickles'][key] = dill.dumps(saved_value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)

        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())

    def get_checkpoint_path(self, tag='latest'):
        if tag == 'latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag == 'best':
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])

    def load_checkpoint(self, path=None, tag='latest',
                        exclude_keys=None,
                        include_keys=None,
                        **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload,
                          exclude_keys=exclude_keys,
                          include_keys=include_keys)
        return payload

    @classmethod
    def create_from_checkpoint(cls, path,
                               exclude_keys=None,
                               include_keys=None,
                               **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


@hydra.main(
    version_base=None,
    config_name='omp_jvp',
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'omp', 'config'))
)
def main(cfg):
    workspace = TrainOMPWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
