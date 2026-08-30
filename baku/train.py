#!/usr/bin/env python3

import warnings
import os
import random

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import hydra
import torch
import cv2
import numpy as np

import utils
from checkpoint_metadata import (
    build_checkpoint_metadata,
    checkpoint_metadata_mismatches,
)
from logger import Logger
from replay_buffer import make_expert_replay_loader
from video import VideoRecorder

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec[key].shape
    if cfg.use_proprio:
        obs_shape[cfg.suite.proprio_key] = obs_spec[cfg.suite.proprio_key].shape
    obs_shape[cfg.suite.feature_key] = obs_spec[cfg.suite.feature_key].shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)


class WorkspaceIL:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # load data
        dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            dataset_iterable, self.cfg.batch_size
        )
        self.expert_replay_iter = iter(self.expert_replay_loader)
        self.stats = self.expert_replay_loader.dataset.stats

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.cfg.suite.task_make_fn.max_episode_len = (
            self.expert_replay_loader.dataset._max_episode_len
        )
        self.cfg.suite.task_make_fn.max_state_dim = (
            self.expert_replay_loader.dataset._max_state_dim
        )
        if self.cfg.suite.name == "dmc":
            self.cfg.suite.task_make_fn.max_action_dim = (
                self.expert_replay_loader.dataset._max_action_dim
            )

        self.env, self.task_descriptions = hydra.utils.call(self.cfg.suite.task_make_fn)

        # create agent
        self.agent = make_agent(
            self.env[0].observation_spec(), self.env[0].action_spec(), cfg
        )

        self.envs_till_idx = self.expert_replay_loader.dataset.envs_till_idx

        # Discretizer for BeT
        if repr(self.agent) != "mtact":
            needs_discretization = repr(self.agent) == "rt1" or self.cfg.agent.policy_head in [
                "bet",
                "vqbet",
            ]
            if needs_discretization and not self.cfg.load_bc:
                self.agent.discretize(
                    self.expert_replay_loader.dataset.actions,
                    self.expert_replay_loader.dataset.preprocess,
                )
            elif needs_discretization:
                print("Skipping action discretizer pretraining while resuming a checkpoint.")

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    def checkpoint_metadata(self):
        """Identity fields required for a safe training continuation."""
        return build_checkpoint_metadata(self.cfg, self.agent)

    def validate_checkpoint_metadata(self, payload):
        saved = payload.get("checkpoint_metadata")
        if saved is None:
            print(
                "[warn] legacy checkpoint has no configuration metadata; "
                "compatibility cannot be fully verified"
            )
            return
        expected = self.checkpoint_metadata()
        mismatches = checkpoint_metadata_mismatches(saved, expected)
        if mismatches:
            details = ", ".join(
                f"{key}: checkpoint={actual!r}, current={current!r}"
                for key, (actual, current) in mismatches.items()
            )
            raise ValueError(f"checkpoint/config mismatch: {details}")

    def eval(self):
        self.agent.train(False)
        episode_rewards = []
        successes = []

        num_envs = self.envs_till_idx

        for env_idx in range(num_envs):
            print(f"evaluating env {env_idx}")
            episode, total_reward = 0, 0
            eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)
            success = []

            while eval_until_episode(episode):
                time_step = self.env[env_idx].reset()
                self.agent.buffer_reset()
                step = 0

                # prompt
                if self.cfg.prompt != None and self.cfg.prompt != "intermediate_goal":
                    prompt = self.expert_replay_loader.dataset.sample_test(env_idx)
                else:
                    prompt = None

                if episode == 0:
                    self.video_recorder.init(self.env[env_idx], enabled=True)

                # plot obs with cv2
                while not time_step.last():
                    if self.cfg.prompt == "intermediate_goal":
                        prompt = self.expert_replay_loader.dataset.sample_test(
                            env_idx, step
                        )
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(
                            time_step.observation,
                            prompt,
                            self.stats,
                            step,
                            self.global_step,
                            eval_mode=True,
                        )
                    time_step = self.env[env_idx].step(action)
                    self.video_recorder.record(self.env[env_idx])
                    total_reward += time_step.reward
                    step += 1

                episode += 1
                success.append(time_step.observation["goal_achieved"])
            self.video_recorder.save(f"{self.global_step}_env{env_idx}.mp4")
            episode_rewards.append(total_reward / episode)
            successes.append(np.mean(success))

        for _ in range(len(self.env) - num_envs):
            episode_rewards.append(0)
            successes.append(0)

        with self.logger.log_and_dump_ctx(self.global_step, ty="eval") as log:
            for env_idx, reward in enumerate(episode_rewards):
                log(f"episode_reward_env{env_idx}", reward)
                log(f"success_env{env_idx}", successes[env_idx])
            log("episode_reward", np.mean(episode_rewards[:num_envs]))
            log("success", np.mean(successes))
            log("episode_length", step * self.cfg.suite.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

        self.agent.train(True)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.suite.num_train_steps, 1)
        log_every_step = utils.Every(self.cfg.suite.log_every_steps, 1)
        eval_every_step = utils.Every(self.cfg.suite.eval_every_steps, 1)
        save_every_step = utils.Every(self.cfg.suite.save_every_steps, 1)

        metrics = None
        while train_until_step(self.global_step):
            # try to evaluate
            if (
                self.cfg.eval
                and eval_every_step(self.global_step)
                and self.global_step > 0
            ):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval()

            # update
            metrics = self.agent.update(self.expert_replay_iter, self.global_step)
            self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # log
            if log_every_step(self.global_step):
                elapsed_time, total_time = self.timer.reset()
                with self.logger.log_and_dump_ctx(self.global_frame, ty="train") as log:
                    log("total_time", total_time)
                    log("actor_loss", metrics["actor_loss"])
                    log("step", self.global_step)

            # The counter stored in a checkpoint is the next update to run.
            # Increment before saving so a resumed process does not apply the
            # just-completed optimizer update a second time.
            self._global_step += 1

            # save snapshot
            if save_every_step(self.global_step):
                self.save_snapshot()

        # The regular interval ends at 150000 for a 151000-step run. Save the
        # actual terminal state as well so evaluation has an unambiguous final
        # checkpoint and a resumed job never has to repeat the last 1000 steps.
        self.save_snapshot()

    def save_snapshot(self):
        snapshot_dir = self.work_dir / "snapshot"
        snapshot_dir.mkdir(exist_ok=True)
        snapshot = snapshot_dir / f"{self.global_step}.pt"
        self.agent.clear_buffers()
        keys_to_save = ["_global_step", "_global_episode", "stats"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload["checkpoint_metadata"] = self.checkpoint_metadata()
        payload["rng_state"] = capture_rng_state()
        payload.update(self.agent.save_snapshot())
        temporary_snapshot = snapshot.with_name(
            f".{snapshot.name}.{os.getpid()}.tmp"
        )
        try:
            with temporary_snapshot.open("wb") as f:
                torch.save(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_snapshot, snapshot)
        finally:
            if temporary_snapshot.exists():
                temporary_snapshot.unlink()

        self.agent.buffer_reset()

    def load_snapshot(self, snapshots):
        with snapshots["bc"].open("rb") as f:
            payload = torch.load(f, weights_only=False)

        self.validate_checkpoint_metadata(payload)
        for k in ["_global_step", "_global_episode", "stats"]:
            if k in payload:
                self.__dict__[k] = payload[k]
        # Timing is attempt-local. Restoring Timer's wall-clock timestamps would
        # count time spent waiting in Slurm's queue as training time.
        self.timer = utils.Timer()
        print(
            f"resumed workspace from step {self._global_step}, episode {self._global_episode}"
        )

        rng_state = payload.get("rng_state")
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__ and k not in {
                "rng_state",
                "checkpoint_metadata",
            }:
                agent_payload[k] = v
        if "vqvae" in snapshots:
            with snapshots["vqvae"].open("rb") as f:
                payload = torch.load(f, weights_only=False)
            agent_payload["vqvae"] = payload
        self.agent.load_snapshot(agent_payload, eval=False, load_opt=True)
        restore_rng_state(rng_state)
        if rng_state is not None:
            print("restored Python, NumPy, Torch, and CUDA RNG state")


@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    from train import WorkspaceIL as W

    root_dir = Path.cwd()
    # Load checkpoint as true resume (model + optimizer + counters)
    snapshots = None
    if cfg.load_bc:
        bc_snapshot = Path(cfg.bc_weight)
        if not bc_snapshot.is_file():
            raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
        print(f"resuming from snapshot: {bc_snapshot}")
        snapshots = {"bc": bc_snapshot}

    workspace = W(cfg)

    if snapshots is not None:
        workspace.load_snapshot(snapshots)

    workspace.train()


if __name__ == "__main__":
    main()
