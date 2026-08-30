#!/usr/bin/env python3

"""Evaluate one checkpoint with two balanced Ray workers on one GPU.

This is intentionally separate from eval_ray.py so the historical evaluator
and the active training campaign remain unchanged.
"""

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
import numpy as np
import ray
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import utils
from checkpoint_metadata import (
    build_checkpoint_metadata,
    checkpoint_metadata_mismatches,
)
from logger import Logger
from replay_buffer import make_expert_replay_loader
from video import VideoRecorder

_SUITE_HORIZONS = {
    "libero_spatial": 197,
    "libero_object": 254,
    "libero_goal": 347,
    "libero_10": 517,
}

_EVAL_CHECKPOINT_STEPS = {
    "libero_spatial": 100000,
    "libero_object": 100000,
    "libero_goal": 100000,
    "libero_10": 125000,
}


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


@ray.remote
class SplitWorkspaceIL:
    """One five-task evaluator with a fixed full-suite episode horizon."""

    def __init__(self, cfg, worker_idx, work_dir):
        self.scene_idx = worker_idx
        self.work_dir = Path(work_dir) / f"worker_{worker_idx}"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        robosuite_log_dir = self.work_dir / "robosuite_logs"
        robosuite_log_dir.mkdir(parents=True, exist_ok=True)
        os.environ["ROBO_SUITE_LOG_PATH"] = str(robosuite_log_dir)
        os.chdir(self.work_dir)
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        self.model_seed = int(cfg.get("model_seed", cfg.seed))
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.cfg.suite.task.scenes = self.cfg.suite.task.scenes[
            worker_idx : worker_idx + 1
        ]
        self.cfg.expert_dataset.scenes = self.cfg.suite.task.scenes
        self.cfg.suite.task_make_fn.scenes = self.cfg.suite.task.scenes

        dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            dataset_iterable, self.cfg.batch_size
        )
        if len(self.expert_replay_loader.dataset._paths) != 5:
            raise ValueError(
                "worker task lookup did not resolve exactly five dataset paths: "
                f"worker_idx={worker_idx} "
                f"resolved={len(self.expert_replay_loader.dataset._paths)}"
        )
        self.expert_replay_iter = iter(self.expert_replay_loader)

        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        suite_name = str(self.cfg.suite.task.suite)
        if suite_name not in _SUITE_HORIZONS:
            raise ValueError(f"no fixed full-suite horizon for {suite_name}")
        self.cfg.suite.task_make_fn.max_episode_len = _SUITE_HORIZONS[suite_name]
        self.cfg.suite.task_make_fn.max_state_dim = (
            self.expert_replay_loader.dataset._max_state_dim
        )
        self.env, self.task_descriptions = hydra.utils.call(
            self.cfg.suite.task_make_fn
        )
        if len(self.env) != 5 or len(self.task_descriptions) != 5:
            raise ValueError(
                "worker environment construction did not resolve exactly five tasks: "
                f"worker_idx={worker_idx} envs={len(self.env)} "
                f"descriptions={len(self.task_descriptions)}"
            )

        self.agent = make_agent(
            self.env[0].observation_spec(), self.env[0].action_spec(), cfg
        )

        self.envs_till_idx = len(self.env)
        self.expert_replay_loader.dataset.envs_till_idx = self.envs_till_idx
        self.expert_replay_iter = iter(self.expert_replay_loader)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )

        snapshot = Path(cfg.bc_weight)
        if not snapshot.exists():
            raise FileNotFoundError(f"bc weight not found: {snapshot}")
        print(f"loading bc weight: {snapshot}")
        self.checkpoint_step = self.load_snapshot({"bc": snapshot})

    @property
    def global_step(self):
        return self._global_step

    def eval(self):
        started = time.monotonic()
        self.agent.train(False)
        mean_episode_rewards = []
        mean_successes = []
        task_episode_rewards = []
        task_episode_successes = []
        episode_lengths = []

        for env_idx in range(self.envs_till_idx):
            print(f"evaluating env {env_idx}")
            episode = 0
            rewards = []
            success = []
            lengths = []
            eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)

            while eval_until_episode(episode):
                time_step = self.env[env_idx].reset()
                self.agent.buffer_reset()
                step = 0
                episode_reward = 0.0

                if self.cfg.prompt is not None and self.cfg.prompt != "intermediate_goal":
                    prompt = self.expert_replay_loader.dataset.sample_test(env_idx)
                else:
                    prompt = None

                if episode == 0:
                    self.video_recorder.init(self.env[env_idx], enabled=True)

                while not time_step.last():
                    if self.cfg.prompt == "intermediate_goal":
                        prompt = self.expert_replay_loader.dataset.sample_test(
                            env_idx, step
                        )
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        img_size = self.cfg.dataloader.bc_dataset.img_size
                        if img_size != 128:
                            for key, value in time_step.observation.items():
                                if key.startswith("pixels") and value.ndim == 3:
                                    crop = (128 - img_size) // 2
                                    time_step.observation[key] = value[
                                        :, crop : crop + img_size, crop : crop + img_size
                                    ]
                        action = self.agent.act(
                            time_step.observation,
                            prompt,
                            self.expert_replay_loader.dataset.stats,
                            step,
                            self.global_step,
                            eval_mode=True,
                        )
                    time_step = self.env[env_idx].step(action)
                    self.video_recorder.record(self.env[env_idx])
                    episode_reward += float(time_step.reward)
                    step += 1

                episode += 1
                rewards.append(episode_reward)
                lengths.append(step)
                success.append(bool(time_step.observation["goal_achieved"]))

            self.video_recorder.save(f"{self.scene_idx}_env{env_idx}.mp4")
            mean_reward = float(np.mean(rewards))
            mean_success = float(np.mean(success))
            print(
                f"Env {env_idx}. Episode {episode} finished with total reward: "
                f"{mean_reward}, success rate: {mean_success}, lengths: {lengths}"
            )
            mean_episode_rewards.append(mean_reward)
            mean_successes.append(mean_success)
            task_episode_rewards.append(rewards)
            task_episode_successes.append(success)
            episode_lengths.append(lengths)

        return {
            "mean_episode_rewards": mean_episode_rewards,
            "mean_successes": mean_successes,
            "episode_rewards": task_episode_rewards,
            "episode_successes": task_episode_successes,
            "episode_lengths": episode_lengths,
            "elapsed_seconds": time.monotonic() - started,
            "checkpoint_step": self.checkpoint_step,
        }

    def load_snapshot(self, snapshots):
        with snapshots["bc"].open("rb") as handle:
            payload = torch.load(handle, weights_only=False)
        metadata = payload.get("checkpoint_metadata")
        if not isinstance(metadata, dict):
            raise ValueError("evaluation checkpoint has no checkpoint_metadata")
        expected = build_checkpoint_metadata(
            self.cfg,
            self.agent,
            seed=self.model_seed,
            include_feature_cache_identity=False,
        )
        mismatches = {
            key: {"checkpoint": metadata.get(key), "evaluation": value}
            for key, (_, value) in checkpoint_metadata_mismatches(
                metadata, expected
            ).items()
        }
        if mismatches:
            raise ValueError(f"checkpoint/evaluation mismatch: {mismatches}")

        checkpoint_step = int(payload.get("_global_step", -1))
        filename_step = snapshots["bc"].stem
        if not filename_step.isdigit():
            raise ValueError(
                f"evaluation requires a step-named checkpoint, found {filename_step}"
            )
        if checkpoint_step != int(filename_step):
            raise ValueError(
                "checkpoint payload step does not match filename: "
                f"payload={checkpoint_step} filename={filename_step}"
            )
        if checkpoint_step < 0:
            raise ValueError(f"invalid checkpoint step: {checkpoint_step}")
        suite_name = str(self.cfg.suite.task.suite)
        expected_step = _EVAL_CHECKPOINT_STEPS[suite_name]
        if checkpoint_step != expected_step:
            raise ValueError(
                f"wrong campaign evaluation step for {suite_name}: "
                f"checkpoint={checkpoint_step} expected={expected_step}"
            )
        saved_horizon = int(payload.get("max_episode_len", -1))
        expected_horizon = _SUITE_HORIZONS[suite_name]
        if saved_horizon != expected_horizon:
            raise ValueError(
                f"checkpoint horizon mismatch: checkpoint={saved_horizon} "
                f"expected={expected_horizon}"
            )

        agent_payload = {
            key: value
            for key, value in payload.items()
            if key not in self.__dict__
        }
        self.agent.load_snapshot(agent_payload, eval=True)
        return checkpoint_step


def _task_map(task_cfg):
    mapping = {}
    for block in task_cfg.tasks:
        for worker, tasks in block.items():
            worker = str(worker)
            if worker in mapping:
                raise ValueError(f"duplicate worker key in task config: {worker}")
            mapping[worker] = [str(task) for task in tasks]
    return mapping


def _atomic_write_text(path, content):
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content)
    os.replace(temporary, path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256_manifest(manifest_path, repo_root):
    checked = 0
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        expected_hash, separator, raw_path = line.partition("  ")
        if not separator or len(expected_hash) != 64:
            raise ValueError(f"malformed evaluation source manifest line: {line!r}")
        source_path = Path(raw_path)
        if not source_path.is_absolute():
            source_path = repo_root / source_path
        if not source_path.is_file():
            raise FileNotFoundError(f"fingerprinted source is missing: {source_path}")
        actual_hash = _sha256(source_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"evaluation source fingerprint mismatch for {source_path}: "
                f"expected={expected_hash} actual={actual_hash}"
            )
        checked += 1
    if checked == 0:
        raise ValueError(f"empty evaluation source manifest: {manifest_path}")
    return checked


@hydra.main(version_base=None, config_path="cfgs", config_name="config_eval")
def main(cfg):
    work_dir = Path(HydraConfig.get().runtime.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    existing_outputs = [
        work_dir / name for name in ("COMPLETE", "results.json", "results.txt")
        if (work_dir / name).exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            "refusing to reuse an evaluation output directory containing: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    workers = [str(worker) for worker in cfg.suite.task.scenes]
    tasks = _task_map(cfg.suite.task)

    if len(workers) != 2:
        raise ValueError(f"expected exactly two workers, found {workers}")
    if set(workers) != set(tasks):
        raise ValueError(f"worker/task keys differ: workers={workers}, tasks={list(tasks)}")
    if any(len(tasks[worker]) != 5 for worker in workers):
        sizes = {worker: len(tasks[worker]) for worker in workers}
        raise ValueError(f"expected five tasks per worker, found {sizes}")
    configured_tasks = [task for worker in workers for task in tasks[worker]]
    if len(configured_tasks) != 10 or len(set(configured_tasks)) != 10:
        raise ValueError("the split must contain exactly ten unique task names")
    suite_name = str(cfg.suite.task.suite)
    canonical_path = (
        Path(__file__).resolve().parent / "cfgs" / "suite" / "task" / f"{suite_name}.yaml"
    )
    canonical_config = OmegaConf.load(canonical_path)
    canonical_map = _task_map(canonical_config)
    canonical_tasks = [task for values in canonical_map.values() for task in values]
    if set(configured_tasks) != set(canonical_tasks) or len(canonical_tasks) != 10:
        missing = sorted(set(canonical_tasks) - set(configured_tasks))
        extra = sorted(set(configured_tasks) - set(canonical_tasks))
        raise ValueError(
            f"split differs from canonical {suite_name} task set: "
            f"missing={missing} extra={extra}"
        )

    try:
        expected_episodes = int(cfg.suite.num_eval_episodes)
    except (TypeError, ValueError) as error:
        raise ValueError("suite.num_eval_episodes must be a positive integer") from error
    if expected_episodes <= 0 or expected_episodes != cfg.suite.num_eval_episodes:
        raise ValueError("suite.num_eval_episodes must be a positive integer")
    if int(cfg.batch_size) != 64 or int(cfg.num_demos_per_task) != 50:
        raise ValueError(
            "campaign evaluation requires batch_size=64 and num_demos_per_task=50"
        )

    if str(cfg.device).startswith("cuda") and torch.cuda.device_count() != 1:
        raise RuntimeError(
            "one evaluator must see exactly one GPU; launch each evaluator with "
            "an exclusive Slurm GPU binding or a one-device CUDA_VISIBLE_DEVICES"
        )

    checkpoint_path = Path(cfg.bc_weight).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"bc weight not found: {checkpoint_path}")
    checkpoint_sha256 = _sha256(checkpoint_path)
    expected_checkpoint_step = _EVAL_CHECKPOINT_STEPS[suite_name]
    if checkpoint_path.stem != str(expected_checkpoint_step):
        raise ValueError(
            f"wrong checkpoint filename for {suite_name}: {checkpoint_path.name}; "
            f"expected {expected_checkpoint_step}.pt"
        )
    cfg.bc_weight = str(checkpoint_path)
    source_sha256 = _sha256(Path(__file__).resolve())
    source_manifest_value = os.environ.get("BAKU_EVAL_SOURCE_MANIFEST")
    source_manifest_path = None
    source_manifest_entries = 0
    source_manifest_sha256 = None
    if source_manifest_value:
        source_manifest_path = Path(source_manifest_value).resolve()
        if not source_manifest_path.is_file():
            raise FileNotFoundError(
                f"evaluation source manifest is missing: {source_manifest_path}"
            )
        repo_root = Path(__file__).resolve().parents[1]
        source_manifest_entries = _verify_sha256_manifest(
            source_manifest_path, repo_root
        )
        source_manifest_sha256 = _sha256(source_manifest_path)
    model_seed = int(cfg.get("model_seed", cfg.seed))
    eval_seed = int(cfg.seed)

    ray_cpus = int(os.environ.get("BAKU_EVAL_RAY_CPUS", "8"))
    actor_cpus = float(os.environ.get("BAKU_EVAL_ACTOR_CPUS", "4"))
    actor_gpus = float(os.environ.get("BAKU_EVAL_ACTOR_GPUS", "0.45"))
    # Ray appends a long session/socket suffix and AF_UNIX paths are limited to
    # 107 bytes. Keep this root deliberately short even when SLURM_TMPDIR is long.
    tmp_base = os.environ.get("BAKU_EVAL_RAY_TMP_BASE", "/tmp")
    if not Path(tmp_base).is_dir():
        raise FileNotFoundError(f"Ray temp base does not exist: {tmp_base}")
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    ray_tmp = tempfile.mkdtemp(prefix=f"r{job_id}-", dir=tmp_base)

    print(
        "[eval-split] "
        f"checkpoint={cfg.bc_weight} suite={cfg.suite.task.suite} "
        f"head={cfg.policy_head} model_seed={model_seed} eval_seed={eval_seed} "
        f"view={cfg.camera_pose_variations} episodes_per_task={cfg.suite.num_eval_episodes} "
        f"workers={workers} tasks_per_worker=5 ray_tmp={ray_tmp}",
        flush=True,
    )

    started = time.monotonic()
    try:
        ray.init(
            num_cpus=ray_cpus,
            num_gpus=1,
            include_dashboard=False,
            log_to_driver=True,
            _temp_dir=ray_tmp,
        )
        actors = [
            SplitWorkspaceIL.options(num_cpus=actor_cpus, num_gpus=actor_gpus).remote(
                cfg, worker_idx, work_dir
            )
            for worker_idx in range(len(workers))
        ]
        results = ray.get([actor.eval.remote() for actor in actors])
    finally:
        if ray.is_initialized():
            ray.shutdown()
        try:
            shutil.rmtree(ray_tmp)
        except FileNotFoundError:
            pass
        except OSError as error:
            print(f"[eval-split-warning] failed to remove {ray_tmp}: {error}", flush=True)

    elapsed_seconds = time.monotonic() - started
    task_results = []
    all_episode_rewards = []
    all_episode_successes = []
    worker_elapsed_seconds = []
    total_env_steps = 0
    checkpoint_steps = set()
    episode_horizon = _SUITE_HORIZONS[suite_name]
    for worker_idx, result in enumerate(results):
        worker = workers[worker_idx]
        mean_rewards = result["mean_episode_rewards"]
        mean_successes = result["mean_successes"]
        episode_rewards = result["episode_rewards"]
        episode_successes = result["episode_successes"]
        lengths = result["episode_lengths"]
        checkpoint_steps.add(int(result["checkpoint_step"]))
        worker_elapsed_seconds.append(float(result["elapsed_seconds"]))
        if any(
            len(values) != 5
            for values in (
                mean_rewards,
                mean_successes,
                episode_rewards,
                episode_successes,
                lengths,
            )
        ):
            raise RuntimeError(
                f"worker {worker} did not return five complete task records"
            )
        for task, mean_reward, mean_success, task_rewards, task_successes, task_lengths in zip(
            tasks[worker],
            mean_rewards,
            mean_successes,
            episode_rewards,
            episode_successes,
            lengths,
        ):
            if not (
                len(task_rewards)
                == len(task_successes)
                == len(task_lengths)
                == expected_episodes
            ):
                raise RuntimeError(
                    f"task {task} did not return {expected_episodes} complete episodes"
                )
            if not all(1 <= int(value) <= episode_horizon for value in task_lengths):
                raise RuntimeError(f"task {task} returned an invalid episode length")
            if not all(np.isfinite(float(value)) for value in task_rewards):
                raise RuntimeError(f"task {task} returned a non-finite reward")
            computed_mean_reward = float(np.mean(task_rewards))
            computed_mean_success = float(np.mean(task_successes))
            if not np.isclose(float(mean_reward), computed_mean_reward):
                raise RuntimeError(f"task {task} returned an inconsistent mean reward")
            if not np.isclose(float(mean_success), computed_mean_success):
                raise RuntimeError(f"task {task} returned an inconsistent success rate")
            total_env_steps += sum(int(value) for value in task_lengths)
            task_results.append(
                {
                    "worker": worker,
                    "task": task,
                    "mean_episode_reward": computed_mean_reward,
                    "success_rate": computed_mean_success,
                    "episode_rewards": [float(value) for value in task_rewards],
                    "episode_successes": [bool(value) for value in task_successes],
                    "episode_lengths": [int(value) for value in task_lengths],
                }
            )
            all_episode_rewards.extend(float(value) for value in task_rewards)
            all_episode_successes.extend(bool(value) for value in task_successes)

    if len(checkpoint_steps) != 1:
        raise RuntimeError(f"workers loaded different checkpoint steps: {checkpoint_steps}")
    checkpoint_step = checkpoint_steps.pop()
    if checkpoint_step != expected_checkpoint_step:
        raise RuntimeError(
            f"workers loaded step {checkpoint_step}, expected {expected_checkpoint_step}"
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": checkpoint_sha256,
        "suite": str(cfg.suite.task.suite),
        "policy_head": str(cfg.policy_head),
        "model_seed": model_seed,
        "eval_seed": eval_seed,
        "camera_pose_variation": (
            None if cfg.camera_pose_variations is None else str(cfg.camera_pose_variations)
        ),
        "episode_horizon": episode_horizon,
        "episodes_per_task": expected_episodes,
        "num_tasks": len(task_results),
        "num_episodes": len(all_episode_rewards),
        "elapsed_seconds": elapsed_seconds,
        "worker_elapsed_seconds": worker_elapsed_seconds,
        "total_env_steps": total_env_steps,
        "env_steps_per_second": total_env_steps / elapsed_seconds,
        "episodes_per_hour": (
            len(all_episode_rewards) * 3600 / elapsed_seconds
        ),
        "mean_episode_reward": float(np.mean(all_episode_rewards)),
        "mean_success_rate": float(np.mean(all_episode_successes)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "evaluator_source_sha256": source_sha256,
        "source_manifest": (
            None if source_manifest_path is None else str(source_manifest_path)
        ),
        "source_manifest_sha256": source_manifest_sha256,
        "source_manifest_entries": source_manifest_entries,
        "tasks": task_results,
    }

    result_text = json.dumps(summary, indent=2) + "\n"
    _atomic_write_text(work_dir / "results.json", result_text)
    _atomic_write_text(
        work_dir / "results.txt",
        f"Episode rewards: {all_episode_rewards}\n"
        f"Successes: {all_episode_successes}\n"
        f"Mean Episode reward: {summary['mean_episode_reward']}\n"
        f"Mean Success rate: {summary['mean_success_rate']}\n"
        f"Elapsed seconds: {elapsed_seconds}\n"
        f"Episodes per hour: {summary['episodes_per_hour']}\n",
    )
    _atomic_write_text(
        work_dir / "COMPLETE",
        f"results_sha256={hashlib.sha256(result_text.encode()).hexdigest()}\n"
        f"checkpoint_sha256={checkpoint_sha256}\n"
        f"checkpoint_step={checkpoint_step}\n",
    )
    print(
        f"[eval-split-complete] elapsed_seconds={elapsed_seconds:.1f} "
        f"episodes_per_hour={summary['episodes_per_hour']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
