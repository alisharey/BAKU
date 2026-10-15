"""Specs-only environment for Franka training.

train.py builds an environment per task to read observation/action specs; the
real robot loop lives in franka/deploy.py (camera topics in, actions out to
gamepad_teleop --input policy), so reset/step are not available here.
"""

import json
from pathlib import Path

import dm_env
import numpy as np
from dm_env import specs


class FrankaSpecEnv(dm_env.Environment):
    def __init__(self, pixel_keys, height, width, max_state_dim, task_description):
        self._task_description = task_description
        self._obs_spec = {
            key: specs.BoundedArray(shape=(3, height, width), dtype=np.uint8, minimum=0, maximum=255, name=key)
            for key in pixel_keys
        }
        self._obs_spec["proprioceptive"] = specs.BoundedArray(
            shape=(8,), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name="proprioceptive")
        self._obs_spec["features"] = specs.BoundedArray(
            shape=(max_state_dim,), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name="features")
        # [dx dy dz droll dpitch dyaw gripper]: TeleMoMa action, gripper 1 = open
        self._action_spec = specs.BoundedArray(
            shape=(7,), dtype=np.float32, minimum=-1.0, maximum=1.0, name="action")

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def reset(self):
        raise NotImplementedError("run the robot with franka/deploy.py")

    def step(self, action):
        raise NotImplementedError("run the robot with franka/deploy.py")


def make(dataset_path, tasks, pixel_keys, height, width, max_episode_len, max_state_dim, seed=0, eval=False,
         **unused):
    path = Path(dataset_path)
    if not tasks:
        tasks = sorted(p.stem for p in path.glob("*.pkl"))
    manifest = path / "dataset.json"
    texts = json.loads(manifest.read_text())["tasks"] if manifest.exists() else {}
    descriptions = [texts.get(task, {}).get("task", task.replace("_", " ")) for task in tasks]
    envs = [FrankaSpecEnv(pixel_keys, height, width, max_state_dim, text) for text in descriptions]
    return envs, descriptions
