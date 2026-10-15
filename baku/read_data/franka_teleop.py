"""BAKU dataset for Franka demos built by franka/build_dataset.py.

Differences from the LIBERO and xArm loaders, on purpose:

* Actions are the recorded TeleMoMa actions (target - current pose, base frame,
  as the teleop executed them), taken as they are. The xArm loader instead
  rebuilds relative actions from absolute poses (get_relative_action); LIBERO
  uses robosuite's actions, which are already scaled to [-1, 1] (OSC: 1 = 5 cm,
  0.5 rad), so its loader does not normalize (stats min 0, max 1).
* Normalization is BAKU's min-max step, (x - min) / (max - min), undone in
  agent.act with the same stats (saved in every snapshot). LIBERO's loader uses
  min 0 / max 1 (no change); by default this one matches the xArm loader:
  min/max of the demo actions per dimension, rotation dims left at 0 / 1.
* Padding past the end of a demo: zero pose action (the arm holds still) and
  the last gripper state.
"""

import json
import pickle as pkl
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset

POSE_DIMS = 6


def action_stats(actions, normalization, percentile):
    """min/max for BAKU's (x - min) / (max - min) (undone in agent.act with the same stats).

    xarm (default): exactly BAKU's xArm loader (read_data/xarm_env.py): per-dimension
        min/max over all demo actions (history padding included), rotation dims set to
        min 0 / max 1 ("nullify action orientation normalization").
    none: BAKU's LIBERO loader: min 0 / max 1 everywhere (actions unchanged).
    minmax: per-dimension min/max, rotation included.
    symmetric: pose dims -s..+s with s = percentile of |a|; gripper 0/1.
    """
    stacked = np.concatenate(actions, axis=0)
    dims = stacked.shape[-1]
    if normalization == 'none':
        return np.zeros(dims, dtype=np.float32), np.ones(dims, dtype=np.float32)
    low = stacked.min(axis=0).astype(np.float32)
    high = stacked.max(axis=0).astype(np.float32)
    if normalization == 'xarm':
        low[3:6], high[3:6] = 0, 1
    elif normalization == 'symmetric':
        scale = np.percentile(np.abs(stacked[:, :POSE_DIMS]), percentile, axis=0)
        scale = np.maximum(scale, 1e-4).astype(np.float32)
        low[:POSE_DIMS], high[:POSE_DIMS] = -scale, scale
        low[POSE_DIMS:], high[POSE_DIMS:] = 0.0, 1.0
    elif normalization != 'minmax':
        raise ValueError('normalization must be xarm, none, minmax or symmetric, got %r' % normalization)
    return low, high


class BCDataset(IterableDataset):
    def __init__(
        self,
        path,
        tasks,
        num_demos_per_task,
        obs_type,
        history,
        history_len,
        prompt,
        temporal_agg,
        num_queries,
        img_size,
        pixel_keys,
        action_key="actions",
        normalization="symmetric",
        norm_percentile=99.5,
        augment=True,
        store_actions=True,
        demo_offset=0,
    ):
        if obs_type != "pixels":
            raise NotImplementedError("franka_teleop data has pixels only")
        if prompt not in (None, "text"):
            raise NotImplementedError("franka_teleop data supports prompt=text only")
        self._obs_type = obs_type
        self._prompt = prompt
        self._history_len = history_len if history else 1
        self._img_size = img_size
        self._pixel_keys = list(pixel_keys)
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries
        self._store_actions = store_actions

        path = Path(path)
        if not tasks:
            tasks = sorted(p.stem for p in path.glob("*.pkl"))
        if not tasks:
            raise FileNotFoundError("no <task>.pkl in %s" % path)
        self.task_names = list(tasks)
        self.task_texts, self.task_embs, self.episode_ids = [], [], []

        self._episodes = {}
        self._max_episode_len = 0
        self._max_state_dim = 8  # ee position (3) + quaternion (4) + gripper (1)
        self._num_samples = 0
        raw_actions = []
        proprio = []
        meta = None
        for task_index, task in enumerate(self.task_names):
            with open(path / ("%s.pkl" % task), "rb") as f:
                data = pkl.load(f)
            if data["img_size"] != img_size:
                raise ValueError("%s was built at %d px, config img_size is %d" % (task, data["img_size"], img_size))
            meta = meta or {k: data[k] for k in ("format", "rate_hz", "image_specs", "img_size")}
            self.task_texts.append(data["task"])
            self.task_embs.append(np.asarray(data["task_emb"], dtype=np.float32))
            chosen = range(demo_offset, min(demo_offset + num_demos_per_task, len(data["observations"])))
            self.episode_ids.append([data["episode_ids"][i] for i in chosen])
            self._episodes[task_index] = []
            for i in chosen:
                observation = dict(data["observations"][i])
                action = np.asarray(data[action_key][i], dtype=np.float32)
                proprio.append(np.concatenate([observation["cartesian_states"],
                                               observation["gripper_states"][:, None]], axis=1))
                # history padding: repeat the last observation; hold still with the last gripper state
                for key in self._pixel_keys + ["cartesian_states", "gripper_states"]:
                    observation[key] = np.concatenate(
                        [observation[key], [observation[key][-1]] * self._history_len], axis=0)
                hold = np.zeros_like(action[-1])
                hold[POSE_DIMS:] = action[-1, POSE_DIMS:]
                action = np.concatenate([action, [hold] * self._history_len], axis=0)
                raw_actions.append(action)  # padded, as the xArm loader computes its min/max
                self._episodes[task_index].append(
                    dict(observation=observation, action=action, task_emb=self.task_embs[-1]))
                self._max_episode_len = max(self._max_episode_len, len(observation[self._pixel_keys[0]]))
                self._num_samples += len(observation[self._pixel_keys[0]])
            if not self._episodes[task_index]:
                raise ValueError("task %s has no demos in range %d..%d" % (
                    task, demo_offset, demo_offset + num_demos_per_task))
            print("Loaded %s: %d demos (%s)" % (task, len(self._episodes[task_index]), data["task"]))

        low, high = action_stats(raw_actions, normalization, norm_percentile)
        proprio = np.concatenate(proprio, axis=0)
        p_low, p_high = proprio.min(axis=0), proprio.max(axis=0)
        p_low[3:7], p_high[3:7] = -1.0, 1.0  # quaternion
        p_low[7], p_high[7] = 0.0, 1.0       # gripper
        self.stats = {
            "actions": {"min": low, "max": high},
            "proprioceptive": {"min": p_low.astype(np.float32), "max": p_high.astype(np.float32)},
        }
        self.preprocess = {
            "actions": lambda x: (x - self.stats["actions"]["min"])
            / (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5),
            "proprioceptive": lambda x: (x - self.stats["proprioceptive"]["min"])
            / (self.stats["proprioceptive"]["max"] - self.stats["proprioceptive"]["min"] + 1e-5),
        }
        if self._store_actions:
            self.actions = [episode["action"] for task in self._episodes.values() for episode in task]
        self.dataset_meta = dict(meta, path=str(path), tasks=self.task_names, task_texts=self.task_texts,
                                 episode_ids=self.episode_ids, action_key=action_key,
                                 normalization=normalization, norm_percentile=norm_percentile)
        print("Action normalization (%s): min %s max %s" % (normalization, np.round(low, 4), np.round(high, 4)))

        steps = [transforms.ToPILImage()]
        if augment:
            steps += [transforms.RandomCrop(self._img_size, padding=4),
                      transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)]
        steps.append(transforms.ToTensor())
        self.aug = transforms.Compose(steps)
        self.envs_till_idx = len(self._episodes)

    def _sample_episode(self, env_idx=None):
        idx = random.randint(0, self.envs_till_idx - 1) if env_idx is None else env_idx
        episode = random.choice(self._episodes[idx])
        return (episode, idx) if env_idx is None else episode

    def _sample(self):
        episode, _ = self._sample_episode()
        observations = episode["observation"]
        actions = episode["action"]
        sample_idx = np.random.randint(0, len(observations[self._pixel_keys[0]]) - self._history_len)
        window = slice(sample_idx, sample_idx + self._history_len)

        sample = {}
        for key in self._pixel_keys:
            sample[key] = torch.stack([self.aug(image) for image in observations[key][window]])
        proprio = np.concatenate([observations["cartesian_states"][window],
                                  observations["gripper_states"][window][:, None]], axis=1)
        sample["proprioceptive"] = self.preprocess["proprioceptive"](proprio)

        if self._temporal_agg:
            num_actions = self._history_len + self._num_queries - 1
            act = np.repeat(actions[-1:], num_actions, axis=0)  # past the end: hold still
            available = actions[sample_idx:sample_idx + num_actions]
            act[:len(available)] = available
            sampled_action = np.lib.stride_tricks.sliding_window_view(
                act, (self._num_queries, actions.shape[-1]))[:, 0]
        else:
            sampled_action = actions[window]
        sample["actions"] = self.preprocess["actions"](sampled_action)
        sample["task_emb"] = episode["task_emb"]
        return sample

    def sample_test(self, env_idx, step=None):
        episode = self._sample_episode(env_idx)
        prompt = {"task_emb": episode["task_emb"], "prompt_proprioceptive": None, "prompt_actions": None}
        for key in self._pixel_keys:
            prompt["prompt_" + key] = None
        return prompt

    def __iter__(self):
        while True:
            yield self._sample()

    def __len__(self):
        return self._num_samples


def read_manifest(path):
    manifest = Path(path) / "dataset.json"
    return json.loads(manifest.read_text()) if manifest.exists() else {}
