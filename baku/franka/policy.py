"""A trained Franka BAKU snapshot as a callable policy (offline check and robot).

    policy = FrankaPolicy('<run>/snapshot/60000.pt')
    policy.reset()
    action = policy.act({'pixels': cam_2_rgb, 'pixels_egocentric': cam_1_rgb})  # raw 640x480 RGB frames
    # action: [dx dy dz droll dpitch dyaw gripper] in robot units (TeleMoMa convention)
    policy.last  # VQ-BeT codebook entropies/probabilities and the action-token feature of this step

The entropies and the feature (the GPT action token, the input of the VQ-BeT
head) are read with a forward hook on the action head, so this works with
plain BAKU as well as versions with an interpretability monitor.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

BAKU_DIR = Path(__file__).resolve().parents[1]
if str(BAKU_DIR) not in sys.path:
    sys.path.insert(0, str(BAKU_DIR))

from franka import images  # noqa: E402


class FrankaPolicy:
    def __init__(self, snapshot, device="cuda", max_steps=1200, task=0):
        from omegaconf import OmegaConf
        import hydra

        snapshot = Path(snapshot).resolve()
        config = snapshot.parents[1] / ".hydra" / "config.yaml"
        if not config.exists():
            raise FileNotFoundError("%s not found (the snapshot must stay in its training run folder)" % config)
        cfg = OmegaConf.load(config)
        payload = torch.load(snapshot, map_location=device, weights_only=False)
        if "franka" not in payload:
            raise ValueError("%s was not trained with train_franka.py" % snapshot)
        self.info = payload["franka"]
        self.stats = payload["stats"]
        self.pixel_keys = list(self.info["pixel_keys"])
        self.size = int(self.info["img_size"])
        self.specs = {key: images.parse(spec) for key, (_, spec) in self.info["dataset"]["image_specs"].items()}
        self.cameras = {key: camera for key, (camera, _) in self.info["dataset"]["image_specs"].items()}
        self.rate_hz = float(self.info["dataset"]["rate_hz"])
        self.max_steps = int(max_steps)

        cfg.device = device
        obs_shape = {key: (3, self.size, self.size) for key in self.pixel_keys}
        if cfg.use_proprio:
            obs_shape[cfg.suite.proprio_key] = (8,)
        obs_shape[cfg.suite.feature_key] = (8,)
        cfg.agent.obs_shape = obs_shape
        cfg.agent.action_shape = (7,)
        cfg.agent.max_episode_len = self.max_steps
        self.agent = hydra.utils.instantiate(cfg.agent)
        self.agent.load_snapshot(payload, eval=True)
        self._head_io = {}
        self.agent.actor._action_head.register_forward_hook(self._capture)
        self.set_task(task)
        self.snapshot = str(snapshot)
        self.reset()

    def _capture(self, module, inputs, output):
        self._head_io = {"features": inputs[0].detach(), "output": output}

    def set_task(self, task):
        """Task index or one of the trained task texts/names."""
        if isinstance(task, str):
            names = self.info["task_names"] + self.info["task_texts"]
            if task not in names:
                raise ValueError("unknown task %r; trained tasks: %s" % (task, self.info["task_texts"]))
            task = names.index(task) % len(self.info["task_names"])
        self.task_index = int(task)
        self.task_text = self.info["task_texts"][self.task_index]
        self.prompt = {"task_emb": np.asarray(self.info["task_embs"][self.task_index], dtype=np.float32)}

    def reset(self):
        self.agent.buffer_reset()
        self.step_index = 0
        self.last = {}

    def preprocess(self, frames):
        """{key: H x W x 3 uint8 RGB camera frame} -> {key: 3 x S x S uint8}, as in the dataset."""
        return {key: images.apply(frames[key], self.specs[key], self.size).transpose(2, 0, 1)
                for key in self.pixel_keys}

    @torch.no_grad()
    def act(self, frames, preprocessed=False):
        if self.step_index >= self.max_steps:
            raise RuntimeError("max_steps (%d) reached; reset() to start a new episode" % self.max_steps)
        obs = frames if preprocessed else self.preprocess(frames)
        obs = dict(obs, features=np.zeros(8, dtype=np.float32))
        self._head_io = {}
        self.agent.train(False)
        action = self.agent.act(obs, self.prompt, self.stats, self.step_index, 0, eval_mode=True)
        self.step_index += 1
        self.last = self._summarize()
        return np.asarray(action, dtype=np.float64)

    def _summarize(self):
        """Entropy and probability of the sampled code of each VQ-BeT codebook (last
        time step), and the action-token feature, as interpretability/monitor.py logs them."""
        if not self._head_io:
            return {}
        out = {"feature": self._head_io["features"].reshape(-1, self._head_io["features"].shape[-1])[-1]
               .float().cpu().numpy()}
        output = self._head_io["output"]
        if not isinstance(output, dict):
            return out
        centers = output.get("sampled_centers")
        entropies, probs = [], []
        for index, name in enumerate(("cbet_logits1", "cbet_logits2")):
            logits = output.get(name)
            if logits is None:
                continue
            p = torch.softmax(logits.float(), dim=-1).reshape(-1, logits.shape[-1])[-1]
            entropies.append(float(-(p * torch.log(p + 1e-8)).sum()))
            if centers is not None:
                probs.append(float(p[int(centers.reshape(-1, centers.shape[-1])[-1, index])]))
        if entropies:
            out["token_entropy"] = entropies
            out["token_prob"] = probs
            out["entropy_avg"] = float(np.mean(entropies))
        return out

    def token_columns(self, count=7):
        """SAFE-compatible per-step entropy/probability columns: the two native VQ-BeT
        codebook values, the rest padded with their mean (interpretability/monitor.py)."""
        entropy = list(self.last.get("token_entropy") or [])
        prob = list(self.last.get("token_prob") or [])
        entropy = (entropy + entropy[-1:] * 2)[:2] if entropy else [math.nan, math.nan]
        if not prob:
            prob = [math.exp(-v) if not math.isnan(v) else math.nan for v in entropy]
        prob = (prob + prob[-1:] * 2)[:2]
        mean_entropy, mean_prob = float(np.nanmean(entropy)), float(np.nanmean(prob))
        return (entropy + [mean_entropy] * (count - 2), prob + [mean_prob] * (count - 2))
