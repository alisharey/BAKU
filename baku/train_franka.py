#!/usr/bin/env python3
"""BAKU training on Franka demos (cfgs/config_franka.yaml). Same loop as train.py,
plus:

* init_from=<snapshot.pt>: start from another BAKU (e.g. LIBERO). Loads the
  image encoder, GPT trunk and language projector; never the VQ-VAE (train.py's
  load_bc would replace the codebook just fit to our actions with LIBERO's);
  the VQ-BeT heads only with init_action_head=true.
* Every snapshot also stores what deployment needs: task texts and embeddings,
  image preprocessing, dataset rate and the action convention ('franka' key).

    cd BAKU/baku
    python train_franka.py dataloader.bc_dataset.path=/abs/expert_demos/franka_teleop \
        init_from=/abs/exp_local/libero_resnet_train/<run>/snapshot/150000.pt
"""

import os
from pathlib import Path

import hydra
import torch

import train as baku_train


class WorkspaceFranka(baku_train.WorkspaceIL):
    def __init__(self, cfg):
        super().__init__(cfg)  # loads the data and fits the VQ-VAE to these actions
        dataset = self.expert_replay_loader.dataset
        self.franka = {
            "dataset": dataset.dataset_meta,
            "task_names": dataset.task_names,
            "task_texts": dataset.task_texts,
            "task_embs": [e.tolist() for e in dataset.task_embs],
            "pixel_keys": list(cfg.suite.pixel_keys),
            "img_size": int(cfg.dataloader.bc_dataset.img_size),
            "num_queries": int(cfg.num_queries),
            "action": "TeleMoMa [dx dy dz droll dpitch dyaw gripper] base frame, target - current, "
                      "rotation scipy xyz Euler of R_target R_current^T, gripper 1 = open, 0 = closed"
            if dataset.dataset_meta["action_key"] == "actions" else dataset.dataset_meta["action_key"],
            "init_from": cfg.init_from,
        }
        if cfg.init_from:
            self.init_from(Path(cfg.init_from), cfg.init_action_head)

    def init_from(self, path, load_action_head):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        agent = self.agent
        report = []

        def load(module, state, name, skip_prefixes=()):
            own = module.state_dict()
            kept = {k: v for k, v in state.items()
                    if k in own and own[k].shape == v.shape and not k.startswith(skip_prefixes)}
            module.load_state_dict(kept, strict=False)
            report.append("%s: %d/%d tensors" % (name, len(kept), len(own)))

        if agent.separate_encoders:
            for key in agent.pixel_keys:
                load(agent.encoder[key], payload["encoder_%s" % key], "encoder_%s" % key)
        else:
            load(agent.encoder, payload["encoder"], "encoder")
        if agent.use_language and "language_projector" in payload:
            load(agent.language_projector, payload["language_projector"], "language_projector")
        if agent.use_proprio and "proprio_projector" in payload:
            load(agent.proprio_projector, payload["proprio_projector"], "proprio_projector")
        load(agent.actor, payload["actor"], "actor",
             skip_prefixes=() if load_action_head else ("_action_head.",))
        agent.reinit_optimizers()
        print("init_from %s (VQ-VAE kept from our data; action head %s):\n  %s" % (
            path, "loaded" if load_action_head else "re-initialized", "\n  ".join(report)))

    def save_snapshot(self):
        snapshot_dir = self.work_dir / "snapshot"
        snapshot_dir.mkdir(exist_ok=True)
        self.agent.clear_buffers()
        payload = {k: self.__dict__[k] for k in ["timer", "_global_step", "_global_episode", "stats"]}
        payload.update(self.agent.save_snapshot())
        payload["franka"] = self.franka
        with (snapshot_dir / f"{self.global_step}.pt").open("wb") as f:
            torch.save(payload, f)
        self.agent.buffer_reset()


@hydra.main(config_path="cfgs", config_name="config_franka")
def main(cfg):
    if cfg.load_bc:
        raise ValueError("use init_from=<snapshot> here (load_bc would also load the other VQ-VAE)")
    workspace = WorkspaceFranka(cfg)
    workspace.train()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
