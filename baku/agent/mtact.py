import numpy as np
from collections import deque

import torch

import utils
from agent.networks.utils.act.policy import ACTPolicy
from agent.networks.mlp import MLP


class Agent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        use_tb,
        lr,
        num_queries,
        kl_weight,
        hidden_dim,
        dim_feedforward,
        lr_backbone,
        backbone,
        enc_layers,
        dec_layers,
        nheads,
        use_proprio,
        pixel_keys,
        proprio_key,
        feature_key,
        temporal_agg,
        episode_len,
        multitask,
        obs_type,
    ):
        self.device = device
        self.use_tb = use_tb
        self.lr = lr
        self.num_queries = num_queries
        self.use_proprio = use_proprio
        self.pixel_keys = pixel_keys
        self.proprio_key = proprio_key
        self.feature_key = feature_key
        self.temporal_agg = temporal_agg
        self.proprioceptive_dim = obs_shape[self.proprio_key][0] if use_proprio else 1
        self.multitask = multitask
        self.obs_type = obs_type

        self.language_dim = 384

        if self.obs_type == "features":
            self.proprioceptive_dim = obs_shape[self.feature_key][0]

        # Query frequency for evaluation
        self.query_freq = 1 if self.temporal_agg else self.num_queries
        if self.temporal_agg:
            self.all_time_actions = torch.zeros(
                [episode_len, episode_len + num_queries, action_shape[0]]
            ).to(device)

        # policy config
        policy_config = {
            "lr": lr,
            "num_queries": num_queries if self.temporal_agg else 1,
            "kl_weight": kl_weight,
            "hidden_dim": hidden_dim,
            "dim_feedforward": dim_feedforward,
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": pixel_keys,
            "state_dim": self.proprioceptive_dim,
            "action_dim": action_shape[0],
            "multitask": self.multitask,
            "obs_type": self.obs_type,
        }

        # actor
        self.actor = ACTPolicy(policy_config, device)

        # task_env projector
        if self.multitask:
            self.language_projector = MLP(
                self.language_dim, hidden_channels=[hidden_dim, hidden_dim]
            ).to(device)
            self.language_projector.apply(utils.weight_init)

        # optimizers
        self.optimizer = self.actor.configure_optimizers()
        if self.multitask:
            self.optimizer.add_param_group(
                {"params": self.language_projector.parameters()}
            )
        # self.print_optimizer_info()
        self.train()
        self.buffer_reset()

    def __repr__(self):
        return "mtact"
    
    def print_optimizer_info(self):
    
        print("=" * 50)
        print("OPTIMIZER PARAMETER GROUPS")
        print("=" * 50)
        
        total_params = 0
        for i, param_group in enumerate(self.optimizer.param_groups):
            group_params = 0
            print(f"\nGroup {i}:")
            print(f"  Learning rate: {param_group['lr']}")
            print("  Parameters:")
            
            for param in param_group['params']:
                param_count = param.numel()
                group_params += param_count
                
                # Find parameter name
                param_name = "unknown"
                for name, module_param in self.actor.named_parameters():
                    if param is module_param:
                        param_name = f"actor.{name}"
                        break
                
                if self.multitask:
                    for name, module_param in self.language_projector.named_parameters():
                        if param is module_param:
                            param_name = f"language_projector.{name}"
                            break
                
                print(f"    - {param_name}: {param.shape} ({param_count:,} params)")
            
            print(f"  Group total: {group_params:,} parameters")
            total_params += group_params
        
        print(f"\nTotal optimizable parameters: {total_params:,}")
        print("=" * 50)
    

    def train(self, training=True):
        self.training = training
        if training:
            self.actor.train()
        else:
            self.actor.eval()

    def buffer_reset(self):
        self.observations_buffer = {} if self.obs_type == "pixels" else deque(maxlen=1)

    def clear_buffers(self):
        del self.observations_buffer

    def act(self, observations, prompt, norm_stats, step, global_step, eval_mode=False):
        if norm_stats is not None:
            pre_process = lambda s_qpos: (
                s_qpos - norm_stats[self.proprio_key]["min"]
            ) / (
                norm_stats[self.proprio_key]["max"]
                - norm_stats[self.proprio_key]["min"]
                + 1e-5
            )
            post_process = (
                lambda a: a
                * (norm_stats["actions"]["max"] - norm_stats["actions"]["min"])
                + norm_stats["actions"]["min"]
            )

        # lang projection
        if self.multitask:
            task_emb = torch.as_tensor(
                prompt["task_emb"], device=self.device,
            ).float()[None]
            task_emb = self.language_projector(task_emb)
        else:
            task_emb = None

        if step % self.query_freq == 0:
            # Only compute action once every num_queries steps

            if self.obs_type == "pixels" and len(self.observations_buffer.keys()) == 0:
                for key in observations.keys():
                    self.observations_buffer[key] = deque(
                        maxlen=1
                    )
            elif self.obs_type == "features" and len(self.observations_buffer) == 0:
                self.observations_buffer = deque(maxlen=1)

            if self.obs_type == "pixels":
                # arrange observations
                obs, proprio = [], []
                for key in observations.keys():
                    self.observations_buffer[key].append(observations[key])
                    if "pixels" in key:
                        obs.append(
                            torch.as_tensor(
                                np.array(self.observations_buffer[key]),
                                device=self.device,
                            ).float()
                        )
                        obs[-1] = obs[-1].unsqueeze(0)

                # concat obs
                obs = torch.cat(obs, dim=1)

                # preprocess
                if self.use_proprio:
                    proprio = np.array(self.observations_buffer[self.proprio_key])
                    if norm_stats is not None:
                        proprio = pre_process(proprio)
                    proprio = (
                        torch.as_tensor(proprio, device=self.device)
                        .float()
                        .unsqueeze(0)
                    )
                else:
                    proprio = (
                        torch.zeros(1, 1, self.proprioceptive_dim)
                        .to(self.device)
                        .float()
                    )
            else:
                obs = None
                self.observations_buffer.append(observations[self.feature_key])
                proprio = torch.as_tensor(
                    np.array(self.observations_buffer), device=self.device
                ).float()

            if self.obs_type == "pixels":
                obs = obs / 255.0
                proprio = proprio[:, 0]

            self.action = self.actor(proprio, obs, task_emb=task_emb)

        if self.temporal_agg:
            self.all_time_actions[[step], step : step + self.num_queries] = self.action
            actions_for_curr_step = self.all_time_actions[:, step]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
            self.action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)

            if norm_stats is not None:
                return post_process(self.action.cpu().numpy()[0])
            return self.action.cpu().numpy()[0]
        else:
            if norm_stats is not None:
                return post_process(self.action.cpu().numpy()[0, -1, :])
            return self.action.cpu().numpy()[0, -1, :]

    def update(self, expert_replay_iter, step):
        metrics = dict()

        batch = next(expert_replay_iter)
        data = utils.to_torch(batch, self.device)
        action = data["actions"].float()
        if len(data["actions"].shape) == 4:
            action = action[:, 0]
        is_pad = torch.zeros(action.shape[0], action.shape[1], dtype=torch.bool).to(
            self.device
        )

        # lang projection
        if self.multitask:
            task_emb = data["task_emb"].float()
            task_emb = self.language_projector(task_emb)
        else:
            task_emb = None

        # arrange observations and proprioceptive states
        if self.obs_type == "pixels":
            observation = []
            for key in self.pixel_keys:
                observation.append(data[key].float())
            observation = torch.cat(observation, dim=1)
            proprioceptive = (
                data[self.proprio_key].float()
                if self.use_proprio
                else torch.zeros(observation.shape[0], 1, self.proprioceptive_dim)
                .to(self.device)
                .float()
            )
        else:
            observation = None
            proprioceptive = data[self.feature_key].float()

        if self.obs_type == "pixels":
            observation = observation / 255.0
        proprioceptive = proprioceptive[:, 0]

        # forward pass
        output = self.actor(
            proprioceptive, observation, action, is_pad, task_emb=task_emb
        )

        # optimize
        self.optimizer.zero_grad(set_to_none=True)
        output["loss"].backward()
        self.optimizer.step()

        if self.use_tb:
            metrics["actor_loss"] = output["loss"].item()
            metrics["actor_l1"] = output["l1"].item()
            metrics["actor_kl"] = output["kl"].item()

        return metrics

    def save_snapshot(self):
        """
        Saves a snapshot of the model state, only including trainable parameters.
        This excludes large frozen backbones like VGGT.
        """
        payload = dict()
        # Save actor's trainable parameters
        payload["actor"] = {
            name: param.cpu()
            for name, param in self.actor.named_parameters()
            if param.requires_grad
        }
        # Save language projector's parameters if it exists
        if self.multitask:
            payload["language_projector"] = self.language_projector.state_dict()
        return payload

    def load_snapshot(self, payload, eval=True):
        """
        Loads a snapshot into the model.
        Uses strict=False to allow loading a state_dict that is a subset of the model's
        parameters, which is necessary because the frozen backbone is not saved.
        """
        if "actor" in payload:
            self.actor.load_state_dict(payload["actor"], strict=False)
        
        if "language_projector" in payload and self.multitask:
            self.language_projector.load_state_dict(payload["language_projector"])

        self.train()

        # It's good practice to re-initialize the optimizer after loading a new state
        self.actor_opt = self.actor.optimizer
        

    def load_snapshot_eval(self, payload, bc=False):
        for k, v in payload.items():
            self.__dict__[k] = v
