# BAKU on a Franka FR3 (branch `franka-teleop`)

Plain BAKU plus files that fine-tune it on demonstrations recorded with
[franka_teleop](https://github.com/alisharey/franka_teleop) (Meta Quest teleop,
TeleMoMa actions, two RealSense cameras). Nothing in BAKU itself is changed. The
robot side (camera topics, sending actions to the arm) is in franka_teleop
(`vla/baku/deploy.py`), which includes this branch as the submodule `vla/BAKU`.

| File | Does |
|---|---|
| `baku/franka/build_dataset.py` | exported episodes -> `<task>.pkl` (BAKU's expert-demo format) |
| `baku/franka/images.py` | camera frame -> 128x128 (resize / center_crop / crop:x,y,w,h / pad); same code in training and on the robot |
| `baku/franka/preview_images.py` | side-by-side preview of the image options |
| `baku/read_data/franka_teleop.py` | dataset (actions taken as recorded; normalization below) |
| `baku/suite/franka_teleop.py` | observation/action specs for `train.py` (no simulator) |
| `baku/train_franka.py`, `baku/cfgs/config_franka.yaml` (+ `suite/`, `suite/task/`, `dataloader/franka_teleop.yaml`) | training |
| `baku/franka/policy.py` | a snapshot as a callable policy (preprocessing, act, VQ-BeT entropies) |
| `baku/franka/offline_eval.py` | runs a snapshot over recorded demos step by step and compares with the labels |

## Actions

Each step's action is TeleMoMa's, as the teleop executed it:
`[dx dy dz droll dpitch dyaw gripper]`, base frame, `d* = target - current`
(position in m; rotation = scipy "xyz" Euler of `R_target R_current^T`), pose
part 0 while the arm was not driven (grip released), gripper 1 = open,
0 = closed. Because it is a target offset (not a per-step delta), it does not
depend on the step rate, so subsampling 30 Hz -> 10 Hz keeps its meaning. On the
robot each action becomes a target (pose + action) that the teleop controller
tracks, exactly as during recording. `delta_pose_actions` (next-step delta
pose) is also stored for comparison (`dataloader.bc_dataset.action_key`).

### Normalization

BAKU's loaders all use the same min-max step on actions, (x - min) / (max - min),
undone in `agent.act` with the same stats (saved in every snapshot). The LIBERO
loader sets min 0 / max 1 (actions unchanged). The xArm (real robot) loader
takes min/max per dimension from the demo actions and sets the rotation
dimensions to 0 / 1. This loader does the same as the xArm one by default
(`normalization: xarm`); `none` = LIBERO's.

### VQ-BeT and starting from a LIBERO snapshot

The VQ-VAE (action codebook) is fit to these actions at the start of every run.
`train.py`'s `load_bc` would then overwrite it with the snapshot's LIBERO
codebook. Use `init_from=<snapshot>` instead: it loads the image encoder, GPT
trunk and language projector, keeps our VQ-VAE and re-initializes the VQ-BeT
heads (`init_action_head=true` loads them too). The image keys match LIBERO's:
`pixels` = third-person camera (cam_2), `pixels_egocentric` = wrist (cam_1).

## Steps

```bash
# robot PC (franka_teleop): export episodes and upload them (no depth)
vla/baku/upload_episodes.sh --task "pick black block and place in front of the white cup" ~/franka_teleop_data/2026*_episode_*

# cluster, on a compute node, in this checkout's baku/ folder, env `baku`
python franka/preview_images.py ../expert_demos/franka_teleop_export/<episode> --out preview.png \
    --spec resize --spec center_crop --spec crop:200,0,440,440
python franka/build_dataset.py ../expert_demos/franka_teleop_export/* --out ../expert_demos/franka_teleop \
    --subsample 3 --agent-image resize --wrist-image resize
python train_franka.py dataloader.bc_dataset.path=$PWD/../expert_demos/franka_teleop \
    init_from=<LIBERO run>/snapshot/<step>.pt     # or without init_from
python franka/offline_eval.py exp_local/franka_teleop_resnet_train/<run>/snapshot/<step>.pt \
    --data ../expert_demos/franka_teleop --plot eval.png

# robot PC: fetch the snapshot and run it (see franka_teleop vla/baku/README)
```
