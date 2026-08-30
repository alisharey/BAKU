# DINOv2 experiments

This code supports two frozen DINOv2-B/14 feature extractors and two trainable
policy projectors. Select them independently with Hydra overrides:

| Option | Values | Meaning |
| --- | --- | --- |
| `dinov2_variant` | `stock`, `3dcorr` | Official DINOv2-B/14 features, or the released 3DCorr LoRA/refinement adapter applied to that backbone |
| `dinov2_projector_type` | `spatial_softmax_v1`, `final_conv_v1` | A spatial-softmax projection, or LayerNorm followed by two 1D convolutions and global pooling |

The DINOv2 backbone and the 3DCorr adapter remain frozen. With
`agent.train_encoder=true`, only the selected 768-to-512 projector is trained
alongside the policy. Setting it to `false` also freezes the projector.

## Inputs

Set the repository and LIBERO dataset roots, then enter the training directory:

```bash
export BAKU_ROOT="$(git rev-parse --show-toplevel)"
export LIBERO_DATA_ROOT=/path/to/libero/datasets
cd "$BAKU_ROOT/baku"
```

The DINOv2 source checkout and weights are external artifacts; they are not
stored in this repository. The source checkout must be clean and at the pinned
revision. Record and pass the SHA-256 digest of every weight file:

```bash
export DINOV2_SOURCE=/path/to/dinov2
export DINOV2_REVISION=<git-commit>
export DINOV2_BACKBONE=/path/to/dinov2_vitb14_pretrain.pth
export DINOV2_BACKBONE_SHA256=<64-lowercase-hex-digest>

# Required when dinov2_variant=3dcorr.
export DINOV2_ADAPTER=/path/to/dinov2_base_3dcorr.ckpt
export DINOV2_ADAPTER_SHA256=<64-lowercase-hex-digest>
```

The defaults in `cfgs/config.yaml` use the repository-local `.cache/dinov2`
layout. Explicit overrides, as below, make the artifact identity visible in a
launch command.

## Train from live images

This LIBERO-10 example trains the deterministic GPT policy for 125,000 steps
and saves every 5,000 steps. Change `VARIANT` and `PROJECTOR` to run any of the
four combinations.

```bash
VARIANT=stock                 # stock or 3dcorr
PROJECTOR=final_conv_v1       # final_conv_v1 or spatial_softmax_v1

python train.py \
  agent=baku suite=libero dataloader=libero suite/task=libero_10 \
  encoder_type=dinov2 "dinov2_variant=$VARIANT" \
  "dinov2_projector_type=$PROJECTOR" \
  "dinov2_source_path=$DINOV2_SOURCE" \
  "dinov2_source_revision=$DINOV2_REVISION" \
  "dinov2_backbone_weight_path=$DINOV2_BACKBONE" \
  "dinov2_backbone_sha256=$DINOV2_BACKBONE_SHA256" \
  "dinov2_adapter_weight_path=$DINOV2_ADAPTER" \
  "dinov2_adapter_sha256=$DINOV2_ADAPTER_SHA256" \
  dinov2_source_image_size=126 dinov2_input_size=224 \
  dinov2_feature_cache_enabled=false \
  obs_type=pixels policy_type=gpt policy_head=deterministic \
  agent.train_encoder=true agent.norm=false agent.separate_encoders=false \
  batch_size=64 num_demos_per_task=50 \
  suite.num_train_steps=125000 suite.save_every_steps=5000
```

The adapter path and digest are used only by the `3dcorr` variant. Both
projectors consume a normalized 768-channel, 16-by-16 terminal feature map and
produce a 512-dimensional policy feature.

## Optional FP16 feature cache

Training can read prepared terminal DINO maps instead of repeatedly running
the frozen backbone. The cache stores pre-L2 FP16 maps; the dataset casts them
to FP32 and applies the same channel-wise L2 normalization as the live path.
The current cache reader expects the published LIBERO-10 cache format, with a
separate `manifest.json` under each variant directory.

Add these overrides to the training command:

```bash
export DINO_FEATURE_CACHE=/path/to/dinov2-feature-cache
export DINO_FEATURE_CACHE_MANIFEST_SHA256=<64-lowercase-hex-digest>

dinov2_feature_cache_enabled=true \
"dinov2_feature_cache_root=$DINO_FEATURE_CACHE" \
"dinov2_feature_cache_manifest_sha256=$DINO_FEATURE_CACHE_MANIFEST_SHA256"
```

The cache variant, source revision, backbone/adapter digests, feature shape,
dataset inventory, and shard digests are validated before use. Cache creation
is a separate step documented in [`tools/dinov2_cache/README.md`](tools/dinov2_cache/README.md).
Resume with the same cache identity that was recorded in the checkpoint.

## Evaluate from live observations

`eval_ray_split.py` divides the ten LIBERO tasks between two Ray actors. One
evaluator must see exactly one GPU. The example below evaluates a LIBERO-10
checkpoint on the original camera; set `VIEW` to `small`, `medium`, or `large`
for a configured novel view. Use a new output directory for every launch.

```bash
export CHECKPOINT=/path/to/125000.pt
export EVAL_OUTPUT=/path/to/new-evaluation-output
VARIANT=stock
PROJECTOR=final_conv_v1
MODEL_SEED=0
ROLLOUT_SEED=0
VIEW=null

CUDA_VISIBLE_DEVICES=0 python eval_ray_split.py \
  agent=baku suite=libero dataloader=libero suite/task=libero_10_eval2 \
  encoder_type=dinov2 "dinov2_variant=$VARIANT" \
  "dinov2_projector_type=$PROJECTOR" \
  "dinov2_source_path=$DINOV2_SOURCE" \
  "dinov2_source_revision=$DINOV2_REVISION" \
  "dinov2_backbone_weight_path=$DINOV2_BACKBONE" \
  "dinov2_backbone_sha256=$DINOV2_BACKBONE_SHA256" \
  "dinov2_adapter_weight_path=$DINOV2_ADAPTER" \
  "dinov2_adapter_sha256=$DINOV2_ADAPTER_SHA256" \
  dinov2_source_image_size=126 dinov2_input_size=224 \
  dinov2_feature_cache_enabled=false \
  obs_type=pixels policy_type=gpt policy_head=deterministic \
  agent.train_encoder=true agent.norm=false agent.separate_encoders=false \
  batch_size=64 num_demos_per_task=50 \
  "bc_weight=$CHECKPOINT" "model_seed=$MODEL_SEED" "seed=$ROLLOUT_SEED" \
  suite.num_eval_episodes=10 "camera_pose_variations=$VIEW" \
  "hydra.run.dir=$EVAL_OUTPUT"
```

`model_seed` is the seed stored in the trained checkpoint; `seed` controls the
rollout. Evaluation always uses live observations, even when training used the
optional cache.

## Checkpoint compatibility

Checkpoint loading validates the encoder variant, pinned artifact digests,
input sizes, projector type, policy/data settings, and trained seed. The two
projectors are not interchangeable: `spatial_softmax_v1` checkpoints use the
v3 DINO metadata contract, while `final_conv_v1` checkpoints use v4 and also
record the projector descriptor, parameter count, and state shapes. Frozen
DINO artifacts are not embedded in policy checkpoints, so the same pinned
source and weight files must be available for resume and live evaluation.

The split evaluator also checks that the checkpoint filename and saved step
match the suite's prescribed evaluation step. Do not rename a checkpoint to
bypass that check; use a checkpoint produced with the matching configuration.
