"""Shared checkpoint identity for training and evaluation.

Only immutable architecture and data-contract fields belong here. Runtime-only
evaluation settings (for example, the number of rollouts) are deliberately not
part of the checkpoint identity.
"""


def build_checkpoint_metadata(
    cfg,
    agent,
    *,
    seed=None,
    include_feature_cache_identity=True,
):
    """Return the checkpoint identity implied by ``cfg`` and ``agent``.

    ``seed`` lets evaluation validate a trained model seed independently from
    the rollout seed. Cached training records the immutable pre-L2 cache
    identity; live evaluation omits those training-only fields while still
    validating every model and projector field.
    """

    encoder_type = str(cfg.encoder_type)
    is_final_conv = (
        encoder_type == "dinov2"
        and str(agent.dinov2_projector_type) == "final_conv_v1"
    )
    metadata = {
        "format_version": (
            4 if is_final_conv else 3 if encoder_type == "dinov2" else 2
        ),
        "step_semantics": "next_update",
        "encoder_type": encoder_type,
        "policy_type": str(cfg.policy_type),
        "policy_head": str(cfg.policy_head),
        "suite": str(cfg.suite.task.suite),
        "seed": int(cfg.seed if seed is None else seed),
        "batch_size": int(cfg.batch_size),
        "num_demos_per_task": int(cfg.num_demos_per_task),
        "vggt_intermediate_layers": [
            int(layer) for layer in cfg.vggt_intermediate_layers
        ],
        "use_proprio": bool(cfg.use_proprio),
        "use_language": bool(cfg.use_language),
        "temporal_agg": bool(cfg.temporal_agg),
        "num_queries": int(cfg.num_queries),
    }
    if encoder_type != "dinov2":
        return metadata

    metadata.update(
        {
            "dinov2_variant": str(agent.dinov2_variant),
            "dinov2_source_revision": str(agent.dinov2_source_revision),
            "dinov2_backbone_sha256": str(agent.dinov2_backbone_sha256),
            "dinov2_adapter_sha256": str(agent.dinov2_adapter_sha256),
            "dinov2_source_image_size": int(agent.dinov2_source_image_size),
            "dinov2_input_size": int(agent.dinov2_input_size),
            "dinov2_projector_type": str(agent.dinov2_projector_type),
            "dinov2_backbone_frozen": bool(agent.dinov2_backbone_frozen),
            "dinov2_adapter_frozen": True,
            "dinov2_projector_trainable": bool(agent.train_encoder),
            "prompt": None if cfg.prompt is None else str(cfg.prompt),
            "history": bool(cfg.suite.history),
            "history_len": int(cfg.suite.history_len),
            "eval_history_len": int(cfg.suite.eval_history_len),
            "pixel_keys": [str(key) for key in cfg.suite.pixel_keys],
            "norm": bool(cfg.agent.norm),
            "film": bool(cfg.agent.film),
            "separate_encoders": bool(cfg.agent.separate_encoders),
            "hidden_dim": int(cfg.suite.hidden_dim),
            "action_dim": int(agent._act_dim),
            "dataloader_img_size": int(cfg.dataloader.bc_dataset.img_size),
        }
    )
    if is_final_conv:
        # These v4-only fields make the new projector state contract
        # unambiguous while retaining exact v3 compatibility for historical
        # spatial_softmax_v1 checkpoints.
        metadata.update(
            {
                "dinov2_projector_descriptor": str(
                    agent.dinov2_projector_descriptor
                ),
                "dinov2_projector_parameter_count": int(
                    agent.dinov2_projector_parameter_count
                ),
                "dinov2_projector_state_shapes": dict(
                    agent.dinov2_projector_state_shapes
                ),
            }
        )
    if (
        include_feature_cache_identity
        and bool(cfg.dinov2_feature_cache_enabled)
    ):
        metadata.update(
            {
                "dinov2_training_feature_source": "fp16_cache_pre_l2",
                "dinov2_feature_cache_manifest_sha256": str(
                    cfg.dinov2_feature_cache_manifest_sha256
                ),
                "dinov2_feature_cache_normalization": (
                    "cast_fp16_to_fp32_then_channelwise_l2"
                ),
            }
        )
    return metadata


def checkpoint_metadata_mismatches(saved, expected):
    """Return fields whose saved values differ from the expected identity."""

    return {
        key: (saved.get(key), value)
        for key, value in expected.items()
        if saved.get(key) != value
    }
