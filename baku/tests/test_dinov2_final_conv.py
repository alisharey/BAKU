import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agent.networks.dinov2_encoder import (  # noqa: E402
    DINOv2FinalConvProjector,
    DINOv2SpatialProjector,
    build_dinov2_projector,
    describe_dinov2_projector,
    normalize_dinov2_dense,
)
from checkpoint_metadata import (  # noqa: E402
    build_checkpoint_metadata,
    checkpoint_metadata_mismatches,
)


class DINOv2FinalConvProjectorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_shape_state_contract_and_gradients(self):
        projector = DINOv2FinalConvProjector(input_size=224, output_dim=512)
        expected_state_shapes = {
            "token_norm.weight": (768,),
            "token_norm.bias": (768,),
            "conv1.weight": (512, 768, 3),
            "conv1.bias": (512,),
            "conv2.weight": (512, 512, 3),
            "conv2.bias": (512,),
        }
        self.assertEqual(
            {key: tuple(value.shape) for key, value in projector.state_dict().items()},
            expected_state_shapes,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in projector.parameters()),
            1_968_640,
        )
        self.assertEqual(
            describe_dinov2_projector(projector),
            {
                "type": "final_conv_v1",
                "descriptor": (
                    "token_layernorm_768__conv1d_768_512_k3p1_relu__"
                    "conv1d_512_512_k3p1_relu__adaptive_avg_pool1d_1"
                ),
                "parameter_count": 1_968_640,
                "state_shapes": {
                    key: list(shape) for key, shape in expected_state_shapes.items()
                },
            },
        )

        dense = torch.randn(2, 768, 16, 16, requires_grad=True)
        projected = projector(dense)
        self.assertEqual(tuple(projected.shape), (2, 512))
        projected.square().mean().backward()
        self.assertIsNotNone(dense.grad)
        self.assertTrue(
            all(parameter.grad is not None for parameter in projector.parameters())
        )

    def test_forward_matches_explicit_token_pipeline(self):
        projector = DINOv2FinalConvProjector(input_size=224, output_dim=512)
        dense = torch.randn(1, 768, 16, 16)

        actual = projector(dense)
        tokens = dense.flatten(2).transpose(1, 2)
        tokens = F.layer_norm(
            tokens,
            (768,),
            projector.token_norm.weight,
            projector.token_norm.bias,
            projector.token_norm.eps,
        ).transpose(1, 2)
        tokens = F.relu(
            F.conv1d(
                tokens,
                projector.conv1.weight,
                projector.conv1.bias,
                padding=1,
            )
        )
        tokens = F.relu(
            F.conv1d(
                tokens,
                projector.conv2.weight,
                projector.conv2.bias,
                padding=1,
            )
        )
        expected = F.adaptive_avg_pool1d(tokens, 1).squeeze(-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_fp16_cache_uses_same_fp32_l2_boundary(self):
        pre_l2 = torch.randn(2, 768, 16, 16).half()
        actual = normalize_dinov2_dense(pre_l2)
        expected = F.normalize(pre_l2.float(), p=2, dim=1)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        channel_norms = torch.linalg.vector_norm(actual, dim=1)
        torch.testing.assert_close(
            channel_norms,
            torch.ones_like(channel_norms),
            rtol=2e-6,
            atol=2e-6,
        )

    def test_live_and_cached_projection_parity_after_shared_l2(self):
        projector = DINOv2FinalConvProjector(input_size=224, output_dim=512)
        # Start from half-representable values so the test isolates routing and
        # normalization parity rather than expected cache quantization error.
        pre_l2 = torch.randn(1, 768, 16, 16).half().float()
        live_map = normalize_dinov2_dense(pre_l2)
        cached_map = normalize_dinov2_dense(pre_l2.half())
        torch.testing.assert_close(live_map, cached_map, rtol=0, atol=0)
        torch.testing.assert_close(
            projector(live_map),
            projector(cached_map),
            rtol=0,
            atol=0,
        )

    def test_named_factory_and_validation(self):
        final_conv = build_dinov2_projector("final_conv_v1", 224, 512)
        spatial = build_dinov2_projector("spatial_softmax_v1", 224, 512)
        self.assertIsInstance(final_conv, DINOv2FinalConvProjector)
        self.assertIsInstance(spatial, DINOv2SpatialProjector)
        self.assertEqual(final_conv.projector_name, "final_conv_v1")

        with self.assertRaisesRegex(ValueError, "unsupported DINOv2 projector"):
            build_dinov2_projector("unknown", 224, 512)
        with self.assertRaisesRegex(ValueError, "expected"):
            final_conv(torch.randn(1, 768, 9, 9))
        with self.assertRaisesRegex(ValueError, "output_dim=512"):
            DINOv2FinalConvProjector(input_size=224, output_dim=256)

    def test_projector_checkpoints_are_architecture_specific(self):
        final_conv = build_dinov2_projector("final_conv_v1", 224, 512)
        spatial = build_dinov2_projector("spatial_softmax_v1", 224, 512)
        with self.assertRaises(RuntimeError):
            final_conv.load_state_dict(spatial.state_dict(), strict=True)

    def test_checkpoint_metadata_rejects_spatial_projector(self):
        projector = DINOv2FinalConvProjector(input_size=224, output_dim=512)
        projector_metadata = describe_dinov2_projector(projector)
        agent = SimpleNamespace(
            dinov2_variant="stock",
            dinov2_source_revision="source-revision",
            dinov2_backbone_sha256="a" * 64,
            dinov2_adapter_sha256="none",
            dinov2_source_image_size=126,
            dinov2_input_size=224,
            dinov2_projector_type=projector_metadata["type"],
            dinov2_projector_descriptor=projector_metadata["descriptor"],
            dinov2_projector_parameter_count=projector_metadata[
                "parameter_count"
            ],
            dinov2_projector_state_shapes=projector_metadata["state_shapes"],
            dinov2_backbone_frozen=True,
            train_encoder=True,
            _act_dim=7,
        )
        cfg = SimpleNamespace(
            encoder_type="dinov2",
            policy_type="gpt",
            policy_head="deterministic",
            suite=SimpleNamespace(
                task=SimpleNamespace(suite="libero_10"),
                history=True,
                history_len=10,
                eval_history_len=10,
                pixel_keys=["pixels", "pixels_egocentric"],
                hidden_dim=256,
            ),
            seed=0,
            batch_size=64,
            num_demos_per_task=50,
            vggt_intermediate_layers=[4, 11, 17, 23],
            use_proprio=True,
            use_language=True,
            temporal_agg=True,
            num_queries=10,
            prompt="text",
            agent=SimpleNamespace(
                norm=False,
                film=True,
                separate_encoders=False,
            ),
            dataloader=SimpleNamespace(
                bc_dataset=SimpleNamespace(img_size=126)
            ),
            dinov2_feature_cache_enabled=True,
            dinov2_feature_cache_manifest_sha256="b" * 64,
        )

        metadata = build_checkpoint_metadata(cfg, agent)
        self.assertEqual(metadata["format_version"], 4)
        self.assertEqual(metadata["dinov2_projector_type"], "final_conv_v1")
        self.assertEqual(
            metadata["dinov2_projector_parameter_count"], 1_968_640
        )
        self.assertEqual(
            metadata["dinov2_training_feature_source"],
            "fp16_cache_pre_l2",
        )

        spatial_projector = DINOv2SpatialProjector(input_size=224, output_dim=512)
        spatial_projector_metadata = describe_dinov2_projector(
            spatial_projector
        )
        spatial_agent = SimpleNamespace(**vars(agent))
        spatial_agent.dinov2_projector_type = spatial_projector_metadata["type"]
        spatial_agent.dinov2_projector_descriptor = spatial_projector_metadata[
            "descriptor"
        ]
        spatial_agent.dinov2_projector_parameter_count = spatial_projector_metadata[
            "parameter_count"
        ]
        spatial_agent.dinov2_projector_state_shapes = spatial_projector_metadata[
            "state_shapes"
        ]
        spatial_metadata = build_checkpoint_metadata(cfg, spatial_agent)
        self.assertEqual(spatial_metadata["format_version"], 3)
        self.assertNotIn("dinov2_projector_descriptor", spatial_metadata)
        mismatches = checkpoint_metadata_mismatches(
            spatial_metadata, metadata
        )
        self.assertEqual(mismatches["format_version"], (3, 4))
        self.assertEqual(
            mismatches["dinov2_projector_type"],
            ("spatial_softmax_v1", "final_conv_v1"),
        )

        # Live evaluation deliberately omits training-cache identity, so a
        # cached-training checkpoint remains compatible after every model and
        # projector field has been validated.
        eval_metadata = build_checkpoint_metadata(
            cfg,
            agent,
            seed=0,
            include_feature_cache_identity=False,
        )
        self.assertNotIn("dinov2_training_feature_source", eval_metadata)
        self.assertEqual(
            checkpoint_metadata_mismatches(metadata, eval_metadata), {}
        )


if __name__ == "__main__":
    unittest.main()
