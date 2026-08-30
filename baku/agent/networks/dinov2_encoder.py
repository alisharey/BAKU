"""Frozen DINOv2 feature extractors for BAKU policies.

The stock and 3DCorrEnhance variants deliberately share the same trainable
policy projector.  The DINOv2 backbone, the released LoRA layers, and the
released refinement convolution are external frozen artifacts and therefore
are not included in BAKU policy snapshots.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


DINOV2_EMBED_DIM = 768
DINOV2_PATCH_SIZE = 14
SUPPORTED_VARIANTS = ("stock", "3dcorr")
SUPPORTED_PROJECTORS = ("spatial_softmax_v1", "final_conv_v1")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path, expected_sha256, label):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected_sha256} "
            f"actual={actual_sha256} path={path}"
        )
    return path


def _verify_source_revision(source_path, expected_revision):
    source_path = Path(source_path).expanduser().resolve()
    if not (source_path / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv2 source checkout is invalid: {source_path}")
    try:
        actual_revision = subprocess.check_output(
            ["git", "-C", str(source_path), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot verify DINOv2 source revision at {source_path}"
        ) from error
    if actual_revision != expected_revision:
        raise ValueError(
            "DINOv2 source revision mismatch: "
            f"expected={expected_revision} actual={actual_revision}"
        )
    dirty_tracked_files = subprocess.check_output(
        [
            "git",
            "-C",
            str(source_path),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()
    if dirty_tracked_files:
        raise ValueError(
            "DINOv2 source checkout has modified tracked files: "
            f"{dirty_tracked_files}"
        )
    return source_path


def _import_dinov2_vitb14(source_path):
    source_text = str(source_path)
    loaded = sys.modules.get("dinov2")
    if loaded is not None:
        loaded_file = Path(loaded.__file__).resolve()
        if source_path not in loaded_file.parents:
            raise RuntimeError(
                "a different 'dinov2' package is already imported: "
                f"{loaded_file}; expected source under {source_path}"
            )
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from dinov2.hub.backbones import dinov2_vitb14

    return dinov2_vitb14


class _LoRAQKV(nn.Module):
    """Apply the released rank-r Q/V updates to one DINOv2 QKV layer."""

    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, inputs):
        qkv = self.qkv(inputs)
        qkv[..., : self.dim] += self.linear_b_q(self.linear_a_q(inputs))
        qkv[..., -self.dim :] += self.linear_b_v(self.linear_a_v(inputs))
        return qkv


class FrozenDINOv2(nn.Module):
    """Pinned, immutable DINOv2-B/14 dense feature extractor."""

    def __init__(
        self,
        variant,
        source_path,
        source_revision,
        backbone_weight_path,
        backbone_sha256,
        adapter_weight_path,
        adapter_sha256,
        source_image_size=126,
        input_size=126,
        lora_rank=4,
        adapted_block_count=4,
    ):
        super().__init__()
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(
                f"unsupported DINOv2 variant {variant!r}; choose {SUPPORTED_VARIANTS}"
            )
        if source_image_size <= 0:
            raise ValueError("source_image_size must be positive")
        if input_size <= 0 or input_size % DINOV2_PATCH_SIZE != 0:
            raise ValueError(
                f"DINOv2 input_size must be positive and divisible by "
                f"{DINOV2_PATCH_SIZE}, found {input_size}"
            )
        if adapted_block_count <= 0 or adapted_block_count > 12:
            raise ValueError("adapted_block_count must be in [1, 12]")

        self.variant = variant
        self.source_image_size = int(source_image_size)
        self.input_size = int(input_size)
        self.grid_size = self.input_size // DINOV2_PATCH_SIZE
        self.source_revision = str(source_revision)
        self.backbone_sha256 = str(backbone_sha256)
        self.adapter_sha256 = (
            str(adapter_sha256) if self.variant == "3dcorr" else "none"
        )

        source_path = _verify_source_revision(source_path, self.source_revision)
        backbone_weight_path = _verify_file(
            backbone_weight_path, self.backbone_sha256, "DINOv2 backbone checkpoint"
        )
        dinov2_vitb14 = _import_dinov2_vitb14(source_path)

        # Model construction initializes parameters before replacing them with the
        # immutable checkpoint.  Preserve the caller's RNG so the stock/adapted
        # runs get identical policy-projector and actor initialization.
        with torch.random.fork_rng(devices=[]):
            backbone = dinov2_vitb14(pretrained=False)
            backbone_state = torch.load(
                backbone_weight_path, map_location="cpu", weights_only=True
            )
            backbone.load_state_dict(backbone_state, strict=True)

            refiner = None
            if self.variant == "3dcorr":
                adapter_weight_path = _verify_file(
                    adapter_weight_path,
                    self.adapter_sha256,
                    "3DCorrEnhance adapter checkpoint",
                )
                adapter_state = torch.load(
                    adapter_weight_path, map_location="cpu", weights_only=True
                )
                blocks = list(backbone.blocks[-adapted_block_count:])
                a_layers = []
                b_layers = []
                for block in blocks:
                    dim = block.attn.qkv.in_features
                    linear_a_q = nn.Linear(dim, lora_rank, bias=False)
                    linear_b_q = nn.Linear(lora_rank, dim, bias=False)
                    linear_a_v = nn.Linear(dim, lora_rank, bias=False)
                    linear_b_v = nn.Linear(lora_rank, dim, bias=False)
                    block.attn.qkv = _LoRAQKV(
                        block.attn.qkv,
                        linear_a_q,
                        linear_b_q,
                        linear_a_v,
                        linear_b_v,
                    )
                    a_layers.extend((linear_a_q, linear_a_v))
                    b_layers.extend((linear_b_q, linear_b_v))

                with torch.no_grad():
                    for index, layer in enumerate(a_layers):
                        key = f"w_a_{index:03d}"
                        value = adapter_state.get(key)
                        if value is None or value.shape != layer.weight.shape:
                            raise ValueError(
                                f"invalid 3DCorrEnhance tensor {key}: "
                                f"expected={tuple(layer.weight.shape)} "
                                f"actual={None if value is None else tuple(value.shape)}"
                            )
                        layer.weight.copy_(value)
                    for index, layer in enumerate(b_layers):
                        key = f"w_b_{index:03d}"
                        value = adapter_state.get(key)
                        if value is None or value.shape != layer.weight.shape:
                            raise ValueError(
                                f"invalid 3DCorrEnhance tensor {key}: "
                                f"expected={tuple(layer.weight.shape)} "
                                f"actual={None if value is None else tuple(value.shape)}"
                            )
                        layer.weight.copy_(value)

                refiner_state = adapter_state.get("state_dict", {}).get(
                    "refine_conv"
                )
                if not isinstance(refiner_state, dict):
                    raise ValueError(
                        "3DCorrEnhance checkpoint is missing state_dict.refine_conv"
                    )
                refiner = nn.Conv2d(
                    DINOV2_EMBED_DIM,
                    DINOV2_EMBED_DIM,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
                refiner.load_state_dict(refiner_state, strict=True)

        self.backbone = backbone
        self.refiner = refiner
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.requires_grad_(False)
        self._input_range_validated = False
        self.eval()

    def train(self, mode=True):
        # This module is an immutable feature extractor even while its owning
        # policy is in training mode.
        super().train(False)
        return self

    def _prepare_images(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"DINOv2 expects NCHW RGB images, found {tuple(images.shape)}"
            )
        if not self._input_range_validated:
            minimum = float(images.detach().amin())
            maximum = float(images.detach().amax())
            if minimum < -1e-4 or maximum > 1.0001:
                raise ValueError(
                    "DINOv2 expects float images in [0, 1], found "
                    f"minimum={minimum:.6f} maximum={maximum:.6f}"
                )
            self._input_range_validated = True
        height, width = images.shape[-2:]
        if height >= self.source_image_size and width >= self.source_image_size:
            top = (height - self.source_image_size) // 2
            left = (width - self.source_image_size) // 2
            images = images[
                :,
                :,
                top : top + self.source_image_size,
                left : left + self.source_image_size,
            ]
        else:
            images = F.interpolate(
                images,
                size=(self.source_image_size, self.source_image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        if self.source_image_size != self.input_size:
            images = F.interpolate(
                images,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return (images - self.pixel_mean) / self.pixel_std

    def extract_pre_l2(self, images):
        """Return the terminal dense map immediately before descriptor L2.

        This is the canonical cache boundary.  Stock returns reshaped
        ``x_norm_patchtokens``.  3DCorr returns those terminal tokens after the
        released frozen 3x3 refinement convolution.  Keeping normalization out
        of the cache lets the normal live and cached paths share the exact same
        FP32 L2 operation.
        """
        images = self._prepare_images(images.float())
        outputs = self.backbone.forward_features(images)
        patch_tokens = outputs["x_norm_patchtokens"]
        expected_patches = self.grid_size * self.grid_size
        if patch_tokens.shape[1:] != (expected_patches, DINOV2_EMBED_DIM):
            raise RuntimeError(
                "unexpected DINOv2 patch-token shape: "
                f"expected=(*,{expected_patches},{DINOV2_EMBED_DIM}) "
                f"actual={tuple(patch_tokens.shape)}"
            )
        dense = patch_tokens.reshape(
            images.shape[0], self.grid_size, self.grid_size, DINOV2_EMBED_DIM
        ).permute(0, 3, 1, 2)
        if self.refiner is not None:
            dense = self.refiner(dense)
        return dense

    def forward(self, images):
        dense = self.extract_pre_l2(images)
        # The released correspondence inference path L2-normalizes descriptors.
        return normalize_dinov2_dense(dense)


def normalize_dinov2_dense(dense_features):
    """Apply the canonical post-cache DINO descriptor normalization.

    Cached features are stored as pre-L2 FP16 maps, while live features arrive
    as pre-L2 FP32 maps.  Casting first and sharing this function guarantees
    that both routes use the same FP32 channelwise normalization.
    """
    if dense_features.ndim != 4 or dense_features.shape[1] != DINOV2_EMBED_DIM:
        raise ValueError(
            "expected a DINOv2 dense map [N,768,H,W], found "
            f"{tuple(dense_features.shape)}"
        )
    return F.normalize(dense_features.float(), p=2, dim=1)


class DINOv2SpatialProjector(nn.Module):
    """BAKU-style spatial-softmax projection from DINO patches to one token."""

    projector_name = "spatial_softmax_v1"
    projector_descriptor = (
        "conv2d_768_256_k1__spatial_softmax_xy__linear_512_512"
    )

    def __init__(self, input_size=126, output_dim=512):
        super().__init__()
        if input_size % DINOV2_PATCH_SIZE != 0:
            raise ValueError(
                f"input_size must be divisible by {DINOV2_PATCH_SIZE}"
            )
        grid_size = input_size // DINOV2_PATCH_SIZE
        if output_dim % 2 != 0:
            raise ValueError("output_dim must be even for spatial softmax")
        self.input_size = int(input_size)
        self.grid_size = int(grid_size)
        self.output_dim = int(output_dim)
        self.num_keypoints = output_dim // 2
        self.spatial_conv = nn.Conv2d(
            DINOV2_EMBED_DIM, self.num_keypoints, kernel_size=1
        )
        self.projection = nn.Linear(self.num_keypoints * 2, output_dim)

        y_coordinates = torch.linspace(-1.0, 1.0, self.grid_size)
        x_coordinates = torch.linspace(-1.0, 1.0, self.grid_size)
        position_y, position_x = torch.meshgrid(
            y_coordinates, x_coordinates, indexing="ij"
        )
        self.register_buffer("position_x", position_x.reshape(1, 1, -1))
        self.register_buffer("position_y", position_y.reshape(1, 1, -1))

    def forward(self, dense_features):
        expected = (
            DINOV2_EMBED_DIM,
            self.grid_size,
            self.grid_size,
        )
        if dense_features.ndim != 4 or dense_features.shape[1:] != expected:
            raise ValueError(
                f"DINOv2 projector expected (N,{expected[0]},{expected[1]},"
                f"{expected[2]}), found {tuple(dense_features.shape)}"
            )
        heatmaps = self.spatial_conv(dense_features)
        attention = F.softmax(heatmaps.flatten(2), dim=-1)
        keypoint_x = (attention * self.position_x).sum(dim=-1)
        keypoint_y = (attention * self.position_y).sum(dim=-1)
        keypoints = torch.cat((keypoint_x, keypoint_y), dim=-1)
        return self.projection(keypoints)


class DINOv2FinalConvProjector(nn.Module):
    """Content-preserving projection of the normalized terminal dense map.

    Tokens are normalized independently across their 768 descriptor channels,
    mixed over the flattened 16x16 token sequence with two trainable 1-D
    convolutions, and globally averaged into one 512-D policy token.
    """

    projector_name = "final_conv_v1"
    projector_descriptor = (
        "token_layernorm_768__conv1d_768_512_k3p1_relu__"
        "conv1d_512_512_k3p1_relu__adaptive_avg_pool1d_1"
    )

    def __init__(self, input_size=224, output_dim=512):
        super().__init__()
        if input_size <= 0 or input_size % DINOV2_PATCH_SIZE != 0:
            raise ValueError(
                f"input_size must be positive and divisible by {DINOV2_PATCH_SIZE}"
            )
        if output_dim != 512:
            raise ValueError("final_conv_v1 requires output_dim=512")
        self.input_size = int(input_size)
        self.grid_size = self.input_size // DINOV2_PATCH_SIZE
        self.output_dim = int(output_dim)

        self.token_norm = nn.LayerNorm(DINOV2_EMBED_DIM)
        self.conv1 = nn.Conv1d(
            DINOV2_EMBED_DIM, self.output_dim, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            self.output_dim, self.output_dim, kernel_size=3, padding=1
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, dense_features):
        expected = (
            DINOV2_EMBED_DIM,
            self.grid_size,
            self.grid_size,
        )
        if dense_features.ndim != 4 or dense_features.shape[1:] != expected:
            raise ValueError(
                f"DINOv2 projector expected (N,{expected[0]},{expected[1]},"
                f"{expected[2]}), found {tuple(dense_features.shape)}"
            )

        tokens = dense_features.flatten(2).transpose(1, 2)
        tokens = self.token_norm(tokens).transpose(1, 2)
        tokens = F.relu(self.conv1(tokens))
        tokens = F.relu(self.conv2(tokens))
        return self.pool(tokens).squeeze(-1)


def build_dinov2_projector(projector_type, input_size, output_dim=512):
    """Build a named trainable projector without constructing the backbone."""
    projector_type = str(projector_type)
    if projector_type == DINOv2SpatialProjector.projector_name:
        return DINOv2SpatialProjector(
            input_size=input_size,
            output_dim=output_dim,
        )
    if projector_type == DINOv2FinalConvProjector.projector_name:
        return DINOv2FinalConvProjector(
            input_size=input_size,
            output_dim=output_dim,
        )
    raise ValueError(
        f"unsupported DINOv2 projector {projector_type!r}; "
        f"choose {SUPPORTED_PROJECTORS}"
    )


def describe_dinov2_projector(projector):
    """Return the stable architecture identity stored in policy snapshots."""
    return {
        "type": str(projector.projector_name),
        "descriptor": str(projector.projector_descriptor),
        "parameter_count": int(
            sum(parameter.numel() for parameter in projector.parameters())
        ),
        "state_shapes": {
            key: [int(dimension) for dimension in value.shape]
            for key, value in projector.state_dict().items()
        },
    }


def build_frozen_dinov2_and_projector(
    variant,
    source_path,
    source_revision,
    backbone_weight_path,
    backbone_sha256,
    adapter_weight_path,
    adapter_sha256,
    source_image_size,
    input_size,
    projector_type=DINOv2SpatialProjector.projector_name,
    output_dim=512,
):
    """Construct both modules without perturbing the policy RNG stream."""
    with torch.random.fork_rng(devices=[]):
        frozen_encoder = FrozenDINOv2(
            variant=variant,
            source_path=source_path,
            source_revision=source_revision,
            backbone_weight_path=backbone_weight_path,
            backbone_sha256=backbone_sha256,
            adapter_weight_path=adapter_weight_path,
            adapter_sha256=adapter_sha256,
            source_image_size=source_image_size,
            input_size=input_size,
        )
    projector = build_dinov2_projector(
        projector_type=projector_type,
        input_size=input_size,
        output_dim=output_dim,
    )
    return frozen_encoder, projector
