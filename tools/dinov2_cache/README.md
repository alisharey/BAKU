# DINOv2 FP16 feature-cache tools

These source-only utilities build the LIBERO-10 feature caches consumed by
`baku/read_data/dinov2_feature_cache.py`. They preserve the production cache
contract: raw little-endian FP16 shards shaped
`[timestep, camera, 768, 16, 16]`, stock or 3DCorr pre-L2 feature semantics,
and an atomically published format-version-1 manifest.

No model weights, generated features, scheduler scripts, or machine-specific
paths are included. Building both variants requires roughly 202 GiB for the
feature shards, in addition to the original LIBERO data.

## Requirements

- Run from this repository's Python environment with PyTorch, torchvision,
  NumPy, and PyYAML available.
- Convert LIBERO-10 to BAKU's ten task pickle files, with at least 50
  demonstrations per task.
- Check out DINOv2 at revision
  `85a24602099d397264d5b30461ad7f3bfd726ca1`.
- Download the DINOv2-B/14 backbone and released 3DCorr adapter whose SHA-256
  values are pinned in `cache_config.example.yaml`.
- Use a CUDA GPU for extraction and live-feature validation. Index preparation
  is CPU-only.

## Configure and index

Copy the example outside the tracked tool directory and replace each
`/path/to/...` value:

```bash
cp tools/dinov2_cache/cache_config.example.yaml /path/to/cache_config.yaml
python tools/dinov2_cache/prepare_index.py \
  --config /path/to/cache_config.yaml
```

The indexer hashes every source pickle and writes `index.jsonl` plus
`index_manifest.json` atomically. Re-running it validates and reuses an intact
index.

## Bind the build sources

Create a `sha256sum`-compatible manifest after configuring the build. Relative
entries are resolved from the repository root by default; absolute entries are
also supported.

```bash
(
  cd /path/to/baku-repository
  sha256sum \
    tools/dinov2_cache/prepare_index.py \
    tools/dinov2_cache/build_cache.py \
    tools/dinov2_cache/validate_cache.py \
    baku/agent/networks/dinov2_encoder.py \
    baku/read_data/dinov2_feature_cache.py \
    /path/to/cache_config.yaml
) > /path/to/cache_sources.sha256
```

`build_cache.py` verifies every manifest entry before loading data or model
artifacts and embeds the manifest file's SHA-256 in the cache metadata.

## Validate and build

The round-trip phase checks that FP16 storage preserves the live pre-L2
features before committing a full cache:

```bash
python tools/dinov2_cache/validate_cache.py \
  --config /path/to/cache_config.yaml \
  --variant stock --phase roundtrip \
  --output /path/to/stock_roundtrip.json

python tools/dinov2_cache/validate_cache.py \
  --config /path/to/cache_config.yaml \
  --variant 3dcorr --phase roundtrip \
  --output /path/to/3dcorr_roundtrip.json
```

Build the variants independently. A valid completed shard is hash-checked and
skipped on restart.

```bash
python tools/dinov2_cache/build_cache.py \
  --config /path/to/cache_config.yaml --variant stock \
  --source-manifest /path/to/cache_sources.sha256

python tools/dinov2_cache/build_cache.py \
  --config /path/to/cache_config.yaml --variant 3dcorr \
  --source-manifest /path/to/cache_sources.sha256
```

Then verify every shard hash, representative live/cache parity, both cameras,
history windows, and a window crossing a shard boundary:

```bash
python tools/dinov2_cache/validate_cache.py \
  --config /path/to/cache_config.yaml \
  --variant stock --phase cache \
  --output /path/to/stock_cache_validation.json

python tools/dinov2_cache/validate_cache.py \
  --config /path/to/cache_config.yaml \
  --variant 3dcorr --phase cache \
  --output /path/to/3dcorr_cache_validation.json
```

## Use a cache for training

Compute the selected variant manifest hash and pass it explicitly to BAKU:

```bash
sha256sum /path/to/cache/libero_10_dinov2_fp16_v1/stock/manifest.json

cd baku
python train.py agent=baku suite=libero dataloader=libero \
  suite/task=libero_10 encoder_type=dinov2 dinov2_variant=stock \
  dinov2_feature_cache_enabled=true \
  dinov2_feature_cache_root=/path/to/cache/libero_10_dinov2_fp16_v1 \
  dinov2_feature_cache_manifest_sha256=<64-hex-manifest-sha256>
```

Use the `3dcorr` variant and its own manifest hash for the adapted cache. The
reader validates the manifest, index, dataset fingerprint, shard layout, and
requested variant before training.
