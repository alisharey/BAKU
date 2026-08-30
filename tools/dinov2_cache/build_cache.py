#!/usr/bin/env python3
"""Stream one LIBERO-10 DINOv2 variant into atomic FP16 feature shards."""

import argparse
import bisect
import fcntl
import gc
import hashlib
import json
import os
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


EXPECTED_TIMESTEPS = 138090
CAMERA_KEYS = ("pixels", "pixels_egocentric")
FEATURE_SHAPE = (768, 16, 16)
SEMANTICS = {
    "stock": "terminal_x_norm_patchtokens_reshaped_pre_l2",
    "3dcorr": "terminal_x_norm_patchtokens_after_released_refiner_pre_l2",
}
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BAKU_SOURCE_ROOT = REPOSITORY_ROOT / "baku"


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256_manifest(path, source_root):
    """Verify a sha256sum-compatible source manifest before extraction."""
    path = Path(path).expanduser().resolve()
    source_root = Path(source_root).expanduser().resolve()
    checked = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            expected, separator, raw_path = line.partition("  ")
            if not separator:
                expected, separator, raw_path = line.partition(" *")
            if (
                not separator
                or len(expected) != 64
                or any(character not in "0123456789abcdef" for character in expected)
                or not raw_path
            ):
                raise ValueError(
                    f"malformed SHA-256 manifest line {line_number}: {line!r}"
                )
            source_path = Path(raw_path).expanduser()
            if not source_path.is_absolute():
                source_path = source_root / source_path
            source_path = source_path.resolve()
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"source manifest entry is missing: {source_path}"
                )
            actual = sha256_file(source_path)
            if actual != expected:
                raise ValueError(
                    "source manifest SHA-256 mismatch: "
                    f"path={source_path} expected={expected} actual={actual}"
                )
            checked += 1
    if checked == 0:
        raise ValueError(f"source manifest is empty: {path}")
    return checked


def add_canonical_baku_source():
    if not (BAKU_SOURCE_ROOT / "agent/networks/dinov2_encoder.py").is_file():
        raise FileNotFoundError(
            f"canonical BAKU source tree is missing: {BAKU_SOURCE_ROOT}"
        )
    source = str(BAKU_SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)


def canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class EpisodeSource:
    """Sequential source that holds at most one task pickle in host RAM."""

    def __init__(self, index_manifest):
        self.episodes = list(index_manifest["episodes"])
        self.episode_starts = [int(item["timestep_start"]) for item in self.episodes]
        self.files = {
            str(item["task_name"]): Path(item["path"])
            for item in index_manifest["dataset_files"]
        }
        self.loaded_task = None
        self.loaded_observations = None

    def _load_task(self, task_name):
        if self.loaded_task == task_name:
            return
        self.loaded_observations = None
        self.loaded_task = None
        gc.collect()
        path = self.files[task_name]
        print(f"[source] loading one task file: {path.name}", flush=True)
        with path.open("rb") as handle:
            data = pickle.load(handle)
        self.loaded_observations = data["observations"]
        self.loaded_task = task_name

    def iter_batches(self, start, end, batch_timesteps):
        episode_index = bisect.bisect_right(self.episode_starts, int(start)) - 1
        episode_index = max(0, episode_index)
        cursor = int(start)
        while cursor < int(end):
            episode = self.episodes[episode_index]
            episode_start = int(episode["timestep_start"])
            episode_end = episode_start + int(episode["timestep_count"])
            if cursor >= episode_end:
                episode_index += 1
                continue
            task_name = str(episode["task_name"])
            local_start = cursor - episode_start
            available = min(episode_end, int(end)) - cursor
            self._load_task(task_name)
            source_episode = self.loaded_observations[int(episode["episode_index"])]
            consumed = 0
            while consumed < available:
                take = min(int(batch_timesteps), available - consumed)
                source_start = local_start + consumed
                source_end = source_start + take
                views = np.stack(
                    [
                        source_episode[camera][source_start:source_end]
                        for camera in CAMERA_KEYS
                    ],
                    axis=1,
                )
                if views.dtype != np.uint8 or views.shape[2:] != (128, 128, 3):
                    raise ValueError(
                        f"unexpected source image batch shape/dtype: {views.shape} {views.dtype}"
                    )
                # Exact deterministic transform used by read_data/libero.py:
                # center crop 128->126, ToTensor layout conversion and /255.
                views = views[:, :, 1:127, 1:127, :]
                tensor = (
                    torch.from_numpy(np.ascontiguousarray(views))
                    .permute(0, 1, 4, 2, 3)
                    .float()
                    .div_(255.0)
                )
                yield cursor + consumed, tensor
                consumed += take
            cursor += available
            if cursor >= episode_end:
                episode_index += 1


def validate_sidecar(sidecar, expected):
    for key, value in expected.items():
        if sidecar.get(key) != value:
            return False
    path = Path(expected["absolute_path"])
    if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
        return False
    recorded_sha256 = sidecar.get("sha256")
    return bool(recorded_sha256) and sha256_file(path) == recorded_sha256


def build_shard(model, source, variant_root, identity, shard_index, start, count, batch):
    shard_name = f"shard_{shard_index:04d}_{start:06d}_{start + count:06d}.fp16.bin"
    sidecar_name = f"{shard_name}.json"
    final_path = variant_root / "shards" / shard_name
    sidecar_path = variant_root / "shards" / sidecar_name
    expected_bytes = count * len(CAMERA_KEYS) * int(np.prod(FEATURE_SHAPE)) * 2
    expected_sidecar = {
        "format_version": 1,
        "variant": identity["variant"],
        "feature_semantics": SEMANTICS[identity["variant"]],
        "dtype": "float16",
        "feature_shape": list(FEATURE_SHAPE),
        "camera_keys": list(CAMERA_KEYS),
        "timestep_start": start,
        "timestep_count": count,
        "image_count": count * len(CAMERA_KEYS),
        "bytes": expected_bytes,
        "config_sha256": identity["config_sha256"],
        "index_manifest_sha256": identity["index_manifest_sha256"],
        "source_manifest_sha256": identity["source_manifest_sha256"],
        "absolute_path": str(final_path),
    }
    if final_path.is_file() and sidecar_path.is_file():
        with sidecar_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if validate_sidecar(existing, expected_sidecar):
            print(f"[shard] resume-skip valid {shard_name}", flush=True)
            return existing
        raise ValueError(f"existing cache shard has incompatible identity: {final_path}")
    if final_path.is_file() and not sidecar_path.exists():
        # A crash can occur after the fully written, fsynced shard is renamed
        # but before its sidecar is published.  The atomic rename and exact
        # byte count establish completeness; re-hash it and reconstruct only
        # the missing metadata instead of discarding ~768 MiB of valid work.
        if final_path.stat().st_size != expected_bytes:
            raise ValueError(f"orphan cache shard has wrong size: {final_path}")
        recovered = dict(expected_sidecar)
        recovered["path"] = f"shards/{shard_name}"
        recovered["sha256"] = sha256_file(final_path)
        recovered["completed_at"] = datetime.now(timezone.utc).isoformat()
        recovered["recovered_sidecar_after_atomic_rename"] = True
        atomic_json(sidecar_path, recovered)
        print(f"[shard] recovered missing sidecar for {shard_name}", flush=True)
        return recovered
    if sidecar_path.exists() and not final_path.exists():
        raise ValueError(f"cache sidecar exists without its data shard: {sidecar_path}")

    temporary = final_path.with_name(f".{final_path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    written_timesteps = 0
    try:
        with temporary.open("wb") as output:
            for global_start, images in source.iter_batches(start, start + count, batch):
                if global_start != start + written_timesteps:
                    raise RuntimeError("source stream returned non-contiguous cache rows")
                timesteps = images.shape[0]
                flattened = images.reshape(
                    timesteps * len(CAMERA_KEYS), *images.shape[2:]
                ).to(device="cuda", non_blocking=True)
                with torch.inference_mode():
                    features = model.extract_pre_l2(flattened)
                features = features.reshape(
                    timesteps, len(CAMERA_KEYS), *FEATURE_SHAPE
                )
                array = np.ascontiguousarray(features.cpu().numpy().astype("<f2"))
                payload = memoryview(array).cast("B")
                output.write(payload)
                digest.update(payload)
                written_timesteps += timesteps
            output.flush()
            os.fsync(output.fileno())
        if written_timesteps != count:
            raise RuntimeError(
                f"shard wrote {written_timesteps} timesteps, expected {count}"
            )
        if temporary.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"shard byte count {temporary.stat().st_size}, expected {expected_bytes}"
            )
        os.replace(temporary, final_path)
        sidecar = dict(expected_sidecar)
        sidecar["path"] = f"shards/{shard_name}"
        sidecar["sha256"] = digest.hexdigest()
        sidecar["completed_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(sidecar_path, sidecar)
        print(
            f"[shard] complete index={shard_index} rows={start}:{start + count} "
            f"sha256={digest.hexdigest()}",
            flush=True,
        )
        return sidecar
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True, choices=tuple(SEMANTICS))
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument(
        "--source-root",
        default=str(REPOSITORY_ROOT),
        help="base directory for relative paths in --source-manifest",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("cache extraction requires a CUDA GPU")
    config_path = Path(args.config).expanduser().resolve()
    source_manifest_path = Path(args.source_manifest).expanduser().resolve()
    source_manifest_entries = verify_sha256_manifest(
        source_manifest_path, args.source_root
    )
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config_sha256 = sha256_file(config_path)
    source_manifest_sha256 = sha256_file(source_manifest_path)
    output_root = Path(config["output_root"]).expanduser().resolve()
    index_manifest_path = output_root / "index_manifest.json"
    if not index_manifest_path.is_file():
        raise FileNotFoundError("run prepare_index.py before cache extraction")
    index_manifest_sha256 = sha256_file(index_manifest_path)
    with index_manifest_path.open("r", encoding="utf-8") as handle:
        index_manifest = json.load(handle)
    if index_manifest.get("config_sha256") != config_sha256:
        raise ValueError("index/config identity mismatch")
    if int(index_manifest.get("timestep_count", -1)) != EXPECTED_TIMESTEPS:
        raise ValueError("index has the wrong LIBERO-10 timestep count")

    variant = args.variant
    variant_root = output_root / variant
    (variant_root / "shards").mkdir(parents=True, exist_ok=True)
    lock_path = variant_root / ".build.lock"
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another process owns cache variant {variant}")

        manifest_path = variant_root / "manifest.json"
        if manifest_path.is_file():
            print(
                f"[cache] manifest already published for variant={variant}: {manifest_path}"
            )
            return

        add_canonical_baku_source()
        from agent.networks.dinov2_encoder import FrozenDINOv2

        dino = config["dinov2"]
        model = FrozenDINOv2(
            variant=variant,
            source_path=dino["source_path"],
            source_revision=dino["source_revision"],
            backbone_weight_path=dino["backbone_path"],
            backbone_sha256=dino["backbone_sha256"],
            adapter_weight_path=dino["adapter_path"],
            adapter_sha256=dino["adapter_sha256"],
            source_image_size=int(config["source_image_size"]),
            input_size=int(config["input_size"]),
        ).cuda().eval()
        source = EpisodeSource(index_manifest)
        identity = {
            "variant": variant,
            "config_sha256": config_sha256,
            "index_manifest_sha256": index_manifest_sha256,
            "source_manifest_sha256": source_manifest_sha256,
        }

        shard_timesteps = int(config["shard_timesteps"])
        batch_timesteps = int(config["batch_timesteps"])
        sidecars = []
        shard_index = 0
        for start in range(0, EXPECTED_TIMESTEPS, shard_timesteps):
            count = min(shard_timesteps, EXPECTED_TIMESTEPS - start)
            sidecars.append(
                build_shard(
                    model,
                    source,
                    variant_root,
                    identity,
                    shard_index,
                    start,
                    count,
                    batch_timesteps,
                )
            )
            shard_index += 1

        index_path = output_root / index_manifest["index_path"]
        shards = [
            {
                "path": sidecar["path"],
                "sidecar_path": f"{sidecar['path']}.json",
                "timestep_start": sidecar["timestep_start"],
                "timestep_count": sidecar["timestep_count"],
                "image_count": sidecar["image_count"],
                "bytes": sidecar["bytes"],
                "sha256": sidecar["sha256"],
            }
            for sidecar in sidecars
        ]
        manifest = {
            "format_version": 1,
            "suite": "libero_10",
            "variant": variant,
            "dtype": "float16",
            "feature_shape": list(FEATURE_SHAPE),
            "camera_keys": list(CAMERA_KEYS),
            "timestep_count": EXPECTED_TIMESTEPS,
            "image_count": EXPECTED_TIMESTEPS * len(CAMERA_KEYS),
            "feature_semantics": SEMANTICS[variant],
            "normalization_on_load": "cast_fp16_to_fp32_then_channelwise_l2",
            "config_sha256": config_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "index_manifest_sha256": index_manifest_sha256,
            "index": {
                "path": index_manifest["index_path"],
                "bytes": index_path.stat().st_size,
                "sha256": index_manifest["index_sha256"],
                "episodes": index_manifest["episodes"],
            },
            "dataset": {
                "root": index_manifest["dataset_root"],
                "fingerprint_sha256": index_manifest["dataset_fingerprint_sha256"],
                "files": index_manifest["dataset_files"],
                "num_demos_per_task": 50,
            },
            "preprocessing": index_manifest["preprocessing"],
            "dinov2": {
                "source_revision": dino["source_revision"],
                "backbone_sha256": dino["backbone_sha256"],
                "adapter_sha256": dino["adapter_sha256"] if variant == "3dcorr" else "none",
                "source_image_size": int(config["source_image_size"]),
                "input_size": int(config["input_size"]),
            },
            "shard_timesteps": shard_timesteps,
            "shards": shards,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "host": platform.node(),
            },
        }
        atomic_json(manifest_path, manifest)
        print(
            f"[cache] complete variant={variant} manifest={manifest_path} "
            f"sha256={sha256_file(manifest_path)} "
            f"verified_sources={source_manifest_entries}"
        )


if __name__ == "__main__":
    main()
