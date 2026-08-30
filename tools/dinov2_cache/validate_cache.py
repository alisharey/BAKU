#!/usr/bin/env python3
"""Validate FP16 fidelity, exact sample indexing, cameras, and history windows."""

import argparse
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision import transforms


CAMERAS = ("pixels", "pixels_egocentric")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BAKU_SOURCE_ROOT = REPOSITORY_ROOT / "baku"


def add_canonical_baku_source():
    if not (BAKU_SOURCE_ROOT / "agent/networks/dinov2_encoder.py").is_file():
        raise FileNotFoundError(
            f"canonical BAKU source tree is missing: {BAKU_SOURCE_ROOT}"
        )
    source = str(BAKU_SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        payload = (
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def source_images(dataset_root, task_name, episode_index, start, length):
    path = Path(dataset_root) / f"{task_name}.pkl"
    with path.open("rb") as handle:
        data = pickle.load(handle)
    episode = data["observations"][int(episode_index)]
    views = np.stack(
        [episode[camera][int(start) : int(start) + int(length)] for camera in CAMERAS],
        axis=1,
    )
    cropped = views[:, :, 1:127, 1:127, :]
    tensor = (
        torch.from_numpy(np.ascontiguousarray(cropped))
        .permute(0, 1, 4, 2, 3)
        .float()
        .div_(255.0)
    )
    return cropped, tensor


def check_dataset_transform_exact(cropped):
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    for timestep in range(cropped.shape[0]):
        for camera in range(cropped.shape[1]):
            reference = transform(cropped[timestep, camera])
            vectorized = (
                torch.from_numpy(np.ascontiguousarray(cropped[timestep, camera]))
                .permute(2, 0, 1)
                .float()
                .div_(255.0)
            )
            if not torch.equal(reference, vectorized):
                maximum = float((reference - vectorized).abs().max())
                raise ValueError(
                    f"vectorized image transform differs from training path: max={maximum}"
                )


def build_model(config, variant):
    add_canonical_baku_source()
    from agent.networks.dinov2_encoder import FrozenDINOv2

    dino = config["dinov2"]
    return FrozenDINOv2(
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


def fidelity_metrics(live):
    cached_roundtrip = live.half().float()
    flattened_live = live.flatten(1)
    flattened_roundtrip = cached_roundtrip.flatten(1)
    cosine = F.cosine_similarity(flattened_live, flattened_roundtrip, dim=1)
    normalized_live = F.normalize(live, p=2, dim=1)
    normalized_roundtrip = F.normalize(cached_roundtrip, p=2, dim=1)
    normalized_cosine = F.cosine_similarity(
        normalized_live.flatten(1), normalized_roundtrip.flatten(1), dim=1
    )
    error = (live - cached_roundtrip).abs()
    return {
        "sample_count": int(live.shape[0]),
        "pre_l2_cosine_min": float(cosine.min()),
        "pre_l2_cosine_mean": float(cosine.mean()),
        "post_l2_cosine_min": float(normalized_cosine.min()),
        "post_l2_cosine_mean": float(normalized_cosine.mean()),
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "all_finite": bool(torch.isfinite(cached_roundtrip).all()),
    }


def extract(model, tensor):
    timesteps, cameras = tensor.shape[:2]
    with torch.inference_mode():
        live = model.extract_pre_l2(
            tensor.reshape(timesteps * cameras, *tensor.shape[2:]).cuda()
        )
    return live.reshape(timesteps, cameras, *live.shape[1:]).cpu()


def representative_specs(index_manifest):
    episodes = index_manifest["episodes"]
    chosen = [episodes[0], episodes[len(episodes) // 2], episodes[-1]]
    specs = []
    for episode in chosen:
        length = int(episode["timestep_count"])
        starts = sorted(set((0, max(0, length // 2 - 1), max(0, length - 2))))
        for start in starts:
            specs.append((episode, start, min(2, length - start)))
    return specs


def find_shard_boundary_window(index_manifest, shard_timesteps):
    for boundary in range(int(shard_timesteps), int(index_manifest["timestep_count"]), int(shard_timesteps)):
        for episode in index_manifest["episodes"]:
            start = int(episode["timestep_start"])
            end = start + int(episode["timestep_count"])
            if start + 2 <= boundary and boundary + 2 <= end:
                return episode, boundary - start - 2, 4, boundary
    raise RuntimeError("no cache shard boundary falls inside a four-frame episode window")


def validate_index_records(index_manifest, index_path, selected_global_indices):
    wanted = set(int(value) for value in selected_global_indices)
    found = {}
    with Path(index_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            index = int(record["timestep_index"])
            if index in wanted:
                found[index] = record
                if len(found) == len(wanted):
                    break
    if set(found) != wanted:
        raise ValueError(f"cache index is missing requested rows: {wanted - set(found)}")
    for index, record in found.items():
        if record["image_indices"] != [2 * index, 2 * index + 1]:
            raise ValueError(f"camera image index mismatch at timestep {index}")
        if record["camera_keys"] != list(CAMERAS):
            raise ValueError(f"camera ordering mismatch at timestep {index}")
    return len(found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True, choices=("stock", "3dcorr"))
    parser.add_argument("--phase", required=True, choices=("roundtrip", "cache"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("validation requires a CUDA GPU")

    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output_root = Path(config["output_root"]).expanduser().resolve()
    index_manifest_path = output_root / "index_manifest.json"
    with index_manifest_path.open("r", encoding="utf-8") as handle:
        index_manifest = json.load(handle)
    model = build_model(config, args.variant)

    all_live = []
    transform_checks = 0
    for episode, start, length in representative_specs(index_manifest):
        cropped, tensor = source_images(
            config["dataset_root"],
            episode["task_name"],
            episode["episode_index"],
            start,
            length,
        )
        check_dataset_transform_exact(cropped)
        transform_checks += int(length) * len(CAMERAS)
        all_live.append(extract(model, tensor).flatten(0, 1))
    live = torch.cat(all_live, dim=0)
    metrics = fidelity_metrics(live)
    if not metrics["all_finite"] or metrics["post_l2_cosine_min"] < 0.99999:
        raise ValueError(f"FP16 fidelity threshold failed: {metrics}")

    result = {
        "format_version": 1,
        "phase": args.phase,
        "variant": args.variant,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(config_path),
        "index_manifest_sha256": sha256_file(index_manifest_path),
        "dataset_transform_exact_samples": transform_checks,
        "fidelity": metrics,
        "cache_lookup": None,
    }

    if args.phase == "cache":
        add_canonical_baku_source()
        from read_data.dinov2_feature_cache import DINOv2FeatureCacheReader

        cache_manifest_path = output_root / args.variant / "manifest.json"
        reader = DINOv2FeatureCacheReader(
            output_root,
            args.variant,
            expected_manifest_sha256=sha256_file(cache_manifest_path),
        )
        full_shard_verification = reader.verify_shard_hashes()
        windows = []
        first = index_manifest["episodes"][0]
        windows.append((first, 1, 4, None))
        windows.append(
            find_shard_boundary_window(index_manifest, config["shard_timesteps"])
        )
        checked_rows = []
        selected_indices = []
        for episode, start, length, boundary in windows:
            _, tensor = source_images(
                config["dataset_root"],
                episode["task_name"],
                episode["episode_index"],
                start,
                length,
            )
            live_window_fp32 = extract(model, tensor)
            live_window_fp16 = live_window_fp32.half().numpy()
            global_start = int(episode["timestep_start"]) + int(start)
            selected_indices.extend(range(global_start, global_start + length))
            for camera_index, camera in enumerate(CAMERAS):
                cached = reader.get_window(
                    episode["task_name"],
                    episode["episode_index"],
                    start,
                    length,
                    camera,
                )
                cached_fp32 = torch.from_numpy(cached.astype(np.float32))
                live_fp32 = live_window_fp32[:, camera_index]
                maximum = float((cached_fp32 - live_fp32).abs().max())
                cosine = F.cosine_similarity(
                    cached_fp32.flatten(1), live_fp32.flatten(1), dim=1
                )
                minimum_cosine = float(cosine.min())
                exact_match = bool(
                    np.array_equal(cached, live_window_fp16[:, camera_index])
                )
                if minimum_cosine < 0.99999 or maximum > 0.02:
                    raise ValueError(
                        "cache/live FP16 fidelity mismatch: "
                        f"camera={camera} min_cosine={minimum_cosine} "
                        f"max_abs_error={maximum}"
                    )
                checked_rows.append(
                    {
                        "task_name": episode["task_name"],
                        "episode_index": int(episode["episode_index"]),
                        "start": int(start),
                        "length": int(length),
                        "camera": camera,
                        "crosses_shard_boundary": boundary is not None,
                        "exact_fp16_match": exact_match,
                        "minimum_cosine": minimum_cosine,
                        "max_abs_error": maximum,
                    }
                )
        indexed = validate_index_records(
            index_manifest,
            output_root / index_manifest["index_path"],
            selected_indices,
        )
        result["cache_lookup"] = {
            "manifest_sha256": reader.manifest_sha256,
            "windows": checked_rows,
            "index_records_checked": indexed,
            "both_camera_keys_checked": True,
            "history_windows_checked": True,
            "shard_boundary_crossing_checked": True,
            "full_shard_sha256_verification": full_shard_verification,
        }

    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
