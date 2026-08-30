#!/usr/bin/env python3
"""Build the canonical LIBERO-10 timestep/camera index atomically."""

import argparse
import fcntl
import gc
import hashlib
import json
import os
import pickle
from pathlib import Path

import yaml


EXPECTED_TIMESTEPS = 138090
EXPECTED_IMAGES = 276180


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_config(config):
    expected = {
        "format_version": 1,
        "suite": "libero_10",
        "num_demos_per_task": 50,
        "camera_keys": ["pixels", "pixels_egocentric"],
        "source_image_size": 126,
        "input_size": 224,
        "patch_grid": 16,
        "embed_dim": 768,
        "dtype": "float16",
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"invalid cache config: {mismatches}")
    if len(config.get("tasks", [])) != 10 or len(set(config["tasks"])) != 10:
        raise ValueError("cache config must name exactly ten unique LIBERO-10 tasks")


def existing_is_valid(output_root, config_sha256):
    manifest_path = output_root / "index_manifest.json"
    index_path = output_root / "index.jsonl"
    if not manifest_path.is_file() or not index_path.is_file():
        return False
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("config_sha256") != config_sha256:
        return False
    if manifest.get("timestep_count") != EXPECTED_TIMESTEPS:
        return False
    if manifest.get("image_count") != EXPECTED_IMAGES:
        return False
    if index_path.stat().st_size != int(manifest.get("index_bytes", -1)):
        return False
    if sha256_file(index_path) != manifest.get("index_sha256"):
        return False
    # The index binds observations while cached training still reads actions,
    # proprioception, and language from the original pickles.  Re-hash every
    # source file on reuse so a changed dataset can never be paired silently
    # with stale visual features.
    for item in manifest.get("dataset_files", []):
        path = Path(item["path"])
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            return False
        if sha256_file(path) != item["sha256"]:
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    config_sha256 = sha256_file(config_path)
    dataset_root = Path(config["dataset_root"]).expanduser().resolve()
    output_root = Path(config["output_root"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    lock_path = output_root / ".index.lock"
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if existing_is_valid(output_root, config_sha256):
            print(f"[index] existing canonical index is valid: {output_root}")
            return

        index_path = output_root / "index.jsonl"
        temporary_index = output_root / f".index.jsonl.tmp.{os.getpid()}"
        episodes = []
        dataset_files = []
        timestep_index = 0
        digest = hashlib.sha256()
        try:
            with temporary_index.open("wb") as index_handle:
                for task_order, task_name in enumerate(config["tasks"]):
                    data_path = dataset_root / f"{task_name}.pkl"
                    if not data_path.is_file():
                        raise FileNotFoundError(f"LIBERO task file not found: {data_path}")
                    data_sha256 = sha256_file(data_path)
                    with data_path.open("rb") as handle:
                        data = pickle.load(handle)
                    observations = data.get("observations")
                    actions = data.get("actions")
                    if len(observations) < 50 or len(actions) < 50:
                        raise ValueError(f"task has fewer than 50 demos: {task_name}")

                    task_start = timestep_index
                    episode_lengths = []
                    for episode_index in range(50):
                        observation = observations[episode_index]
                        length = len(observation["pixels"])
                        episode_lengths.append(length)
                        if len(observation["pixels_egocentric"]) != length:
                            raise ValueError(
                                f"camera length mismatch: {task_name} episode {episode_index}"
                            )
                        if len(actions[episode_index]) != length:
                            raise ValueError(
                                f"action length mismatch: {task_name} episode {episode_index}"
                            )
                        episodes.append(
                            {
                                "task_order": task_order,
                                "task_name": task_name,
                                "episode_index": episode_index,
                                "timestep_start": timestep_index,
                                "timestep_count": length,
                            }
                        )
                        for local_timestep in range(length):
                            record = {
                                "timestep_index": timestep_index,
                                "image_indices": [
                                    2 * timestep_index,
                                    2 * timestep_index + 1,
                                ],
                                "task_order": task_order,
                                "task_name": task_name,
                                "episode_index": episode_index,
                                "episode_timestep": local_timestep,
                                "camera_keys": config["camera_keys"],
                            }
                            payload = canonical_bytes(record)
                            index_handle.write(payload)
                            digest.update(payload)
                            timestep_index += 1
                    dataset_files.append(
                        {
                            "task_order": task_order,
                            "task_name": task_name,
                            "path": str(data_path),
                            "bytes": data_path.stat().st_size,
                            "sha256": data_sha256,
                            "episode_count": 50,
                            "episode_lengths": episode_lengths,
                            "timestep_start": task_start,
                            "timestep_count": sum(episode_lengths),
                        }
                    )
                    del data, observations, actions
                    gc.collect()
                    print(
                        f"[index] task={task_order + 1}/10 name={task_name} "
                        f"timesteps={sum(episode_lengths)} cumulative={timestep_index}",
                        flush=True,
                    )
                index_handle.flush()
                os.fsync(index_handle.fileno())

            if timestep_index != EXPECTED_TIMESTEPS:
                raise ValueError(
                    f"unexpected LIBERO-10 timestep count: {timestep_index} "
                    f"expected={EXPECTED_TIMESTEPS}"
                )
            if 2 * timestep_index != EXPECTED_IMAGES:
                raise ValueError("unexpected LIBERO-10 image count")
            os.replace(temporary_index, index_path)

            dataset_fingerprint = hashlib.sha256(
                canonical_bytes(dataset_files)
            ).hexdigest()
            manifest = {
                "format_version": 1,
                "suite": "libero_10",
                "config_path": str(config_path),
                "config_sha256": config_sha256,
                "dataset_root": str(dataset_root),
                "dataset_fingerprint_sha256": dataset_fingerprint,
                "dataset_files": dataset_files,
                "num_demos_per_task": 50,
                "camera_keys": config["camera_keys"],
                "timestep_count": timestep_index,
                "image_count": 2 * timestep_index,
                "index_path": "index.jsonl",
                "index_bytes": index_path.stat().st_size,
                "index_sha256": digest.hexdigest(),
                "episodes": episodes,
                "preprocessing": {
                    "source_storage": "uint8_nhwc_128x128",
                    "dataset_crop": "deterministic_center_crop_128_to_126",
                    "dataset_transform": "ToPILImage_then_ToTensor",
                    "stochastic_augmentation": False,
                    "encoder_resize": "bilinear_antialias_126_to_224",
                    "encoder_normalization": "imagenet_mean_std",
                },
            }
            atomic_json(output_root / "index_manifest.json", manifest)
            print(
                f"[index] complete timesteps={timestep_index} images={2 * timestep_index} "
                f"sha256={digest.hexdigest()}"
            )
        finally:
            if temporary_index.exists():
                temporary_index.unlink()


if __name__ == "__main__":
    main()
