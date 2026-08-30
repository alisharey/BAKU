"""Read-only, memory-mapped DINOv2 feature cache support.

The cache is indexed by (task name, demonstration index, timestep, camera).
Rows are grouped by timestep and hold both LIBERO camera views.  Feature files
are raw little-endian FP16 arrays; their dimensions and hashes live in the
atomically published variant manifest.
"""

import hashlib
import json
import re
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import numpy as np


CACHE_FORMAT_VERSION = 1
CACHE_DTYPE = "float16"
CAMERA_KEYS = ("pixels", "pixels_egocentric")
FEATURE_SHAPE = (768, 16, 16)
FEATURE_SEMANTICS = {
    "stock": "terminal_x_norm_patchtokens_reshaped_pre_l2",
    "3dcorr": "terminal_x_norm_patchtokens_after_released_refiner_pre_l2",
}


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


class DINOv2FeatureCacheReader:
    """Validated lazy reader for one stock or 3DCorr cache variant."""

    def __init__(
        self,
        cache_root,
        variant,
        expected_manifest_sha256=None,
        max_open_shards=128,
    ):
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.variant = str(variant)
        if self.variant not in FEATURE_SEMANTICS:
            raise ValueError(f"unsupported cache variant: {self.variant!r}")
        self.variant_root = self.cache_root / self.variant
        self.manifest_path = self.variant_root / "manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"feature-cache manifest not found: {self.manifest_path}")
        if not isinstance(expected_manifest_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", expected_manifest_sha256
        ) is None:
            raise ValueError(
                "cached training requires an explicit lowercase 64-hex "
                "dinov2_feature_cache_manifest_sha256"
            )
        self.manifest_sha256 = sha256_file(self.manifest_path)
        if self.manifest_sha256 != expected_manifest_sha256:
            raise ValueError(
                "feature-cache manifest SHA-256 mismatch: "
                f"expected={expected_manifest_sha256} actual={self.manifest_sha256}"
            )
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self._validate_manifest()

        self.camera_keys = tuple(self.manifest["camera_keys"])
        self.camera_to_index = {name: index for index, name in enumerate(self.camera_keys)}
        self.feature_shape = tuple(int(value) for value in self.manifest["feature_shape"])
        self.shards = tuple(self.manifest["shards"])
        self.shard_starts = tuple(int(shard["timestep_start"]) for shard in self.shards)
        self.max_open_shards = int(max_open_shards)
        self._open_shards = OrderedDict()
        self.episodes = {
            (str(item["task_name"]), int(item["episode_index"])): item
            for item in self.manifest["index"]["episodes"]
        }

    @property
    def dataset_root(self):
        return Path(self.manifest["dataset"]["root"]).expanduser().resolve()

    def _validate_manifest(self):
        manifest = self.manifest
        expected = {
            "format_version": CACHE_FORMAT_VERSION,
            "suite": "libero_10",
            "variant": self.variant,
            "dtype": CACHE_DTYPE,
            "camera_keys": list(CAMERA_KEYS),
            "feature_shape": list(FEATURE_SHAPE),
            "feature_semantics": FEATURE_SEMANTICS[self.variant],
            "normalization_on_load": "cast_fp16_to_fp32_then_channelwise_l2",
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"invalid feature-cache manifest fields: {mismatches}")
        if int(manifest.get("timestep_count", -1)) != 138090:
            raise ValueError("LIBERO-10 cache must contain exactly 138090 timesteps")
        if int(manifest.get("image_count", -1)) != 276180:
            raise ValueError("LIBERO-10 cache must contain exactly 276180 camera images")
        index = manifest.get("index", {})
        index_path = self.cache_root / str(index.get("path", ""))
        if not index_path.is_file():
            raise FileNotFoundError(f"cache index not found: {index_path}")
        if index_path.stat().st_size != int(index.get("bytes", -1)):
            raise ValueError(f"cache index byte-size mismatch: {index_path}")
        if sha256_file(index_path) != str(index.get("sha256")):
            raise ValueError(f"cache index SHA-256 mismatch: {index_path}")

        expected_start = 0
        for shard in manifest.get("shards", []):
            start = int(shard["timestep_start"])
            count = int(shard["timestep_count"])
            if start != expected_start or count <= 0:
                raise ValueError(
                    f"non-contiguous cache shard: start={start} expected={expected_start}"
                )
            path = self.variant_root / str(shard["path"])
            if not path.is_file() or path.stat().st_size != int(shard["bytes"]):
                raise ValueError(f"cache shard missing or wrong size: {path}")
            expected_start += count
        if expected_start != int(manifest["timestep_count"]):
            raise ValueError(
                f"cache shards cover {expected_start} timesteps, expected "
                f"{manifest['timestep_count']}"
            )

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_open_shards"] = OrderedDict()
        return state

    def close(self):
        self._open_shards.clear()

    def verify_dataset_files(self):
        """Fully verify action/proprio/language source pickles against cache."""
        verified = []
        for item in self.manifest["dataset"]["files"]:
            path = Path(item["path"])
            if not path.is_file() or path.stat().st_size != int(item["bytes"]):
                raise ValueError(f"cached-training dataset file missing/wrong size: {path}")
            actual = sha256_file(path)
            if actual != str(item["sha256"]):
                raise ValueError(
                    "cached-training dataset SHA-256 mismatch: "
                    f"path={path} expected={item['sha256']} actual={actual}"
                )
            verified.append(str(path))
        return verified

    def verify_shard_hashes(self):
        """Read and verify every published shard against its manifest SHA-256."""
        verified_bytes = 0
        for shard in self.shards:
            path = self.variant_root / str(shard["path"])
            actual = sha256_file(path)
            if actual != str(shard["sha256"]):
                raise ValueError(
                    "feature-cache shard SHA-256 mismatch: "
                    f"path={path} expected={shard['sha256']} actual={actual}"
                )
            verified_bytes += int(shard["bytes"])
        return {"shards": len(self.shards), "bytes": verified_bytes}

    def _shard_for_timestep(self, timestep_index):
        timestep_index = int(timestep_index)
        if timestep_index < 0 or timestep_index >= int(self.manifest["timestep_count"]):
            raise IndexError(f"cache timestep out of range: {timestep_index}")
        shard_index = bisect_right(self.shard_starts, timestep_index) - 1
        shard = self.shards[shard_index]
        local_index = timestep_index - int(shard["timestep_start"])
        if local_index >= int(shard["timestep_count"]):
            raise RuntimeError("cache shard lookup produced an invalid local index")
        return shard_index, local_index

    def _open_shard(self, shard_index):
        existing = self._open_shards.pop(shard_index, None)
        if existing is not None:
            self._open_shards[shard_index] = existing
            return existing
        shard = self.shards[shard_index]
        shape = (
            int(shard["timestep_count"]),
            len(self.camera_keys),
            *self.feature_shape,
        )
        path = self.variant_root / str(shard["path"])
        mapped = np.memmap(path, dtype="<f2", mode="r", shape=shape, order="C")
        self._open_shards[shard_index] = mapped
        while len(self._open_shards) > self.max_open_shards:
            self._open_shards.popitem(last=False)
        return mapped

    def episode_timestep_start(self, task_name, episode_index):
        key = (str(task_name), int(episode_index))
        try:
            return int(self.episodes[key]["timestep_start"])
        except KeyError as error:
            raise KeyError(f"episode absent from feature cache: {key}") from error

    def episode_length(self, task_name, episode_index):
        key = (str(task_name), int(episode_index))
        try:
            return int(self.episodes[key]["timestep_count"])
        except KeyError as error:
            raise KeyError(f"episode absent from feature cache: {key}") from error

    def get_window(self, task_name, episode_index, start, length, camera_key):
        """Return a copied FP16 window shaped [T, 768, 16, 16]."""
        start = int(start)
        length = int(length)
        if length <= 0:
            raise ValueError("cache window length must be positive")
        episode_length = self.episode_length(task_name, episode_index)
        if start < 0 or start + length > episode_length:
            raise IndexError(
                f"cache window [{start},{start + length}) exceeds episode length "
                f"{episode_length}"
            )
        try:
            camera_index = self.camera_to_index[str(camera_key)]
        except KeyError as error:
            raise KeyError(f"unknown cache camera key: {camera_key!r}") from error

        global_start = self.episode_timestep_start(task_name, episode_index) + start
        output = np.empty((length, *self.feature_shape), dtype=np.float16)
        written = 0
        while written < length:
            shard_index, local_index = self._shard_for_timestep(global_start + written)
            shard = self.shards[shard_index]
            available = int(shard["timestep_count"]) - local_index
            take = min(available, length - written)
            mapped = self._open_shard(shard_index)
            output[written : written + take] = mapped[
                local_index : local_index + take, camera_index
            ]
            written += take
        return output
