#!/usr/bin/env python3
from __future__ import annotations

"""
Fast + fallback-compatible Sentinel-2 L1C tile timeseries builder.

Default profile (fast_auto):
    L1C filter:  tileinfo_metadata[dataCoveragePercentage]
    fallback:    matching L2A tileinfo_metadata[dataCoveragePercentage]
    final fall:  original JP2 valid-pixel preview on coverage_band
    read path:   managed S3 download to local temp file, then local JP2 decode

Compatibility profile (original_compatible):
    filter:      original JP2 valid-pixel preview on coverage_band
    read path:   boto3 get_object(...).read() + rasterio.MemoryFile

The script keeps the fast one-shot local Zarr-v2 write even in original_compatible mode,
so "revert" here means the original filter/read strategy, not the old append-heavy Zarr writer.
"""

import argparse
import atexit
import contextlib
import datetime as dt
import json
import os
import shutil
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlparse, unquote

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.session import AWSSession
from rasterio.transform import Affine
import xarray as xr
try:
    from numcodecs import Blosc
except Exception:  # pragma: no cover
    Blosc = None
from pystac_client import Client


# =========================
# Defaults you can edit directly
# =========================

@dataclass(frozen=True)
class RunConfig:
    # Source search
    stac_api_url: str = "https://earth-search.aws.element84.com/v1"
    l1c_collection: str = "sentinel-2-l1c"
    l2a_collection: str = "sentinel-2-l2a"
    in_bucket: str = "sentinel-s2-l1c"

    # Tile/time window
    tile: str = "31TCJ"
    start_date: str = "2015-06-01"
    end_date: str = "2025-12-31"

    # Output product
    mode: str = "ALL12_FLOAT"   # ALL12_FLOAT | ALL12_UINT16 | RGB_FLOAT | RGB_UINT16
    out_dim: int = 128
    quantification_value: float = 10000.0
    resampling_name: str = "average"  # average | bilinear | nearest

    # Profiles / strategies
    profile: str = "fast_auto"   # fast_auto | original_compatible
    filter_strategy: str = "l1c_tileinfo"  # auto | auto_scl | l1c_tileinfo | l2a_tileinfo | l2a_scl_clear | jp2_mask | none
    read_strategy: str = "download_tempfile"  # download_tempfile | get_object_memoryfile | gdal_vsi
    jp2_filter_read_strategy: str = "download_tempfile"
    on_filter_failure: str = "keep"  # keep | skip | raise

    # Coverage / mask settings
    min_valid_fraction: float = 0.80
    coverage_band: str = "B02"
    mask_signal_threshold: float = 0.001
    mask_preview_dim: int = 128

    # Optional true clear-sky fallback from L2A SCL
    scl_clear_classes: Tuple[int, ...] = (4, 5, 6, 7, 11)

    # IO / concurrency
    aws_profile: Optional[str] = "source-keys"
    aws_region: str = "eu-central-1"
    requester_pays_buckets: Tuple[str, ...] = ("sentinel-s2-l1c", "sentinel-s2-l2a")

    metadata_workers: int = max(16, min(64, (os.cpu_count() or 8) * 4))
    # read_workers: int = max(8, min(32, (os.cpu_count() or 8) * 3))
    read_workers = 12
    upload_workers: int = 16
    boto_max_pool_connections: int = 64
    transfer_max_concurrency: int = 4
    preferred_transfer_client: str = "auto"  # auto | classic
    jp2_decode_threads: int = 1
    http_timeout_seconds: int = 3

    # Output / upload
    output_root: str = "outputs"
    overwrite_local: bool = True
    time_chunk: int = 128
    compressor_cname: str = "zstd"
    compressor_clevel: int = 1
    max_cube_gib: float = 4.0

    s3_bucket: str = "ai4qc2-high-compute-sentinel-zarr"
    s3_prefix_base: str = "zarr-runs"
    upload_to_s3: bool = False
    delete_local_after_upload: bool = False

    # Logging / retry behavior
    retries: int = 1
    retry_delay_seconds: float = 1.0
    progress_every: int = 16

    scene_workers: int = 10
    download_threads_per_scene: int = 2


CFG = RunConfig()


# =========================
# Constants
# =========================

BANDS_ALL12 = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
BANDS_RGB = ["B04", "B03", "B02"]

ASSET_KEY_FOR_BAND: Dict[str, str] = {
    "B01": "coastal",
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B09": "nir09",
    "B11": "swir16",
    "B12": "swir22",
}

RESAMPLING_MAP = {
    "average": Resampling.average,
    "bilinear": Resampling.bilinear,
    "nearest": Resampling.nearest,
}

DEFAULT_RESAMPLING_NAME = "average"

_TLS = threading.local()
_RIO_ENVS: List[rasterio.Env] = []
_RIO_ENVS_LOCK = threading.Lock()
_TRANSFER_CONFIG_CACHE: Dict[Tuple[int, str], TransferConfig] = {}


# =========================
# Small data models
# =========================

@dataclass(frozen=True)
class S2Task:
    system_index: str
    scene_key: str
    datetime_utc: str
    band_hrefs: Dict[str, str]
    l1c_tileinfo_uri: Optional[str]
    eo_cloud_cover: Optional[float]


@dataclass(frozen=True)
class L2AFallback:
    system_index: str
    scene_key: str
    datetime_utc: str
    l2a_tileinfo_uri: Optional[str]
    scl_uri: Optional[str]
    eo_cloud_cover: Optional[float]


@dataclass(frozen=True)
class SceneFilterResult:
    task: S2Task
    keep: bool
    coverage_pct: Optional[float]
    threshold_pct: float
    source: str
    cloudy_pixel_pct: Optional[float]
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ReadResult:
    index: int
    ok: bool
    cube: Optional[np.ndarray]
    error: Optional[str]


# =========================
# Helpers
# =========================

def read_local_band_resampled(local_path: str, out_dim: int, resampling: Resampling) -> np.ndarray:
    with rasterio.open(local_path) as src:
        arr = src.read(1, out_shape=(out_dim, out_dim), resampling=resampling, out_dtype="float32")
    return arr.astype(np.float32)

def parse_tile(tile: str) -> Tuple[int, str, str]:
    tile = tile.strip().upper()
    return int(tile[:2]), tile[2], tile[3:]



def scene_key_from_id(item_id: str) -> str:
    return item_id.rsplit("_", 1)[0]



def bands_for_mode(mode: str) -> List[str]:
    if mode in ("ALL12_FLOAT", "ALL12_UINT16"):
        return BANDS_ALL12
    if mode in ("RGB_FLOAT", "RGB_UINT16"):
        return BANDS_RGB
    raise ValueError(f"Unknown mode: {mode}")



def output_dtype_for_mode(mode: str) -> np.dtype:
    if mode.endswith("_UINT16"):
        return np.dtype("uint16")
    if mode.endswith("_FLOAT"):
        return np.dtype("float32")
    raise ValueError(f"Unknown mode: {mode}")



def iso_to_npdt64ns(iso_z: str) -> np.datetime64:
    return np.datetime64(iso_z.replace("Z", ""), "ns")



def timestamp_utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")



def sizeof_gib(nbytes: int) -> float:
    return float(nbytes) / float(1024 ** 3)



def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)



def parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an s3:// URI: {uri}")
    bucket, _, key = uri[5:].partition("/")
    return bucket, key



def normalize_l1c_band_href(href: str, expected_bucket: str) -> str:
    """
    Keep the original notebook workaround for historical Earth Search v1 L1C asset URLs
    that mistakenly pointed at the sentinel-s2-l2a bucket.
    """
    if href.startswith("s3://sentinel-s2-l2a/"):
        return "s3://" + expected_bucket + "/" + href[len("s3://sentinel-s2-l2a/"):]
    return href



def normalize_http_to_s3_if_possible(href: str, expected_bucket: Optional[str] = None) -> str:
    if href.startswith("s3://"):
        return href
    parsed = urlparse(href)
    if parsed.scheme not in {"http", "https"}:
        return href
    host = parsed.netloc
    path = unquote(parsed.path.lstrip("/"))
    if expected_bucket and host.startswith(f"{expected_bucket}.s3"):
        return f"s3://{expected_bucket}/{path}"
    if host.startswith("s3") and expected_bucket and path.startswith(expected_bucket + "/"):
        return f"s3://{path}"
    return href



def safe_float(v: object) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None



def normalize_key(k: object) -> str:
    return "".join(ch.lower() for ch in str(k) if ch.isalnum())



def recursive_find_value(obj: object, target_keys: Sequence[str]) -> Optional[object]:
    targets = {normalize_key(k) for k in target_keys}

    def _walk(x: object) -> Optional[object]:
        if isinstance(x, dict):
            for k, v in x.items():
                if normalize_key(k) in targets:
                    return v
            for v in x.values():
                hit = _walk(v)
                if hit is not None:
                    return hit
        elif isinstance(x, list):
            for v in x:
                hit = _walk(v)
                if hit is not None:
                    return hit
        return None

    return _walk(obj)



def try_fetch_json(uri: str, cfg: RunConfig) -> dict:
    if uri.startswith("s3://"):
        bucket, key = parse_s3_uri(uri)
        s3 = get_thread_s3_client(cfg)
        extra = {"RequestPayer": "requester"} if bucket in cfg.requester_pays_buckets else {}
        obj = s3.get_object(Bucket=bucket, Key=key, **extra)
        body = obj["Body"]
        try:
            return json.loads(body.read())
        finally:
            body.close()
    if uri.startswith("http://") or uri.startswith("https://"):
        with urllib.request.urlopen(uri, timeout=cfg.http_timeout_seconds) as resp:
            return json.loads(resp.read().decode("utf-8"))
    with open(uri, "rb") as f:
        return json.loads(f.read().decode("utf-8"))



def fetch_tileinfo_metrics(tileinfo_uri: str, cfg: RunConfig) -> Tuple[Optional[float], Optional[float]]:
    info = try_fetch_json(tileinfo_uri, cfg)
    cov = safe_float(recursive_find_value(info, ["dataCoveragePercentage", "data_coverage_percentage", "dataCoverage"]))
    cloudy = safe_float(recursive_find_value(info, ["cloudyPixelPercentage", "cloudy_pixel_percentage"]))
    return cov, cloudy



def log_failure(log_path: str | Path, stage: str, item_id: str, err: object, extra: Optional[dict] = None) -> None:
    rec = {
        "ts_utc": timestamp_utc_now(),
        "stage": stage,
        "item_id": item_id,
        "error": repr(err),
    }
    if extra:
        rec.update(extra)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")



def str2bool(v: str) -> bool:
    x = v.strip().lower()
    if x in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if x in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {v}")

def _process_scene_star(args):
    return process_scene_via_local_stage(*args)

def process_scene_via_local_stage(index, scene, cfg, bands, out_dtype):
    tmp_dir = tempfile.mkdtemp(prefix=f"s2_scene_{index}_")
    try:
        resampling = RESAMPLING_MAP[cfg.resampling_name]
        local_paths = {}

        def _download_one(band):
            href = scene.task.band_hrefs[band]
            local_path = os.path.join(tmp_dir, f"{band}.jp2")
            download_uri_to_file(href, local_path, cfg)
            return band, local_path

        with ThreadPoolExecutor(max_workers=min(cfg.download_threads_per_scene, len(bands))) as ex:
            for band, local_path in ex.map(_download_one, bands):
                local_paths[band] = local_path

        if out_dtype == np.uint16:
            cube = np.empty((cfg.out_dim, cfg.out_dim, len(bands)), dtype=np.uint16)
        else:
            cube = np.empty((cfg.out_dim, cfg.out_dim, len(bands)), dtype=np.float32)

        for band_idx, band in enumerate(bands):
            with rasterio.open(local_paths[band]) as src:
                arr_dn = src.read(
                    1,
                    out_shape=(cfg.out_dim, cfg.out_dim),
                    resampling=resampling,
                    out_dtype="float32",
                )

            if out_dtype == np.uint16:
                cube[:, :, band_idx] = np.rint(np.clip(arr_dn, 0.0, 65535.0)).astype(np.uint16)
            else:
                cube[:, :, band_idx] = arr_dn.astype(np.float32) / np.float32(cfg.quantification_value)

        return ReadResult(index=index, ok=True, cube=cube, error=None)

    except Exception as e:
        return ReadResult(index=index, ok=False, cube=None, error=repr(e))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# =========================
# AWS / rasterio clients
# =========================


def make_boto_session(cfg: RunConfig) -> boto3.Session:
    if cfg.aws_profile:
        return boto3.Session(profile_name=cfg.aws_profile, region_name=cfg.aws_region)
    return boto3.Session(region_name=cfg.aws_region)



def get_thread_s3_client(cfg: RunConfig):
    if not hasattr(_TLS, "s3_client"):
        session = make_boto_session(cfg)
        boto_cfg = BotoConfig(
            region_name=cfg.aws_region,
            max_pool_connections=cfg.boto_max_pool_connections,
            retries={"mode": "adaptive", "max_attempts": 10},
        )
        _TLS.s3_client = session.client("s3", config=boto_cfg)
    return _TLS.s3_client



def get_transfer_config(cfg: RunConfig) -> TransferConfig:
    key = (cfg.transfer_max_concurrency, cfg.preferred_transfer_client)
    if key not in _TRANSFER_CONFIG_CACHE:
        kwargs = dict(
            multipart_threshold=8 * 1024 * 1024,
            multipart_chunksize=8 * 1024 * 1024,
            max_concurrency=cfg.transfer_max_concurrency,
            num_download_attempts=5,
            use_threads=True,
        )
        if cfg.preferred_transfer_client in {"auto", "classic"}:
            kwargs["preferred_transfer_client"] = cfg.preferred_transfer_client
        try:
            _TRANSFER_CONFIG_CACHE[key] = TransferConfig(**kwargs)
        except TypeError:
            kwargs.pop("preferred_transfer_client", None)
            _TRANSFER_CONFIG_CACHE[key] = TransferConfig(**kwargs)
    return _TRANSFER_CONFIG_CACHE[key]



def _cleanup_rio_envs() -> None:
    with _RIO_ENVS_LOCK:
        while _RIO_ENVS:
            env = _RIO_ENVS.pop()
            try:
                env.__exit__(None, None, None)
            except Exception:
                pass


atexit.register(_cleanup_rio_envs)



def get_thread_rio_env(cfg: RunConfig) -> rasterio.Env:
    if not hasattr(_TLS, "rio_env"):
        os.environ.setdefault("OPJ_NUM_THREADS", str(cfg.jp2_decode_threads))
        aws_sess = AWSSession(make_boto_session(cfg), requester_pays=True)
        env = rasterio.Env(
            aws_sess,
            AWS_REGION=cfg.aws_region,
            AWS_REQUEST_PAYER="requester",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_GEOREF_SOURCES="INTERNAL",
            GDAL_NUM_THREADS=str(cfg.jp2_decode_threads),
            GDAL_HTTP_MAX_RETRY="6",
            GDAL_HTTP_RETRY_DELAY="1",
            GDAL_PAM_ENABLED="NO",
            VSI_CACHE="TRUE",
            CPL_VSIL_CURL_CACHE_SIZE=str(64 * 1024 * 1024),
        )
        env.__enter__()
        _TLS.rio_env = env
        with _RIO_ENVS_LOCK:
            _RIO_ENVS.append(env)
    return _TLS.rio_env


# =========================
# STAC discovery
# =========================


def discover_l1c_tasks(cfg: RunConfig, bands: Sequence[str]) -> List[S2Task]:
    utm, lat_band, grid_square = parse_tile(cfg.tile)
    client = Client.open(cfg.stac_api_url)
    search = client.search(
        collections=[cfg.l1c_collection],
        datetime=f"{cfg.start_date}/{cfg.end_date}",
        query={
            "mgrs:utm_zone": {"eq": utm},
            "mgrs:latitude_band": {"eq": lat_band},
            "mgrs:grid_square": {"eq": grid_square},
        },
    )

    items = list(search.items())
    items.sort(key=lambda it: (it.datetime, it.id))

    tasks: List[S2Task] = []
    for it in items:
        if it.datetime is None:
            continue
        ts = it.datetime.astimezone(dt.timezone.utc) if it.datetime.tzinfo else it.datetime.replace(tzinfo=dt.timezone.utc)
        assets = it.assets
        band_hrefs: Dict[str, str] = {}
        for band in bands:
            asset_key = ASSET_KEY_FOR_BAND[band]
            asset = assets.get(asset_key)
            if asset is None or not asset.href:
                raise KeyError(f"Missing asset {asset_key} for {it.id}")
            href = normalize_http_to_s3_if_possible(asset.href, expected_bucket=cfg.in_bucket)
            href = normalize_l1c_band_href(href, cfg.in_bucket)
            band_hrefs[band] = href

        tileinfo_asset = assets.get("tileinfo_metadata")
        tileinfo_uri = tileinfo_asset.href if (tileinfo_asset and tileinfo_asset.href) else None
        cloud_cover = safe_float((it.properties or {}).get("eo:cloud_cover"))

        tasks.append(
            S2Task(
                system_index=it.id,
                scene_key=scene_key_from_id(it.id),
                datetime_utc=ts.isoformat().replace("+00:00", "Z"),
                band_hrefs=band_hrefs,
                l1c_tileinfo_uri=tileinfo_uri,
                eo_cloud_cover=cloud_cover,
            )
        )
    return tasks



def discover_l2a_index(cfg: RunConfig) -> Dict[str, L2AFallback]:
    utm, lat_band, grid_square = parse_tile(cfg.tile)
    client = Client.open(cfg.stac_api_url)
    search = client.search(
        collections=[cfg.l2a_collection],
        datetime=f"{cfg.start_date}/{cfg.end_date}",
        query={
            "mgrs:utm_zone": {"eq": utm},
            "mgrs:latitude_band": {"eq": lat_band},
            "mgrs:grid_square": {"eq": grid_square},
        },
    )

    items = list(search.items())
    out: Dict[str, L2AFallback] = {}
    for it in items:
        if it.datetime is None:
            continue
        ts = it.datetime.astimezone(dt.timezone.utc) if it.datetime.tzinfo else it.datetime.replace(tzinfo=dt.timezone.utc)
        assets = it.assets
        tileinfo_asset = assets.get("tileinfo_metadata")
        scl_asset = assets.get("scl") or assets.get("scl-jp2")
        rec = L2AFallback(
            system_index=it.id,
            scene_key=scene_key_from_id(it.id),
            datetime_utc=ts.isoformat().replace("+00:00", "Z"),
            l2a_tileinfo_uri=(tileinfo_asset.href if (tileinfo_asset and tileinfo_asset.href) else None),
            scl_uri=(scl_asset.href if (scl_asset and scl_asset.href) else None),
            eo_cloud_cover=safe_float((it.properties or {}).get("eo:cloud_cover")),
        )
        prev = out.get(rec.scene_key)
        if prev is None:
            out[rec.scene_key] = rec
        else:
            # Prefer the entry with more useful fallback assets.
            prev_score = int(prev.l2a_tileinfo_uri is not None) + int(prev.scl_uri is not None)
            rec_score = int(rec.l2a_tileinfo_uri is not None) + int(rec.scl_uri is not None)
            if rec_score > prev_score:
                out[rec.scene_key] = rec
    return out


# =========================
# Filtering
# =========================


def needs_l2a_index(cfg: RunConfig) -> bool:
    return cfg.filter_strategy in {"auto", "auto_scl", "l2a_tileinfo", "l2a_scl_clear"}



def read_preview_from_jp2(uri: str, cfg: RunConfig, strategy: str) -> np.ndarray:
    with open_raster_dataset(uri, cfg, strategy=strategy) as src:
        arr = src.read(
            1,
            out_shape=(cfg.mask_preview_dim, cfg.mask_preview_dim),
            resampling=Resampling.average,
            out_dtype="float32",
        )
    return arr.astype(np.float32) / np.float32(cfg.quantification_value)



def fallback_jp2_mask_coverage_percent(task: S2Task, cfg: RunConfig) -> float:
    arr = read_preview_from_jp2(task.band_hrefs[cfg.coverage_band], cfg, strategy=cfg.jp2_filter_read_strategy)
    return float((arr > np.float32(cfg.mask_signal_threshold)).mean() * 100.0)



def l2a_scl_clear_coverage_percent(l2a: L2AFallback, cfg: RunConfig) -> float:
    if not l2a.scl_uri:
        raise FileNotFoundError("L2A SCL asset missing")
    href = normalize_http_to_s3_if_possible(l2a.scl_uri)
    get_thread_rio_env(cfg)
    with rasterio.open(href) as src:
        scl = src.read(
            1,
            out_shape=(cfg.mask_preview_dim, cfg.mask_preview_dim),
            resampling=Resampling.nearest,
            out_dtype="uint8",
        )
    valid = np.isin(scl, np.asarray(cfg.scl_clear_classes, dtype=np.uint8))
    return float(valid.mean() * 100.0)



def resolve_filter_chain(cfg: RunConfig) -> List[str]:
    if cfg.filter_strategy == "none":
        return ["none"]
    if cfg.filter_strategy == "l1c_tileinfo":
        return ["l1c_tileinfo"]
    if cfg.filter_strategy == "l2a_tileinfo":
        return ["l2a_tileinfo"]
    if cfg.filter_strategy == "l2a_scl_clear":
        return ["l2a_scl_clear"]
    if cfg.filter_strategy == "jp2_mask":
        return ["jp2_mask"]
    if cfg.filter_strategy == "auto":
        return ["l1c_tileinfo", "l2a_tileinfo", "jp2_mask"]
    if cfg.filter_strategy == "auto_scl":
        return ["l1c_tileinfo", "l2a_tileinfo", "l2a_scl_clear", "jp2_mask"]
    raise ValueError(f"Unknown filter_strategy: {cfg.filter_strategy}")



def evaluate_scene_filter(task: S2Task, l2a: Optional[L2AFallback], cfg: RunConfig) -> SceneFilterResult:
    threshold_pct = float(cfg.min_valid_fraction * 100.0)
    notes: List[str] = []

    def _return(cov: Optional[float], source: str, cloudy: Optional[float] = None) -> SceneFilterResult:
        keep = True if cov is None else (cov >= threshold_pct)
        return SceneFilterResult(
            task=task,
            keep=keep,
            coverage_pct=cov,
            threshold_pct=threshold_pct,
            source=source,
            cloudy_pixel_pct=cloudy,
            notes=tuple(notes),
        )

    for method in resolve_filter_chain(cfg):
        try:
            if method == "none":
                return _return(100.0, "no_filter")

            if method == "l1c_tileinfo":
                if not task.l1c_tileinfo_uri:
                    raise FileNotFoundError("L1C tileinfo_metadata asset missing")
                cov, cloudy = fetch_tileinfo_metrics(task.l1c_tileinfo_uri, cfg)
                if cov is None:
                    raise KeyError("dataCoveragePercentage missing in L1C tileinfo metadata")
                return _return(cov, "l1c_tileinfo", cloudy)

            if method == "l2a_tileinfo":
                if l2a is None or not l2a.l2a_tileinfo_uri:
                    raise FileNotFoundError("matching L2A tileinfo_metadata asset missing")
                cov, cloudy = fetch_tileinfo_metrics(l2a.l2a_tileinfo_uri, cfg)
                if cov is None:
                    raise KeyError("dataCoveragePercentage missing in L2A tileinfo metadata")
                return _return(cov, "l2a_tileinfo", cloudy)

            if method == "l2a_scl_clear":
                if l2a is None:
                    raise FileNotFoundError("matching L2A item missing")
                cov = l2a_scl_clear_coverage_percent(l2a, cfg)
                return _return(cov, "l2a_scl_clear", None)

            if method == "jp2_mask":
                cov = fallback_jp2_mask_coverage_percent(task, cfg)
                return _return(cov, "jp2_mask", None)

            raise ValueError(f"Unhandled filter method: {method}")

        except Exception as e:
            notes.append(f"{method}:{repr(e)}")
            continue

    if cfg.on_filter_failure == "keep":
        notes.append("all_filter_methods_failed_keep")
        return SceneFilterResult(
            task=task,
            keep=True,
            coverage_pct=None,
            threshold_pct=threshold_pct,
            source="filter_error_keep",
            cloudy_pixel_pct=None,
            notes=tuple(notes),
        )
    if cfg.on_filter_failure == "skip":
        notes.append("all_filter_methods_failed_skip")
        return SceneFilterResult(
            task=task,
            keep=False,
            coverage_pct=None,
            threshold_pct=threshold_pct,
            source="filter_error_skip",
            cloudy_pixel_pct=None,
            notes=tuple(notes),
        )
    raise RuntimeError(f"All filter methods failed for {task.system_index}: {' | '.join(notes)}")



def filter_tasks(tasks: Sequence[S2Task], l2a_index: Dict[str, L2AFallback], cfg: RunConfig) -> Tuple[List[SceneFilterResult], List[SceneFilterResult], dict]:
    t0 = time.time()
    results: Dict[str, SceneFilterResult] = {}
    source_counts: Dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=cfg.metadata_workers) as ex:
        futs = {
            ex.submit(evaluate_scene_filter, task, l2a_index.get(task.scene_key), cfg): task.system_index
            for task in tasks
        }
        for i, fut in enumerate(as_completed(futs), start=1):
            res = fut.result()
            results[res.task.system_index] = res
            source_counts[res.source] = source_counts.get(res.source, 0) + 1
            if i % cfg.progress_every == 0 or i == len(tasks):
                elapsed = max(time.time() - t0, 1e-6)
                rate = i / (elapsed / 60.0)
                print(f"[filter] {i}/{len(tasks)} scenes | {rate:.1f} scenes/min")

    kept: List[SceneFilterResult] = []
    skipped: List[SceneFilterResult] = []
    for task in tasks:
        res = results[task.system_index]
        if res.keep:
            kept.append(res)
        else:
            skipped.append(res)

    stats = {
        "filter_seconds": round(time.time() - t0, 3),
        "n_total": len(tasks),
        "n_kept": len(kept),
        "n_skipped": len(skipped),
        "source_counts": source_counts,
    }
    return kept, skipped, stats


# =========================
# Raster read strategies
# =========================


@contextlib.contextmanager
def open_raster_dataset(uri: str, cfg: RunConfig, strategy: Optional[str] = None) -> Iterator[rasterio.DatasetReader]:
    strategy = strategy or cfg.read_strategy
    strategy = strategy.strip().lower()

    if strategy == "gdal_vsi":
        get_thread_rio_env(cfg)
        with rasterio.open(uri) as src:
            yield src
        return

    if strategy == "get_object_memoryfile":
        if not uri.startswith("s3://"):
            # Fallback to rasterio for non-S3 assets.
            get_thread_rio_env(cfg)
            with rasterio.open(uri) as src:
                yield src
            return
        bucket, key = parse_s3_uri(uri)
        s3 = get_thread_s3_client(cfg)
        extra = {"RequestPayer": "requester"} if bucket in cfg.requester_pays_buckets else {}
        obj = s3.get_object(Bucket=bucket, Key=key, **extra)
        body = obj["Body"]
        try:
            data = body.read()
        finally:
            body.close()
        with MemoryFile(data) as mem:
            with mem.open() as src:
                yield src
        return

    if strategy == "download_tempfile":
        suffix = Path(urlparse(uri).path).suffix or ".bin"
        fd, tmp_path = tempfile.mkstemp(prefix="s2_", suffix=suffix)
        os.close(fd)
        try:
            download_uri_to_file(uri, tmp_path, cfg)
            with rasterio.open(tmp_path) as src:
                yield src
        finally:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
        return

    raise ValueError(f"Unknown read strategy: {strategy}")



def download_uri_to_file(uri: str, local_path: str, cfg: RunConfig) -> None:
    if uri.startswith("s3://"):
        bucket, key = parse_s3_uri(uri)
        s3 = get_thread_s3_client(cfg)
        extra = {"RequestPayer": "requester"} if bucket in cfg.requester_pays_buckets else None
        kwargs = {}
        if extra:
            kwargs["ExtraArgs"] = extra
        s3.download_file(
            Bucket=bucket,
            Key=key,
            Filename=local_path,
            Config=get_transfer_config(cfg),
            **kwargs,
        )
        return

    req = urllib.request.Request(uri)
    with urllib.request.urlopen(req, timeout=cfg.http_timeout_seconds) as resp, open(local_path, "wb") as dst:
        shutil.copyfileobj(resp, dst, length=1024 * 1024)



def read_band_resampled(uri: str, cfg: RunConfig, out_dim: int, resampling: Resampling) -> np.ndarray:
    with open_raster_dataset(uri, cfg) as src:
        arr = src.read(1, out_shape=(out_dim, out_dim), resampling=resampling, out_dtype="float32")
    return arr.astype(np.float32)



def compute_output_grid(task: S2Task, cfg: RunConfig) -> Tuple[np.ndarray, np.ndarray, str, Affine]:
    sample_uri = task.band_hrefs.get("B02") or next(iter(task.band_hrefs.values()))
    with open_raster_dataset(sample_uri, cfg, strategy=cfg.read_strategy) as src:
        crs_wkt = src.crs.to_wkt()
        dst_transform = src.transform * Affine.scale(src.width / cfg.out_dim, src.height / cfg.out_dim)

    cols = np.arange(cfg.out_dim, dtype=np.float64)
    rows = np.arange(cfg.out_dim, dtype=np.float64)
    x_easting, _ = dst_transform * (cols + 0.5, np.zeros_like(cols) + 0.5)
    _, y_northing = dst_transform * (np.zeros_like(rows) + 0.5, rows + 0.5)
    return np.asarray(x_easting), np.asarray(y_northing), crs_wkt, dst_transform



def read_scene_quicklook(index: int, scene: SceneFilterResult, cfg: RunConfig, bands: Sequence[str], out_dtype: np.dtype) -> ReadResult:
    try:
        resampling = RESAMPLING_MAP[cfg.resampling_name]
        if out_dtype == np.uint16:
            cube = np.empty((cfg.out_dim, cfg.out_dim, len(bands)), dtype=np.uint16)
        else:
            cube = np.empty((cfg.out_dim, cfg.out_dim, len(bands)), dtype=np.float32)

        for band_idx, band in enumerate(bands):
            arr_dn = read_band_resampled(scene.task.band_hrefs[band], cfg, cfg.out_dim, resampling)
            if out_dtype == np.uint16:
                cube[:, :, band_idx] = np.rint(np.clip(arr_dn, 0.0, 65535.0)).astype(np.uint16)
            else:
                cube[:, :, band_idx] = arr_dn / np.float32(cfg.quantification_value)

        return ReadResult(index=index, ok=True, cube=cube, error=None)
    except Exception as e:
        return ReadResult(index=index, ok=False, cube=None, error=repr(e))


# =========================
# Zarr output
# =========================


def estimate_cube_bytes(n_time: int, out_dim: int, n_bands: int, dtype: np.dtype) -> int:
    return int(n_time) * int(out_dim) * int(out_dim) * int(n_bands) * int(np.dtype(dtype).itemsize)



def make_cube_storage(n_time: int, cfg: RunConfig, bands: Sequence[str], out_dtype: np.dtype, tmp_root: str) -> Tuple[np.ndarray, Optional[str]]:
    est = estimate_cube_bytes(n_time, cfg.out_dim, len(bands), out_dtype)
    est_gib = sizeof_gib(est)
    shape = (n_time, cfg.out_dim, cfg.out_dim, len(bands))
    if est_gib <= cfg.max_cube_gib:
        print(f"Allocating in-memory cube: {est_gib:.3f} GiB")
        return np.empty(shape, dtype=out_dtype), None

    ensure_dir(tmp_root)
    fd, mm_path = tempfile.mkstemp(prefix="cube_", suffix=".mmap", dir=tmp_root)
    os.close(fd)
    print(f"Allocating memmap cube: {est_gib:.3f} GiB -> {mm_path}")
    return np.memmap(mm_path, mode="w+", dtype=out_dtype, shape=shape), mm_path



def build_dataset(
    cube: np.ndarray,
    kept: Sequence[SceneFilterResult],
    good_indices: Sequence[int],
    cfg: RunConfig,
    bands: Sequence[str],
    x_easting: np.ndarray,
    y_northing: np.ndarray,
) -> xr.Dataset:
    if len(good_indices) != cube.shape[0]:
        cube = cube[np.asarray(good_indices, dtype=np.int64)]
        kept = [kept[i] for i in good_indices]

    pid_len = max(len(x.task.system_index) for x in kept) if kept else 1
    pid_arr = np.asarray([x.task.system_index for x in kept], dtype=f"U{pid_len}")
    time_arr = np.asarray([iso_to_npdt64ns(x.task.datetime_utc) for x in kept], dtype="datetime64[ns]")
    cov_arr = np.asarray([np.nan if x.coverage_pct is None else x.coverage_pct for x in kept], dtype=np.float32)

    ds = xr.Dataset(
        data_vars={
            "reflectance": (("time", "y", "x", "band"), cube),
        },
        coords={
            "time": ("time", time_arr),
            "y": ("y", y_northing.astype(np.float64)),
            "x": ("x", x_easting.astype(np.float64)),
            "band": ("band", np.arange(len(bands), dtype=np.int16)),
            "band_name": ("band", np.asarray(bands, dtype=f"U{max(map(len, bands))}")),
            "system_index": ("time", pid_arr),
            "coverage_pct": ("time", cov_arr),
        },
        attrs={
            "mgrs_tile": cfg.tile,
            "source_collection": cfg.l1c_collection,
            "processing_profile": cfg.profile,
            "filter_strategy": cfg.filter_strategy,
            "read_strategy": cfg.read_strategy,
            "jp2_filter_read_strategy": cfg.jp2_filter_read_strategy,
            "min_valid_fraction": float(cfg.min_valid_fraction),
            "out_dim": int(cfg.out_dim),
            "quantification_value": float(cfg.quantification_value),
            "built_utc": timestamp_utc_now(),
        },
    )

    ds["x"].attrs.update({"standard_name": "projection_x_coordinate", "units": "m"})
    ds["y"].attrs.update({"standard_name": "projection_y_coordinate", "units": "m"})
    ds["coverage_pct"].attrs.update({"units": "percent", "long_name": "scene valid coverage percentage used for filtering"})
    ds["time"].encoding = {
        "units": "nanoseconds since 1970-01-01 00:00:00",
        "calendar": "proleptic_gregorian",
        "dtype": "int64",
    }

    if cube.dtype == np.uint16:
        ds["reflectance"].attrs.update({
            "stored_as": "uint16_scaled_reflectance",
            "scale_factor": 1.0 / float(cfg.quantification_value),
            "add_offset": 0.0,
            "note": "Recover reflectance_float = stored_uint16 * scale_factor + add_offset",
        })
    else:
        ds["reflectance"].attrs.update({"stored_as": "float32_reflectance"})

    time_chunk = min(cfg.time_chunk, max(1, cube.shape[0]))
    ds = ds.chunk({"time": time_chunk, "y": cfg.out_dim, "x": cfg.out_dim, "band": len(bands)})
    return ds



def write_zarr(ds: xr.Dataset, local_store: str, cfg: RunConfig, crs_wkt: str, dst_transform: Affine, filter_stats: dict) -> None:
    encoding = {
        "reflectance": {
            "chunks": (
                min(cfg.time_chunk, max(1, ds.sizes["time"])),
                ds.sizes["y"],
                ds.sizes["x"],
                ds.sizes["band"],
            ),
        }
    }
    if Blosc is not None:
        encoding["reflectance"]["compressor"] = Blosc(
            cname=cfg.compressor_cname,
            clevel=cfg.compressor_clevel,
            shuffle=Blosc.BITSHUFFLE,
        )

    ds = ds.assign_attrs({
        **ds.attrs,
        "crs_wkt": crs_wkt,
        "dst_transform_gdal": tuple(dst_transform.to_gdal()),
        "filter_stats_json": json.dumps(filter_stats, sort_keys=True),
    })

    if os.path.exists(local_store):
        if cfg.overwrite_local:
            shutil.rmtree(local_store)
        else:
            raise FileExistsError(f"Local Zarr store already exists: {local_store}")

    ds.to_zarr(
        local_store,
        mode="w",
        consolidated=True,
        encoding=encoding,
        zarr_version=2,
    )


# =========================
# Upload
# =========================


def iter_files(root: str) -> Iterable[Tuple[str, str]]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            yield full, rel.replace(os.sep, "/")



def upload_zarr_directory(local_store: str, cfg: RunConfig, prefix: str) -> Tuple[int, int]:
    xfer = get_transfer_config(cfg)
    files = list(iter_files(local_store))
    total_bytes = sum(os.path.getsize(full) for full, _ in files)

    def _upload_one(full_rel: Tuple[str, str]) -> None:
        full, rel = full_rel
        key = f"{prefix}/{Path(local_store).name}/{rel}".replace("//", "/")
        s3 = get_thread_s3_client(cfg)
        s3.upload_file(full, cfg.s3_bucket, key, Config=xfer)

    with ThreadPoolExecutor(max_workers=cfg.upload_workers) as ex:
        futs = [ex.submit(_upload_one, item) for item in files]
        for fut in as_completed(futs):
            fut.result()

    return len(files), total_bytes


# =========================
# Main pipeline
# =========================


def output_paths(cfg: RunConfig) -> Tuple[str, str, str]:
    zarr_name = f"s2_l1c_tile_{cfg.tile}_{cfg.start_date}_to_{cfg.end_date}_{cfg.mode}.zarr"
    out_dir = os.path.join(cfg.output_root, cfg.tile)
    local_store = os.path.join(out_dir, zarr_name)
    failure_log = os.path.join(out_dir, "failures.jsonl")
    return out_dir, local_store, failure_log



def apply_profile(cfg: RunConfig) -> RunConfig:
    if cfg.profile == "fast_auto":
        return cfg
    if cfg.profile == "original_compatible":
        return replace(
            cfg,
            filter_strategy="jp2_mask",
            read_strategy="get_object_memoryfile",
            jp2_filter_read_strategy="get_object_memoryfile",
        )
    raise ValueError(f"Unknown profile: {cfg.profile}")



def parse_args(default_cfg: RunConfig) -> RunConfig:
    parser = argparse.ArgumentParser(description="Build fast 128x128 Sentinel-2 L1C tile timeseries Zarr and optionally upload to S3.")
    parser.add_argument("--tile", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--mode", choices=["ALL12_FLOAT", "ALL12_UINT16", "RGB_FLOAT", "RGB_UINT16"], default=None)
    parser.add_argument("--out-dim", type=int, default=None)
    parser.add_argument("--profile", choices=["fast_auto", "original_compatible"], default=None)
    parser.add_argument("--resampling-name", choices=["average", "bilinear", "nearest"], default=None)
    parser.add_argument("--filter-strategy", choices=["auto", "auto_scl", "l1c_tileinfo", "l2a_tileinfo", "l2a_scl_clear", "jp2_mask", "none"], default=None)
    parser.add_argument("--read-strategy", choices=["download_tempfile", "get_object_memoryfile", "gdal_vsi"], default=None)
    parser.add_argument("--jp2-filter-read-strategy", choices=["download_tempfile", "get_object_memoryfile", "gdal_vsi"], default=None)
    parser.add_argument("--min-valid-fraction", type=float, default=None)
    parser.add_argument("--coverage-band", default=None)
    parser.add_argument("--mask-signal-threshold", type=float, default=None)
    parser.add_argument("--metadata-workers", type=int, default=None)
    parser.add_argument("--read-workers", type=int, default=None)
    parser.add_argument("--upload-workers", type=int, default=None)
    parser.add_argument("--boto-max-pool-connections", type=int, default=None)
    parser.add_argument("--transfer-max-concurrency", type=int, default=None)
    parser.add_argument("--upload-to-s3", type=str2bool, default=None)
    parser.add_argument("--delete-local-after-upload", type=str2bool, default=None)
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument("--s3-prefix-base", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--overwrite-local", type=str2bool, default=None)
    parser.add_argument("--on-filter-failure", choices=["keep", "skip", "raise"], default=None)
    args = parser.parse_args()

    cfg = default_cfg
    if args.profile is not None:
        cfg = replace(cfg, profile=args.profile)
    cfg = apply_profile(cfg)

    for name in [
        "tile",
        "start_date",
        "end_date",
        "mode",
        "out_dim",
        "filter_strategy",
        "read_strategy",
        "jp2_filter_read_strategy",
        "min_valid_fraction",
        "coverage_band",
        "mask_signal_threshold",
        "metadata_workers",
        "read_workers",
        "upload_workers",
        "boto_max_pool_connections",
        "transfer_max_concurrency",
        "upload_to_s3",
        "delete_local_after_upload",
        "s3_bucket",
        "s3_prefix_base",
        "output_root",
        "overwrite_local",
        "on_filter_failure",
    ]:
        v = getattr(args, name)
        if v is not None:
            cfg = replace(cfg, **{name: v})
    return cfg



def main() -> None:
    cfg = parse_args(CFG)
    bands = bands_for_mode(cfg.mode)
    out_dtype = output_dtype_for_mode(cfg.mode)
    out_dir, local_store, failure_log = output_paths(cfg)
    ensure_dir(out_dir)
    if os.path.exists(failure_log):
        os.remove(failure_log)

    print(f"Tile={cfg.tile} | mode={cfg.mode} | out_dim={cfg.out_dim} | bands={len(bands)}")
    print(f"Profile={cfg.profile} | filter={cfg.filter_strategy} | read={cfg.read_strategy} | jp2_filter_read={cfg.jp2_filter_read_strategy} | resampling={cfg.resampling_name}")
    print(f"Workers: metadata={cfg.metadata_workers}, read={cfg.read_workers}, upload={cfg.upload_workers}")

    t0 = time.time()
    tasks = discover_l1c_tasks(cfg, bands)
    print(f"Discovered {len(tasks)} L1C STAC items in {time.time() - t0:.1f}s")
    if not tasks:
        raise RuntimeError("No L1C STAC items found for the requested tile/date range.")

    l2a_index: Dict[str, L2AFallback] = {}
    if needs_l2a_index(cfg):
        t1 = time.time()
        l2a_index = discover_l2a_index(cfg)
        print(f"Discovered {len(l2a_index)} matching-scene L2A items in {time.time() - t1:.1f}s")

    kept, skipped, filter_stats = filter_tasks(tasks, l2a_index, cfg)
    print(f"Filtering kept={len(kept)} skipped={len(skipped)} | stats={json.dumps(filter_stats, sort_keys=True)}")
    if not kept:
        raise RuntimeError("No scenes remained after filtering.")

    x_easting, y_northing, crs_wkt, dst_transform = compute_output_grid(kept[0].task, cfg)

    cube_store, cube_mmap_path = make_cube_storage(len(kept), cfg, bands, out_dtype, tmp_root=out_dir)
    good_mask = np.zeros(len(kept), dtype=bool)

    t2 = time.time()
    args_iter = [
        (i, scene, cfg, bands, out_dtype)
        for i, scene in enumerate(kept)
    ]

    with ProcessPoolExecutor(max_workers=cfg.scene_workers) as ex:
        for n, rr in enumerate(
            ex.map(_process_scene_star, args_iter, chunksize=4),
            start=1,
        ):
            if rr.ok and rr.cube is not None:
                cube_store[rr.index] = rr.cube
                good_mask[rr.index] = True
            else:
                scene = kept[rr.index]
                log_failure(failure_log, "read_scene", scene.task.system_index, rr.error or "unknown")

            if n % cfg.progress_every == 0 or n == len(kept):
                elapsed = max(time.time() - t2, 1e-6)
                rate = n / (elapsed / 60.0)
                ok_count = int(good_mask.sum())
                print(f"[read] {n}/{len(kept)} scenes | ok={ok_count} | rate={rate:.1f} scenes/min")

    good_indices = np.flatnonzero(good_mask).tolist()
    print(f"Read complete: ok={len(good_indices)} failed={len(kept) - len(good_indices)}")
    if not good_indices:
        raise RuntimeError("All kept scenes failed during read.")

    ds = build_dataset(cube_store, kept, good_indices, cfg, bands, x_easting, y_northing)
    write_zarr(ds, local_store, cfg, crs_wkt, dst_transform, filter_stats)
    print(f"Wrote local Zarr store: {local_store}")

    if cube_mmap_path:
        try:
            del cube_store
        except Exception:
            pass

    if cfg.upload_to_s3:
        prefix = f"{cfg.s3_prefix_base}/{cfg.tile}/{cfg.start_date}_to_{cfg.end_date}/{cfg.mode}"
        t3 = time.time()
        n_files, n_bytes = upload_zarr_directory(local_store, cfg, prefix)
        print(f"Uploaded {n_files} files ({sizeof_gib(n_bytes):.3f} GiB) to s3://{cfg.s3_bucket}/{prefix}/{Path(local_store).name} in {time.time() - t3:.1f}s")
        if cfg.delete_local_after_upload:
            shutil.rmtree(local_store)
            print(f"Deleted local store: {local_store}")

    if cube_mmap_path:
        with contextlib.suppress(FileNotFoundError):
            os.remove(cube_mmap_path)

    print(f"Total wall time: {(time.time() - t0) / 60.0:.2f} min")


if __name__ == "__main__":
    main()
