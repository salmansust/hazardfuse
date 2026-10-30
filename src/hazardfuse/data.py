"""Raster input and output: band renaming, band selection by name, valid mask and standardisation."""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def find_tif(folder, stem) -> Path:
    for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
        p = Path(folder) / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No {stem}.tif/.tiff in {folder}")


def canonical_name(name: str, rules: dict | None) -> str:
    """Apply the rename rules (regex -> replacement, case-insensitive) to one band name."""
    for pattern, repl in (rules or {}).items():
        name = re.sub(pattern, repl, name, flags=re.IGNORECASE)
    return name


def read_raster(path, rules: dict | None = None):
    """All bands as float32 (bands, rows, cols) with nodata as NaN, the profile and the (renamed) band names."""
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read(masked=True).astype("float32").filled(np.nan)
        names = [d if d else f"band_{i + 1}" for i, d in enumerate(src.descriptions)]
        return arr, src.profile.copy(), [canonical_name(n, rules) for n in names]


def write_geotiff(path, array, dtype, nodata, profile) -> None:
    import rasterio
    meta = {"driver": "GTiff", "height": array.shape[0], "width": array.shape[1], "count": 1, "dtype": dtype,
            "crs": profile.get("crs"), "transform": profile.get("transform"), "nodata": nodata, "compress": "lzw"}
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(array.astype(dtype), 1)


@dataclass
class Site:
    """One site: raw features (rows, cols, bands), band names, labels (0/1), valid mask and raster profile."""
    X_raw: np.ndarray
    names: list
    profile: dict
    valid: np.ndarray
    y: np.ndarray | None = None
    files: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.valid.shape


def load_site(features_path, labels_path=None, rules: dict | None = None) -> Site:
    """Read a feature stack (and optional labels). Valid pixels: at least one finite band and, with labels, label 0/1."""
    arr, profile, names = read_raster(features_path, rules)
    if len({n.lower() for n in names}) != len(names):
        raise ValueError(f"{Path(features_path).name}: duplicate band names after renaming")
    X = np.transpose(arr, (1, 2, 0))
    valid = np.isfinite(X).any(axis=-1)
    y = None
    if labels_path is not None:
        lab = read_raster(labels_path)[0][0]
        if lab.shape != valid.shape:
            raise ValueError(f"feature grid {valid.shape} and label grid {lab.shape} differ")
        valid &= np.isin(lab, [0, 1])
        y = np.where(valid, lab, 0).astype(np.uint8)
    return Site(X, names, profile, valid, y, {"features": Path(features_path), "labels": labels_path and Path(labels_path)})


def band_index(names, wanted) -> list:
    """Positions of `wanted` in `names` (case-insensitive); raises if a band is missing."""
    pos = {n.lower(): i for i, n in enumerate(names)}
    missing = [b for b in wanted if b.lower() not in pos]
    if missing:
        raise ValueError(f"bands not in the raster: {missing}")
    return [pos[b.lower()] for b in wanted]


def input_bands(names, drop) -> list:
    """Model input: every band except `drop`, in raster order."""
    d = {b.lower() for b in drop or []}
    return [n for n in names if n.lower() not in d]


def group_indices(names, groups: dict) -> dict:
    """{group: positions in `names`} for the bands of each group that are present."""
    pos = {n.lower(): i for i, n in enumerate(names)}
    return {g: [pos[b.lower()] for b in dict.fromkeys(bands) if b.lower() in pos] for g, bands in groups.items()}


def band_stats(X, mask):
    """Mean and SD of every band over the pixels in `mask` (NaN ignored); SD 0 or NaN -> 1, mean NaN -> 0."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean, std = np.nanmean(X[mask], axis=0), np.nanstd(X[mask], axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
    return mean, std


def standardise(X, valid, mean, std) -> np.ndarray:
    """(X - mean) / std; NaN (missing values) -> 0, i.e. the mean; pixels outside `valid` -> 0."""
    Xs = np.nan_to_num((X - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Xs[~valid] = 0.0
    return Xs
