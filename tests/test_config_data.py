import numpy as np
import pytest

from hazardfuse.config import REPO_ROOT, load_config
from hazardfuse.data import band_index, band_stats, canonical_name, group_indices, input_bands, standardise


def test_config_base_and_overrides():
    cfg = load_config(REPO_ROOT / "configs" / "smoke.yaml", ["training.epochs=1", "seeds=[1, 2]"])
    assert cfg["training"]["epochs"] == 1 and cfg["seeds"] == [1, 2]
    assert cfg["model"]["patch"] == 21                              # from the base file
    assert cfg["data"]["features"] == "synthetic_Features"
    full = load_config(REPO_ROOT / "configs" / "glen_lyn.yaml")
    assert full["seeds"] == [42, 43, 44] and full["split"]["block_size"] == 32


def test_band_names():
    rules = {r"^(CSTV_.+?)_Diff$": r"\1"}
    assert canonical_name("CSTV_NDVI_Diff", rules) == "CSTV_NDVI"
    assert canonical_name("cstv_lai_diff", rules) == "cstv_lai"
    names = ["NDVI", "QA60", "DEM"]
    assert input_bands(names, ["qa60"]) == ["NDVI", "DEM"]
    assert band_index(names, ["dem", "NDVI"]) == [2, 0]
    with pytest.raises(ValueError):
        band_index(names, ["TWI"])
    assert group_indices(names, {"Topo": ["DEM", "Slope_deg"]}) == {"Topo": [2]}


def test_standardise():
    rng = np.random.default_rng(0)
    X = rng.normal(5, 2, size=(20, 30, 3)).astype(np.float32)
    X[0, 0, 0] = np.nan
    X[..., 2] = 7.0                                                 # constant band
    valid = np.ones((20, 30), bool); valid[-1] = False
    m, s = band_stats(X, valid)
    assert s[2] == 1.0
    Xs = standardise(X, valid, m, s)
    assert Xs[0, 0, 0] == 0 and (Xs[-1] == 0).all() and np.isfinite(Xs).all()
    assert abs(Xs[valid][:, 1].mean()) < 1e-5
