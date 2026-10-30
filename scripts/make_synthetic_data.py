"""
Write a small synthetic site with the band names of the real 66-band stack (data/synthetic/synthetic_Features.tif and
synthetic_Label.tif) for the smoke test and the unit tests. Values are random fields with a weak class signal; the
numbers produced on them mean nothing.

    python scripts/make_synthetic_data.py [--rows 96 --cols 128 --out data/synthetic]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BANDS_66 = ["NDVI", "MTCI", "SAVI", "CIRE", "NDRE1", "NDRE2", "LAI", "LCC", "CCC",
            "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "QA60",
            "CSTV_NDVI_Diff", "CSTV_MTCI_Diff", "CSTV_SAVI_Diff", "CSTV_CIRE_Diff", "CSTV_NDRE1_Diff", "CSTV_NDRE2_Diff",
            "CSTV_LAI_Diff", "CSTV_LCC_Diff", "CSTV_CCC_Diff",
            "DEM", "Slope_deg", "Aspect_deg", "Curvature", "Temperature", "Humidity", "Moisture",
            "pH", "Sand_pct_0_5cm", "Silt_pct_0_5cm", "Clay_pct_0_5cm", "PrecipJJA_mm_mean", "UpstreamArea_km2", "HAND_m",
            "TWI", "NDVI_Rate", "MTCI_Rate", "SAVI_Rate", "CIRE_Rate", "NDRE1_Rate", "NDRE2_Rate", "LAI_Rate", "LCC_Rate",
            "CCC_Rate", "DW_Water", "DW_Trees", "DW_Grass", "DW_FloodedVeg", "DW_Crops", "DW_Shrubs", "DW_Built",
            "DW_Bare", "DW_Snow", "DW_LandCover", "DW_Vegetation"]


def make(rows=96, cols=128, seed=0):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    rr, cc = np.mgrid[:rows, :cols]
    hazard = gaussian_filter(rng.normal(size=(rows, cols)), 8) + (rr + 0.5 * cc) / (rows + cols)
    y = (hazard > np.quantile(hazard, 0.25)).astype(np.float32)                  # 75 % class 1, as the real labels
    X = np.stack([gaussian_filter(rng.normal(size=(rows, cols)), rng.uniform(1, 6)) for _ in BANDS_66]).astype(np.float32)
    for k in range(0, len(BANDS_66), 5):                                          # weak class signal in some bands
        X[k] += 0.6 * (y - y.mean())
    X[BANDS_66.index("QA60")] = 0.0
    X[:, :2, :2] = np.nan                                                         # a no-data corner
    y[:2, :2] = 255
    return X, y


def write(out_dir, X, y, stem="synthetic"):
    import rasterio
    from rasterio.transform import from_origin
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", height=X.shape[1], width=X.shape[2], crs="EPSG:4326",
                transform=from_origin(-80.886, 37.383, 1e-4, 1e-4), compress="lzw")
    with rasterio.open(out / f"{stem}_Features.tif", "w", count=X.shape[0], dtype="float32", nodata=np.nan, **prof) as d:
        d.write(X)
        for i, n in enumerate(BANDS_66, start=1):
            d.set_band_description(i, n)
    with rasterio.open(out / f"{stem}_Label.tif", "w", count=1, dtype="uint8", nodata=255, **prof) as d:
        d.write(y.astype(np.uint8), 1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--cols", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "synthetic"))
    a = ap.parse_args()
    print("written to", write(a.out, *make(a.rows, a.cols, a.seed)))
