# Data

The rasters are not stored in git (size and licence of the derived products). Place them in `data/raw/`, or pass
another folder with `--data-dir`.

| File | Content |
|---|---|
| `glen_lyn_Features.tif` | 66-band feature stack, EPSG:4326, 10 m, 212 × 288 pixels; band names in the band descriptions |
| `glen_lyn_Label.tif` | binary label: 0 = lower hazard index (HI), 1 = higher HI; any other value = no data |
| `bremo_Features.tif`, `bremo_Label.tif` | optional: transfer site (same band names) |

## Where to get them

- **Published rasters (exact reproduction):** archive the files used in the paper in a public data repository (e.g.
  Zenodo or Figshare) and give the DOI here: `DOI: <to be added>`. Then run
  `python scripts/checksums.py --write` once on these files and commit `data/checksums.sha256`. Users verify their
  copy with `python scripts/checksums.py`.
- **Regenerating the features:** `gee/CCR_features_export.js` (Google Earth Engine Code Editor; set
  `selectedSite = 'Glen Lyn'`) builds the same 66-band layout and exports `glen_lyn_Features` to Google Drive.
  Earth Engine collections (Sentinel-2, Cloud Score+, Dynamic World, MODIS, NLDAS, GLDAS, CHIRPS, SoilGrids, MERIT
  Hydro) are reprocessed and extended over time, so regenerated values can differ from the published rasters.
  Use the archived rasters for exact reproduction.
- **Labels** are derived from kriged hazard-index values at monitoring wells (`EXPORT_LABELS = true` in the same
  script). The well data are an Earth Engine asset of the author and are available on request.

## Band layout (66 bands)

Bands are matched by **name**, so the order in the file does not matter. `CSTV_*_Diff` names are renamed to `CSTV_*`
on reading (`data.band_rename` in the configuration). `QA60` is removed from the model input (`data.drop_bands`),
which leaves **65 input bands**.

| Group | Bands |
|---|---|
| Vegetation indices (9) | NDVI, MTCI, SAVI, CIRE, NDRE1, NDRE2, LAI, LCC, CCC |
| Sentinel-2 reflectance (12) | B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12 |
| Quality (1) | QA60 |
| Change, CSTV (9) | CSTV_NDVI_Diff … CSTV_CCC_Diff |
| Topography (4) | DEM, Slope_deg, Aspect_deg, Curvature |
| Climate and soil moisture (3) | Temperature, Humidity, Moisture |
| Soil (4) | pH, Sand_pct_0_5cm, Silt_pct_0_5cm, Clay_pct_0_5cm |
| Hydrology (4) | PrecipJJA_mm_mean, UpstreamArea_km2, HAND_m, TWI |
| Rate of change (9) | NDVI_Rate … CCC_Rate |
| Land cover, Dynamic World (11) | DW_Water, DW_Trees, DW_Grass, DW_FloodedVeg, DW_Crops, DW_Shrubs, DW_Built, DW_Bare, DW_Snow, DW_LandCover, DW_Vegetation |

## Values of the published data

`scripts/check_reproduction.py` checks these values exactly (`results/reference/glen_lyn_data.json`):

- 61,056 valid pixels, class-1 fraction 0.739.
- Split: 30,828 training, 13,555 buffer, 7,457 validation and 9,216 test pixels in 9 test blocks.
- 23,312 class-1 training pixels.

## Synthetic data

`python scripts/make_synthetic_data.py` writes `data/synthetic/` (96 × 128 pixels, the same 66 band names) for the
smoke test and the unit tests. Its numbers carry no meaning.
