# Reproducibility

## What is fixed

| Element | How |
|---|---|
| Data | SHA-256 of every input raster is written to `run_manifest.json`. `scripts/checksums.py` verifies `data/raw` against `data/checksums.sha256`. |
| Split | Deterministic block split (`split.seed = 42`, 500 trials). Pixel counts per subset are checked exactly by `scripts/check_reproduction.py`. |
| Standardisation | Mean and SD of every band over the training pixels only. Missing values are set to the training mean. The values are stored in the bundle manifest. |
| Seeds | Each seed in `seeds` (42, 43, 44) is a full refit. `set_seed` seeds Python, NumPy and TensorFlow before the model is built. |
| Determinism | `deterministic: true` calls `tf.config.experimental.enable_op_determinism()`. The same seed, data, package versions and hardware then give the same weights. |
| Software | `requirements.txt` pins the versions. `run_manifest.json` records the versions in use, the devices, the CUDA/cuDNN build and the git commit. |
| Configuration | The full configuration and its hash are written to `run_manifest.json`. |
| Model | Every seed's model is saved (`.keras`) with `manifest.json`: bands, statistics, patch size, threshold and test metrics. |
| Bundle check | After training, the bundle is re-applied to the training site; the result must equal the fitted maps (max \|difference\| < 1e-3). |

If TensorFlow reports that no deterministic GPU implementation exists for an op, there are two options:

- set `deterministic: false` (or pass `--set deterministic=false`);
- run on a CPU.

The run is then reproducible statistically rather than bit for bit.

## Exact versus statistical reproduction

**Same hardware, same versions, `deterministic: true`:** a rerun gives identical weights and identical tables.

**The published run** used TensorFlow 2.17.1 on a CPU, without op determinism. Across machines, the following can
change the trained weights slightly:

- CPU instruction sets (oneDNN kernels);
- GPU kernels;
- thread scheduling in `tf.data`.

The split, the standardisation and every table built from given probabilities do not change. The test metrics of each
seed should agree with `results/reference/glen_lyn_hazardfuse_test_per_seed.csv` within the seed-to-seed variation.
The published seed SD is 0.008 for Macro-F1 and 0.015 for MCC. `scripts/check_reproduction.py` uses a tolerance of
0.03 per metric and seed.

To reproduce the published maps exactly, use the exported bundle of the published run. Archive it with the rasters,
then apply it with `scripts/transfer.py --standardise source --adabn false` on the training site.

## Steps

```bash
pip install -r requirements.txt -e .
python scripts/checksums.py                           # input rasters = published rasters
python scripts/train.py --config configs/glen_lyn.yaml
python scripts/check_reproduction.py --config configs/glen_lyn.yaml
python scripts/ablation.py --config configs/glen_lyn.yaml          # optional, about 6 h on a CPU (39 fits)
```

Run time of the published run (CPU): about 10 min per seed for HazardFuse (22–27 epochs, early stopping), so about
30 min for three seeds. A GPU reduces this to a few minutes.

## Full benchmark

`notebooks/HazardFuse_paper.ipynb` is the notebook of the published run. It covers HazardFuse and ten benchmarks:

- Random Forest and XGBoost;
- Patch CNN;
- U-Net, U-Net++, Attention U-Net, ResU-Net and SE-ResU-Net;
- DeepLabv3+ and SegFormer.

The notebook also contains the paired block bootstrap, both ablations and the transfer to Bremo. Its outputs are
cleared. Set `DATA_DIR` in section 1 and run all cells. It uses the same split, standardisation and HazardFuse
definition as this package.
