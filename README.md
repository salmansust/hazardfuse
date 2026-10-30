# HazardFuse

**Pixel–patch fusion network for mapping leached-metal groundwater hazard around coal combustion residual (CCR) sites
from multisource satellite features.**

HazardFuse classifies every 10 m pixel as higher or lower hazard index (HI). It combines two branches:

* **Center-pixel band-attention branch:** the spectral and environmental signature of the pixel, with learned band
weights.
* **Patch spatial-embedding branch:** the 21 × 21 pixel neighbourhood.

A learned gate fuses the two branches. At Glen Lyn (Virginia, USA) it reaches test-block Macro-F1 0.941 ± 0.008 and
MCC 0.886 ± 0.015 (mean ± SD over three seeds). This repository holds:

* the model;
* the training and evaluation protocol;
* the ablations;
* transfer to other sites;
* everything needed to reproduce the published run.

```
                 21 x 21 x 65 patch
                        |
        +---------------+----------------+
        |                                |
  center pixel (65)               Conv3x3(32)-BN-ReLU
  band attention                  Conv3x3(64)-BN-ReLU
  \[raw, attended] -> Dense 96     max-pool, SepConv(96), GAP
        |                                |
        +------------ concat ------------+
                        |
       Dense 96 (ReLU)  x  sigmoid gate (Dense 96)
                        |
             dropout -> sigmoid: P(higher HI)
```

## Repository layout

```
configs/            glen\_lyn.yaml (published settings), smoke.yaml (synthetic data, 2 epochs)
src/hazardfuse/     data.py, split.py, model.py, pipeline.py, train.py, metrics.py, transfer.py, plotting.py
scripts/            train.py, ablation.py, transfer.py, check\_reproduction.py, checksums.py, make\_synthetic\_data.py
notebooks/          HazardFuse\_paper.ipynb - published run: HazardFuse + 10 benchmarks, bootstrap, ablations, transfer
gee/                CCR\_features\_export.js - Google Earth Engine script that builds the 66-band feature stacks
results/reference/  published values used by check\_reproduction.py
data/               README.md (files, band layout, where to get them); rasters go in data/raw/
docs/               REPRODUCIBILITY.md
tests/              unit tests and an end-to-end smoke test on synthetic data
```

## Installation

```bash
git clone https://github.com/<user>/hazardfuse.git
cd hazardfuse
python -m venv .venv \&\& source .venv/bin/activate           # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt -e .
# or: conda env create -f environment.yml \&\& conda activate hazardfuse
python -m pytest -q                                         # unit tests + smoke test (about 1 min)
```

Python 3.9–3.12. TensorFlow 2.17.1 (the published run used a CPU; a GPU works and is faster).

## Quick start

```bash
make smoke                      # synthetic data, 2 epochs: checks the whole pipeline
```

## Reproducing the published run (Glen Lyn)

1. Put `glen\_lyn\_Features.tif` and `glen\_lyn\_Label.tif` in `data/raw/` (see [data/README.md](data/README.md)).
2. Run:

```bash
python scripts/checksums.py                                    # optional: inputs = published rasters
python scripts/train.py --config configs/glen\_lyn.yaml         # 3 seeds, about 30 min on a CPU
python scripts/check\_reproduction.py --config configs/glen\_lyn.yaml
python scripts/ablation.py --config configs/glen\_lyn.yaml      # architecture and feature-group ablations
```

`check\_reproduction.py` makes two checks:

* **Exact:** grid, valid pixels, class balance and split.
* **Within tolerance:** the per-seed test metrics against the published values.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

Google Colab:

```python
!git clone https://github.com/<user>/hazardfuse.git
%cd hazardfuse
!pip install -q -e .
from google.colab import drive; drive.mount("/content/drive")
!python scripts/train.py --config configs/glen\_lyn.yaml --data-dir /content/drive/MyDrive/CCR\_Yearly\_Feature\_Stacks
```

Any setting can be changed without editing the file:

```bash
python scripts/train.py --config configs/glen\_lyn.yaml --set seeds=\[42] --set training.epochs=50
```

## Protocol

|Step|Setting|
|-|-|
|Input|66-band stack; QA60 removed, 65 bands; standardised with training-pixel mean and SD; missing values set to the mean|
|Split|32-pixel blocks, 70/15/15 % training/validation/test, matched to the site class balance, 8-pixel buffer; test blocks never used for fitting or selection|
|Training|21 × 21 patches centred on labelled pixels, reflect padding, random rotations and flips; class-weighted binary cross-entropy; Adam (1e-3), batch 256, at most 100 epochs; early stopping (patience 20) and learning-rate halving (patience 6) on validation loss; best weights restored|
|Decision|Threshold 0.5|
|Repeats|Seeds 42, 43, 44, each a full refit; tables give mean ± SD|
|Uncertainty|Block bootstrap over the test blocks (2,000 resamples) for every metric|

## Outputs (`outputs/<site>/`)

|File|Content|
|-|-|
|`run\_manifest.json`|Configuration, package versions, devices, git commit, SHA-256 of the inputs|
|`tables/T0\_split.csv`, `tables/data\_check.json`|Pixels per subset, class-1 fraction|
|`tables/per\_seed\_metrics.csv`|Every metric × seed × split (training, validation, test)|
|`tables/T1\_test\_metrics.csv`|Test blocks, mean and SD over seeds|
|`tables/T2\_block\_bootstrap.csv`|Main seed: every metric with a 95 % block-bootstrap interval|
|`tables/T3\_train\_val\_test.csv`|Mean over seeds per split (overfitting check)|
|`tables/T4\_band\_attention.csv`|Band-attention weight per band and seed|
|`tables/T5\_\*`, `T6\_\*`|Architecture and feature-group ablations (variant − full model, paired by seed)|
|`figures/`|Split, probability/class/error maps, training curves, band attention, ablations (PNG 300 dpi, PDF, SVG)|
|`geotiffs/`|Probability, class and error maps (main seed and seed mean), split map|
|`exported\_models/HazardFuse/`|One `.keras` file per seed and `manifest.json` (bands, statistics, threshold, metrics)|

**Metrics** in every table:

* accuracy and balanced accuracy;
* precision, recall and F1 for class 0 and class 1;
* macro precision, macro recall and macro-F1;
* Cohen's kappa and MCC;
* ROC-AUC;
* IoU for each class and mean IoU;
* PR-AUC for each class.

Figure fonts, sizes and colours are set in the `figure` block of the configuration.

## Transfer to another site

```bash
python scripts/transfer.py --bundle outputs/glen\_lyn/exported\_models/HazardFuse \\
    --features data/raw/bremo\_Features.tif --labels data/raw/bremo\_Label.tif --site Bremo --all-options
```

The site needs the same band names; labels are optional. The prediction is the mean over the seed models.

|Option|Choices|
|-|-|
|Standardisation|Training-site statistics (`source`) or the new site's own statistics (`target`)|
|AdaBN|Re-estimate the BatchNorm statistics on the new site; no labels used|
|Threshold|0.5, the training class-1 share, or a number|

## Citation

See [CITATION.cff](CITATION.cff). Please cite the article once published (reference to be added).

## Licence

MIT (code). The data are covered by the licences of their sources (Copernicus Sentinel-2, Google Dynamic World,
NASA, USGS, ISRIC SoilGrids, MERIT Hydro, CHIRPS).

