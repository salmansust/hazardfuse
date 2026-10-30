"""Whole pipeline on synthetic data (1 epoch): training, tables, maps, GeoTIFFs, bundle, self-check, transfer."""
import json

import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("rasterio")


def test_smoke(tmp_path):
    import make_synthetic_data
    import train
    import transfer
    data = make_synthetic_data.write(tmp_path / "data", *make_synthetic_data.make(64, 96))
    out = tmp_path / "out"
    runs = train.main(["--config", "configs/smoke.yaml", "--data-dir", str(data), "--output-dir", str(out),
                       "--set", "training.epochs=1", "--set", "outputs.figures=false"])
    assert len(runs) == 1
    for f in ("run_manifest.json", "tables/per_seed_metrics.csv", "tables/T1_test_metrics.csv",
              "tables/T2_block_bootstrap.csv", "geotiffs/HazardFuse_seed_mean_probability.tif",
              "exported_models/HazardFuse/manifest.json"):
        assert (out / f).exists(), f
    man = json.loads((out / "exported_models/HazardFuse/manifest.json").read_text())
    assert len(man["bands"]) == 65 and "QA60" not in man["bands"]
    transfer.main(["--bundle", str(out / "exported_models/HazardFuse"), "--features",
                   str(data / "synthetic_Features.tif"), "--labels", str(data / "synthetic_Label.tif"),
                   "--site", "Synthetic", "--out", str(tmp_path / "transfer")])
    assert list((tmp_path / "transfer").glob("*_probability.tif"))
