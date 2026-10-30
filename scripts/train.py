"""
Train and evaluate HazardFuse (all seeds), write tables, maps, GeoTIFFs and the exported model bundle.

    python scripts/train.py --config configs/glen_lyn.yaml
    python scripts/train.py --config configs/glen_lyn.yaml --data-dir /content/drive/MyDrive/CCR_Yearly_Feature_Stacks
    python scripts/train.py --config configs/smoke.yaml            # synthetic data, 2 epochs

Outputs (output_dir):
    run_manifest.json              configuration, package versions, devices, git commit, SHA-256 of the inputs
    tables/T0_split.csv            pixels per subset and class-1 fraction
    tables/per_seed_metrics.csv    every metric, every seed, training / validation / test blocks
    tables/T1_test_metrics.csv     test blocks: mean and SD over seeds
    tables/T2_block_bootstrap.csv  main seed: every metric with a 95% block-bootstrap interval
    tables/T3_train_val_test.csv   mean over seeds per split (overfitting check)
    tables/T4_band_attention.csv   mean band-attention weight per band and seed
    geotiffs/, figures/, exported_models/HazardFuse/ (one .keras per seed + manifest.json)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hazardfuse.config import REPO_ROOT, config_hash, load_config, resolve  # noqa: E402
from hazardfuse.data import (band_index, band_stats, find_tif, group_indices, input_bands, load_site,  # noqa: E402
                             standardise, write_geotiff)
from hazardfuse.metrics import METRICS, block_bootstrap  # noqa: E402
from hazardfuse.split import block_split, split_summary  # noqa: E402
from hazardfuse.utils import set_seed, write_run_manifest  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", help="folder with the GeoTIFFs (overrides data.dir)")
    ap.add_argument("--output-dir", help="overrides output_dir")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any configuration key, e.g. --set seeds=[42] --set training.epochs=50")
    return ap.parse_args(argv)


def prepare(cfg):
    """Read the site, select the input bands, split and standardise. Returns a dict used by every script."""
    ddir = resolve(cfg["data"]["dir"])
    feat, lab = find_tif(ddir, cfg["data"]["features"]), find_tif(ddir, cfg["data"]["labels"])
    site = load_site(feat, lab, cfg["data"].get("band_rename"))
    bands = input_bands(site.names, cfg["data"].get("drop_bands"))
    idx = band_index(site.names, bands)
    sp = block_split(site.valid, site.y, **{k: cfg["split"][k] for k in ("block_size", "fractions", "buffer_px",
                                                                          "trials", "seed")})
    mean, std = band_stats(site.X_raw, sp["train"])                 # training-pixel statistics, every band
    X_all = standardise(site.X_raw, site.valid, mean, std)
    return {"site": site, "bands": bands, "idx": idx, "split": sp, "mean": mean, "std": std, "X_all": X_all,
            "X": X_all[..., idx], "groups": group_indices(site.names, cfg["feature_groups"]),
            "masks": {"train": sp["train"], "val": sp["val"], "test": sp["test"], "valid": site.valid}}


def mean_sd_table(runs, split="test"):
    df = pd.DataFrame([r["metrics"][split] for r in runs])
    t = pd.concat({"mean": df.mean(), "sd": df.std(ddof=1) if len(df) > 1 else df.std(ddof=0) * np.nan}, axis=1)
    return t.loc[METRICS + [c for c in df.columns if c not in METRICS]]


def main(argv=None):
    args = parse_args(argv)
    over = list(args.set)
    if args.data_dir:
        over.append(f"data.dir={args.data_dir}")
    if args.output_dir:
        over.append(f"output_dir={args.output_dir}")
    cfg = load_config(args.config, over)
    out = resolve(cfg["output_dir"])
    dirs = {k: out / k for k in ("tables", "figures", "geotiffs", "exported_models", "checkpoints")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    set_seed(cfg["seeds"][0], cfg.get("deterministic", True))

    from hazardfuse import plotting
    from hazardfuse.train import class_weights, run_seeds
    plotting.apply_style(cfg.get("figure"))

    P = prepare(cfg)
    site, sp, masks = P["site"], P["split"], P["masks"]
    write_run_manifest(out, cfg, site.files, {"config_hash": config_hash(cfg)}, REPO_ROOT)
    print(f"{cfg['site']}: grid {site.shape} | bands {len(site.names)} -> model input {len(P['bands'])} "
          f"(dropped {[n for n in site.names if n not in P['bands']]}) | valid {site.valid.sum():,} | "
          f"class-1 fraction {site.y[site.valid].mean():.3f}")
    T0 = split_summary(sp, site.valid, site.y)
    T0.to_csv(dirs["tables"] / "T0_split.csv")
    print(T0.round(3).to_string(), f"\nTest blocks: {len(np.unique(sp['block_id'][sp['test']]))}")
    data_check = {"rows": site.shape[0], "cols": site.shape[1], "bands": len(site.names), "input_bands": len(P["bands"]),
                  "valid_pixels": int(site.valid.sum()), "class1_fraction": round(float(site.y[site.valid].mean()), 3),
                  "train_class1_pixels": int(site.y[sp["train"]].sum()),
                  **{f"{k}_pixels": int(sp[k].sum()) for k in ("train", "buffer", "val", "test")},
                  "test_blocks": int(len(np.unique(sp["block_id"][sp["test"]])))}
    (dirs["tables"] / "data_check.json").write_text(json.dumps(data_check, indent=2))

    # ---- fit every seed
    cw = class_weights(site.y[sp["train"]], cfg.get("class_weights", True))
    print(f"Training pixels {int(sp['train'].sum()):,} | class weights (0, 1) = ({cw[0]:.3f}, {cw[1]:.3f})")
    model_dir = dirs["exported_models"] / "fitted"
    runs = run_seeds(P["X"], site.y, masks, cfg, cfg["seeds"], dirs["checkpoints"], model_dir=model_dir,
                     threshold=cfg["threshold"])

    # ---- tables
    long = pd.DataFrame([{"Seed": r["seed"], "Split": s, "Seconds": r["seconds"], "Epochs": r["epochs"],
                          "Parameters": r["params"], **m} for r in runs for s, m in r["metrics"].items()])
    long.to_csv(dirs["tables"] / "per_seed_metrics.csv", index=False)
    T1 = mean_sd_table(runs, "test")
    T1.to_csv(dirs["tables"] / "T1_test_metrics.csv")
    print(f"\nTest blocks, mean ± SD over seeds {cfg['seeds']}:\n" + T1.loc[METRICS].round(4).to_string())
    T3 = long.groupby("Split")[METRICS].mean().reindex(["train", "val", "test"]).T
    T3.to_csv(dirs["tables"] / "T3_train_val_test.csv")

    main = runs[0]
    bs = block_bootstrap(site.y[sp["test"]], main["prob"][sp["test"]], sp["block_id"][sp["test"]], cfg["threshold"],
                         cfg["evaluation"]["n_boot"], cfg["evaluation"]["n_boot_auc"], cfg["split"]["seed"])
    T2 = pd.DataFrame(bs["ci"], index=["value", "ci_2.5", "ci_97.5"]).T
    T2.index.name = f"Seed {main['seed']}; {bs['n_blocks']} test blocks; {cfg['evaluation']['n_boot']} resamples"
    T2.to_csv(dirs["tables"] / "T2_block_bootstrap.csv")

    # ---- band attention
    if cfg["model"].get("band_attention") and cfg["model"].get("center_branch"):
        import tensorflow as tf
        from hazardfuse.pipeline import band_attention_weights, padded
        Xp = padded(P["X"], cfg["model"]["patch"])
        W = pd.DataFrame({r["seed"]: band_attention_weights(tf.keras.models.load_model(r["model_file"], compile=False),
                                                            Xp, sp["test"], cfg["model"]["patch"]) for r in runs},
                         index=P["bands"])
        W.to_csv(dirs["tables"] / "T4_band_attention.csv")
        bgroup = {site.names[i]: g for g, ix in P["groups"].items() for i in ix}
        if cfg["outputs"].get("figures", True):
            plotting.fig_band_attention(W.mean(axis=1), W.std(axis=1, ddof=1) if W.shape[1] > 1 else None,
                                        {b: bgroup.get(b, "Other") for b in P["bands"]}, dirs["figures"])

    # ---- maps and GeoTIFFs: main seed and seed mean
    ens = np.mean([r["prob"] for r in runs], axis=0)
    maps = {f"HazardFuse_seed{main['seed']}": main["prob"], "HazardFuse_seed_mean": ens}
    if cfg["outputs"].get("figures", True):
        plotting.fig_split(sp, site.valid, dirs["figures"])
        plotting.fig_training_curves(runs, dirs["figures"])
        for name, p in maps.items():
            plotting.fig_maps(p, site.y, site.valid, sp["test"], dirs["figures"], f"Fig2_maps_{name}",
                              f"{name.replace('_', ' ')} - {cfg['site']}", cfg["threshold"])
    if cfg["outputs"].get("geotiffs", True):
        for name, p in maps.items():
            cls, err = plotting.class_and_error_maps(p, site.y, site.valid, threshold=cfg["threshold"])
            write_geotiff(dirs["geotiffs"] / f"{name}_probability.tif", np.where(site.valid, p, np.nan), "float32",
                          np.nan, site.profile)
            write_geotiff(dirs["geotiffs"] / f"{name}_class.tif", cls, "uint8", 255, site.profile)
            write_geotiff(dirs["geotiffs"] / f"{name}_error.tif", err, "uint8", 255, site.profile)
        split_code = np.full(site.shape, 255, np.uint8)
        for i, k in enumerate(("train", "buffer", "val", "test")):
            split_code[sp[k]] = i
        write_geotiff(dirs["geotiffs"] / "split.tif", split_code, "uint8", 255, site.profile)

    # ---- exported bundle and self-check (re-applying the bundle must reproduce the fitted maps)
    if cfg["outputs"].get("export_bundle", True):
        from hazardfuse.transfer import export_bundle, transfer_to_site
        bdir = export_bundle(dirs["exported_models"] / "HazardFuse", runs, cfg, P["bands"], P["mean"][P["idx"]],
                             P["std"][P["idx"]], site.y[sp["train"]].mean(), cw,
                             {k: float(T1.at[k, "mean"]) for k in METRICS})
        chk = transfer_to_site(bdir, site.files["features"], site.files["labels"], standardise_with="source",
                               adabn=False)
        diff = float(np.nanmax(np.abs(chk["prob"][site.valid] - ens[site.valid])))
        print(f"\nBundle {bdir} | self-check max |re-applied − fitted| = {diff:.2e} "
              f"({'OK' if diff < 1e-3 else 'CHECK band order / statistics'})")
    print(f"\nDone: {out}")
    return runs


if __name__ == "__main__":
    main()
