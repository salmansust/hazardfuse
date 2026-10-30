"""
HazardFuse ablations: architecture (each variant removes one component) and input feature groups (each variant
removes one group). Every variant is refitted with every seed in ablation.seeds; differences are paired by seed
(variant − full model) on the test blocks.

    python scripts/ablation.py --config configs/glen_lyn.yaml                  # both ablations
    python scripts/ablation.py --config configs/glen_lyn.yaml --only architecture

The full model is refitted here unless outputs/<site>/tables/per_seed_metrics.csv of scripts/train.py holds the same
seeds (then its test metrics are reused).
Outputs: tables/T5_architecture_ablation*.csv, tables/T6_feature_ablation*.csv, figures/Fig5*, Fig6*.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hazardfuse.config import load_config, resolve  # noqa: E402
from hazardfuse.metrics import METRICS  # noqa: E402
from train import prepare  # noqa: E402


def test_metrics(runs):
    return pd.DataFrame([{"Seed": r["seed"], **{k: r["metrics"]["test"][k] for k in METRICS}} for r in runs]).set_index("Seed")


def paired(variant: pd.DataFrame, full: pd.DataFrame):
    d = (variant - full.loc[variant.index]).dropna(how="all")
    return d.mean(), d.std(ddof=1) if len(d) > 1 else d.std(ddof=0) * np.nan


def summarise(results: dict, full: pd.DataFrame, extra: dict):
    rows, deltas, sds = [], {}, {}
    for name, m in results.items():
        mean, sd = m.mean(), m.std(ddof=1) if len(m) > 1 else m.std(ddof=0) * np.nan
        row = {"Variant": name, **extra.get(name, {}), **{f"{k}_mean": mean[k] for k in METRICS},
               **{f"{k}_sd": sd[k] for k in METRICS}}
        if name != "Full":
            dm, ds = paired(m, full)
            deltas[name], sds[name] = dm, ds
            row.update({f"d{k}_mean": dm[k] for k in METRICS})
            row.update({f"d{k}_sd": ds[k] for k in METRICS})
        rows.append(row)
    return pd.DataFrame(rows).set_index("Variant"), pd.DataFrame(deltas).T, pd.DataFrame(sds).T


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir")
    ap.add_argument("--only", choices=["architecture", "features"])
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set + ([f"data.dir={args.data_dir}"] if args.data_dir else []))
    out = resolve(cfg["output_dir"])
    tab, fig_dir, ckpt = out / "tables", out / "figures", out / "checkpoints"
    for d in (tab, fig_dir, ckpt):
        d.mkdir(parents=True, exist_ok=True)

    from hazardfuse import plotting
    from hazardfuse.train import run_seeds
    plotting.apply_style(cfg.get("figure"))
    P = prepare(cfg)
    site, masks = P["site"], P["masks"]
    seeds = cfg["ablation"]["seeds"]

    prev = tab / "per_seed_metrics.csv"
    full = None
    if prev.exists():
        d = pd.read_csv(prev)
        d = d[(d["Split"] == "test") & d["Seed"].isin(seeds)]
        if set(d["Seed"]) == set(seeds):
            full = d.set_index("Seed")[METRICS]
            print(f"Full model: test metrics of {prev} reused (seeds {seeds})")
    if full is None:
        print("Full model")
        full = test_metrics(run_seeds(P["X"], site.y, masks, cfg, seeds, ckpt, label="HazardFuse full",
                                      threshold=cfg["threshold"]))

    if args.only in (None, "architecture"):
        res, extra = {"Full": full}, {}
        for name, switches in cfg["ablation"]["architecture"].items():
            print(f"\n{name}")
            runs = run_seeds(P["X"], site.y, masks, cfg, seeds, ckpt, model_overrides=switches,
                             label=f"HazardFuse {name}", threshold=cfg["threshold"])
            res[name], extra[name] = test_metrics(runs), {"Parameters": runs[0]["params"]}
        T, dmean, dsd = summarise(res, full, extra)
        T.to_csv(tab / "T5_architecture_ablation.csv")
        print(T[["Macro_F1_mean", "dMacro_F1_mean", "dMCC_mean", "dROC_AUC_mean"]].round(4).to_string())
        if cfg["outputs"].get("figures", True):
            plotting.fig_ablation(dmean, dsd, fig_dir, "Fig5_architecture_ablation")

    if args.only in (None, "features"):
        base = [i for i in P["idx"]]                                  # model input of the full model
        groups = {g: [i for i in ix if i in base] for g, ix in P["groups"].items()}
        groups = {g: ix for g, ix in groups.items() if ix}
        res, extra = {"Full": full}, {"Full": {"Bands": len(base)}}
        for g, ix in groups.items():
            keep = [i for i in base if i not in ix]
            name = f"w/o {g} ({len(ix)})"
            print(f"\n{name}: {len(keep)} bands")
            runs = run_seeds(P["X_all"][..., keep], site.y, masks, cfg, seeds, ckpt, label=f"HazardFuse {name}",
                             threshold=cfg["threshold"])
            res[name], extra[name] = test_metrics(runs), {"Bands": len(keep)}
        T, dmean, dsd = summarise(res, full, extra)
        T.to_csv(tab / "T6_feature_ablation.csv")
        print(T[["Bands", "Macro_F1_mean", "dMacro_F1_mean", "dKappa_mean", "dIoU_1_mean"]].round(4).to_string())
        if cfg["outputs"].get("figures", True):
            plotting.fig_ablation(dmean, dsd, fig_dir, "Fig6_feature_ablation")


if __name__ == "__main__":
    main()
