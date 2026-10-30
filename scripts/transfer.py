"""
Apply an exported HazardFuse bundle to another site (labels optional).

    python scripts/transfer.py --bundle outputs/glen_lyn/exported_models/HazardFuse \
        --features data/raw/bremo_Features.tif --labels data/raw/bremo_Label.tif --site Bremo
    python scripts/transfer.py ... --all-options        # with labels: every standardisation x AdaBN option scored

Outputs (--out, default outputs/transfer_<site>): probability, class (and error) GeoTIFFs, a map figure and
<site>_transfer_summary.json; with --all-options, T_transfer_options.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hazardfuse.config import load_config, resolve  # noqa: E402
from hazardfuse.data import write_geotiff  # noqa: E402
from hazardfuse.metrics import METRICS  # noqa: E402
from hazardfuse.utils import slug  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--site", default="new site")
    ap.add_argument("--config", default="configs/glen_lyn.yaml", help="transfer defaults and figure style")
    ap.add_argument("--standardise", choices=["source", "target"])
    ap.add_argument("--adabn", choices=["true", "false"])
    ap.add_argument("--threshold", help="source, prevalence or a number")
    ap.add_argument("--all-options", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    cfg = load_config(resolve(args.config))
    tcfg = cfg.get("transfer", {})
    std = args.standardise or tcfg.get("standardise", "target")
    ada = (args.adabn == "true") if args.adabn else bool(tcfg.get("adabn", True))
    thr = args.threshold or tcfg.get("threshold", "source")
    out = Path(args.out) if args.out else resolve(f"outputs/transfer_{slug(args.site)}")
    out.mkdir(parents=True, exist_ok=True)

    from hazardfuse import plotting
    from hazardfuse.transfer import transfer_to_site
    plotting.apply_style(cfg.get("figure"))
    res = transfer_to_site(resolve(args.bundle), resolve(args.features), resolve(args.labels) if args.labels else None,
                           standardise_with=std, adabn=ada, threshold=thr)
    site, prob = res["site"], res["prob"]
    tag = f"HazardFuse_{slug(args.site)}"
    labels = site.y if site.y is not None else np.zeros(site.shape, np.uint8)
    cls, err = plotting.class_and_error_maps(prob, labels, site.valid, threshold=res["threshold"])
    write_geotiff(out / f"{tag}_probability.tif", np.where(site.valid, prob, np.nan), "float32", np.nan, site.profile)
    write_geotiff(out / f"{tag}_class.tif", cls, "uint8", 255, site.profile)
    if site.y is not None:
        write_geotiff(out / f"{tag}_error.tif", err, "uint8", 255, site.profile)
        plotting.fig_maps(prob, site.y, site.valid, site.valid, out, f"Fig_transfer_{tag}",
                          f"HazardFuse trained at {res['training_site']}, applied to {args.site} "
                          f"({std} statistics{', AdaBN' if ada else ''})", res["threshold"])
    summary = {k: v for k, v in res.items() if k not in ("prob", "site")}
    (out / f"{tag}_transfer_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2, default=float))
    if "metrics" in res:
        print(pd.Series({k: res["metrics"][k] for k in METRICS}).round(4).to_string())

    if args.all_options and site.y is not None:
        rows = {}
        for s in ("source", "target"):
            for a in (False, True):
                r = transfer_to_site(resolve(args.bundle), resolve(args.features), resolve(args.labels),
                                     standardise_with=s, adabn=a, threshold=thr)
                rows[(s, "AdaBN" if a else "none")] = {**{k: r["metrics"][k] for k in METRICS},
                                                       "Class1_share": r["class1_share"], "Threshold": r["threshold"]}
        T = pd.DataFrame(rows).T
        T.index.names = ["Statistics", "Adaptation"]
        T.to_csv(out / "T_transfer_options.csv")
        print(T[["Macro_F1", "MCC", "Kappa", "Balanced_Acc", "ROC_AUC", "Class1_share"]].round(3).to_string())


if __name__ == "__main__":
    main()
