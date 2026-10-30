"""
Compare a run with the published reference values.

    python scripts/check_reproduction.py --config configs/glen_lyn.yaml

1. Data and split (exact): grid size, band count, valid pixels, class-1 fraction, pixels per subset and number of
   test blocks must equal results/reference/glen_lyn_data.json. A difference means different input rasters (check the
   SHA-256 in run_manifest.json against data/README.md) or a changed split setting.
2. Test metrics per seed (tolerance): Macro-F1, MCC, kappa and ROC-AUC of every seed are compared with
   results/reference/glen_lyn_hazardfuse_test_per_seed.csv. The published run used TensorFlow 2.17.1 on a CPU without
   op determinism, so exact equality is not expected on other hardware; the default tolerance (0.03) is about twice the
   seed-to-seed SD of the published run. The mean over seeds is also reported.
Exit status 0 = all checks passed, 1 = a check failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hazardfuse.config import load_config, resolve  # noqa: E402

KEYS = ["Macro_F1", "MCC", "Kappa", "ROC_AUC"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tolerance", type=float, default=0.03)
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    tab = resolve(cfg["output_dir"]) / "tables"
    ok = True

    ref_data = cfg.get("reference", {}).get("data")
    if ref_data:
        ref = json.loads(resolve(ref_data).read_text())
        got = json.loads((tab / "data_check.json").read_text())
        rows = [(k, ref[k], got.get(k), ref[k] == got.get(k)) for k in ref]
        print("Data and split (exact):")
        print(pd.DataFrame(rows, columns=["Item", "Reference", "This run", "Match"]).to_string(index=False))
        ok &= all(r[3] for r in rows)

    ref_m = cfg.get("reference", {}).get("metrics")
    if ref_m:
        ref = pd.read_csv(resolve(ref_m)).set_index("Seed")[KEYS]
        run = pd.read_csv(tab / "per_seed_metrics.csv")
        run = run[run["Split"] == "test"].set_index("Seed")[KEYS]
        seeds = [s for s in ref.index if s in run.index]
        d = (run.loc[seeds] - ref.loc[seeds])
        print(f"\nTest metrics per seed, this run − reference (tolerance ±{args.tolerance}):")
        print(pd.concat({"reference": ref.loc[seeds], "this run": run.loc[seeds], "difference": d}, axis=1).round(4).to_string())
        print("\nMean over seeds:\n" + pd.DataFrame({"reference": ref.loc[seeds].mean(), "this run": run.loc[seeds].mean()})
              .round(4).to_string())
        ok &= bool((d.abs() <= args.tolerance).all().all())
    print("\nREPRODUCED" if ok else "\nNOT REPRODUCED (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
