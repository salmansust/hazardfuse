"""Model bundles (one .keras file per seed + manifest.json) and application to another site."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from .data import band_index, band_stats, load_site, standardise
from .metrics import compute_metrics
from .utils import environment_info, slug


def export_bundle(out_dir, runs, cfg, bands, band_mean, band_std, train_class1_fraction, class_weights,
                  test_summary=None) -> Path:
    """Self-contained folder: every seed's model and manifest.json with everything needed to re-apply it."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files, seeds = [], []
    for r in runs:
        if r.get("model_file"):
            dst = out / Path(r["model_file"]).name
            if Path(r["model_file"]).resolve() != dst.resolve():
                shutil.copy2(r["model_file"], dst)
            files.append(dst.name); seeds.append(r["seed"])
    env = environment_info()
    manifest = {
        "model": "HazardFuse", "architecture": cfg["model"], "training_site": cfg["site"],
        "created": datetime.now().isoformat(timespec="seconds"),
        "tensorflow": env["packages"].get("tensorflow"), "keras": env["packages"].get("keras"),
        "seeds": seeds, "files": files,
        "bands": list(bands), "band_mean": np.asarray(band_mean).tolist(), "band_std": np.asarray(band_std).tolist(),
        "band_rename": cfg["data"].get("band_rename"),
        "nan_fill": "0 after standardisation (the training mean)",
        "patch": int(cfg["model"]["patch"]), "threshold": float(cfg["threshold"]),
        "train_class1_fraction": float(train_class1_fraction), "class_weights": list(map(float, class_weights)),
        "split": cfg["split"], "test_metrics_mean_over_seeds": test_summary or {},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


def load_bundle(bundle_dir, seeds=None):
    import tensorflow as tf
    bundle_dir = Path(bundle_dir)
    man = json.loads((bundle_dir / "manifest.json").read_text())
    pairs = [(s, f) for s, f in zip(man["seeds"], man["files"]) if seeds is None or s in seeds]
    return man, [tf.keras.models.load_model(bundle_dir / f, compile=False) for _, f in pairs]


def adapt_batchnorm(model, X, mask, patch, n=2000, seed=42) -> int:
    """AdaBN: replace the BatchNorm moving statistics by those of the target site, layer by layer (no labels)."""
    import tensorflow as tf
    bns = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
    if not bns:
        return 0
    rng = np.random.default_rng(seed)
    r, c = np.nonzero(mask)
    k = rng.choice(len(r), size=min(n, len(r)), replace=False)
    p = patch // 2
    Xpad = np.pad(X, ((p, p), (p, p), (0, 0)), mode="reflect")
    xs = np.stack([Xpad[i:i + patch, j:j + patch] for i, j in zip(r[k], c[k])])
    for bn in bns:
        feat = tf.keras.Model(model.input, bn.input).predict(xs, batch_size=256, verbose=0)
        axes = tuple(range(feat.ndim - 1))
        bn.moving_mean.assign(feat.mean(axis=axes).astype("float32"))
        bn.moving_variance.assign(feat.var(axis=axes).astype("float32"))
    return len(bns)


def resolve_threshold(prob, valid, man, rule="source") -> float:
    if rule == "source":
        return float(man["threshold"])
    if rule == "prevalence":                    # same predicted class-1 share as the training labels; no target labels
        return float(np.quantile(prob[valid], 1 - man["train_class1_fraction"]))
    return float(rule)


def transfer_to_site(bundle_dir, features_tif, labels_tif=None, standardise_with="target", adabn=True,
                     threshold="source", seeds=None, seed=42):
    """Apply a bundle to a site; returns the seed-ensemble probability, threshold, valid mask and (with labels) metrics."""
    from .pipeline import padded, predict_map
    man, models = load_bundle(bundle_dir, seeds)
    site = load_site(features_tif, labels_tif, man.get("band_rename"))
    X = site.X_raw[..., band_index(site.names, man["bands"])]
    if standardise_with == "source":
        mean, std = np.asarray(man["band_mean"]), np.asarray(man["band_std"])
    elif standardise_with == "target":
        mean, std = band_stats(X, site.valid)
    else:
        raise ValueError("standardise_with must be 'source' or 'target'")
    Xs = standardise(X, site.valid, mean, std)
    Xp = padded(Xs, man["patch"])
    probs = []
    for m in models:
        if adabn:
            adapt_batchnorm(m, Xs, site.valid, man["patch"], seed=seed)
        probs.append(predict_map(m, Xp, site.valid, man["patch"]))
    prob = np.mean(probs, axis=0)
    thr = resolve_threshold(prob, site.valid, man, threshold)
    res = {"model": man["model"], "training_site": man["training_site"], "standardise": standardise_with,
           "adabn": bool(adabn), "threshold": thr, "members": len(models), "pixels": int(site.valid.sum()),
           "class1_share": float((prob[site.valid] >= thr).mean()), "prob": prob, "site": site}
    if site.y is not None:
        res["metrics"] = compute_metrics(site.y[site.valid], prob[site.valid], thr)
    return res
