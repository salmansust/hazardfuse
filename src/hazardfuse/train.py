"""Training of HazardFuse with seed repeats: Adam, early stopping and learning-rate reduction on validation loss."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from .metrics import compute_metrics
from .model import architecture_kwargs, build_hazardfuse
from .pipeline import patch_dataset, predict_map
from .utils import set_seed, slug


def class_weights(y_train, use=True):
    """(w0, w1) = n / (2 n_c): inverse class frequency of the training pixels; (1, 1) when `use` is False."""
    n, n1 = len(y_train), int(np.sum(y_train))
    if not use:
        return 1.0, 1.0
    return n / (2 * (n - n1)), n / (2 * n1)


def fit_one(X, labels, masks, cfg, seed, ckpt_dir, model_overrides=None, tag=None):
    """
    Fit HazardFuse once. X: standardised input (rows, cols, bands); labels: 0/1 grid; masks: dict with 'train', 'val'
    and 'valid'. Returns (model, probability map over 'valid', history, seconds, epochs).
    """
    tr = cfg["training"]
    mcfg = {**cfg["model"], **(model_overrides or {})}
    patch = int(mcfg["patch"])
    cw = class_weights(labels[masks["train"]], cfg.get("class_weights", True))
    Xp = tf.constant(np.pad(X, ((patch // 2,) * 2, (patch // 2,) * 2, (0, 0)), mode="reflect"))
    set_seed(seed, cfg.get("deterministic", True))
    # same order as the published notebook: seed -> build (weight initialisation) -> datasets -> fit
    model = build_hazardfuse((patch, patch, X.shape[-1]), **architecture_kwargs(mcfg))
    train_ds = patch_dataset(Xp, masks["train"], labels, cw, patch, tr["batch"], True, seed, tr.get("augment", True))
    val_ds = patch_dataset(Xp, masks["val"], labels, cw, patch, tr["batch"], False, seed)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tr["learning_rate"]), loss="binary_crossentropy")
    ckpt = Path(ckpt_dir) / f"{slug(tag or f'HazardFuse seed {seed}')}.weights.h5"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [EarlyStopping(monitor="val_loss", patience=tr["patience"], restore_best_weights=True),
                 ReduceLROnPlateau(monitor="val_loss", factor=tr["lr_factor"], patience=tr["lr_patience"], min_lr=tr["min_lr"]),
                 ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True, save_weights_only=True)]
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=tr["epochs"], callbacks=callbacks, verbose=0)
    seconds = time.time() - t0
    model.load_weights(str(ckpt))                                   # weights of the lowest validation loss
    prob = predict_map(model, Xp, masks["valid"], patch)
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    return model, prob, history, seconds, len(history.get("loss", []))


def run_seeds(X, labels, masks, cfg, seeds, ckpt_dir, model_dir=None, model_overrides=None, label="HazardFuse",
              threshold=0.5, log=print):
    """Fit one model per seed; metrics on every split. With `model_dir`, every fitted model is saved (.keras)."""
    runs = []
    for seed in seeds:
        tag = f"{label} seed {seed}"
        model, prob, history, seconds, epochs = fit_one(X, labels, masks, cfg, seed, ckpt_dir, model_overrides, tag)
        rec = {"seed": int(seed), "prob": prob, "history": history, "seconds": seconds, "epochs": epochs,
               "params": int(model.count_params()),
               "metrics": {s: compute_metrics(labels[masks[s]], prob[masks[s]], threshold) for s in ("train", "val", "test")}}
        if model_dir is not None:
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            rec["model_file"] = str(Path(model_dir) / f"{slug(label)}_seed{seed}.keras")
            model.save(rec["model_file"])
        m = rec["metrics"]["test"]
        log(f"  {tag:<44s} test Macro-F1 {m['Macro_F1']:.4f} | MCC {m['MCC']:.4f} | kappa {m['Kappa']:.4f} | "
            f"ROC-AUC {m['ROC_AUC']:.4f} | {seconds:6.1f} s | {epochs} epochs")
        runs.append(rec)
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    return runs
