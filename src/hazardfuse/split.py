"""Buffered spatial block split (training / validation / test) matched to target fractions and class balance."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


def block_split(valid, y, block_size=32, fractions=(0.70, 0.15, 0.15), buffer_px=8, trials=500, seed=42) -> dict:
    """
    Square blocks are assigned to training (0), validation (1) or test (2). `trials` random block orders are drawn;
    blocks are filled in that order into test, validation and training up to the target `fractions` of the valid
    pixels, and the order whose subsets best match the fractions and the site class-1 share is kept (every subset
    must contain both classes). Training pixels within `buffer_px` of a validation or test block, and validation pixels
    within `buffer_px` of a test block, form the buffer and are not used. Test blocks are never used for fitting or
    model selection.
    """
    rows, cols = valid.shape
    nbc = int(np.ceil(cols / block_size))
    nb = int(np.ceil(rows / block_size)) * nbc
    rr, cc = np.indices((rows, cols))
    block_id = (rr // block_size) * nbc + (cc // block_size)
    blk_n = np.bincount(block_id[valid], minlength=nb).astype(float)
    blk_pos = np.bincount(block_id[valid], weights=y[valid].astype(float), minlength=nb)
    candidates = np.flatnonzero(blk_n > 0)
    n_valid, p_all = valid.sum(), y[valid].mean()
    f_tr, f_va, f_te = fractions

    def propose(rng):
        order = rng.permutation(candidates)
        frac = blk_n[order] / n_valid
        mid = np.cumsum(frac) - frac / 2
        a = np.full(nb, -1, dtype=int)
        a[order] = np.where(mid < f_te, 2, np.where(mid < f_te + f_va, 1, 0))
        return a

    def score(a):
        s = 0.0
        for code, target in ((2, f_te), (1, f_va), (0, f_tr)):
            sel = a == code
            n, pos = blk_n[sel].sum(), blk_pos[sel].sum()
            if n == 0 or pos == 0 or pos == n:
                return np.inf
            s += abs(n / n_valid - target) + abs(pos / n - p_all)
        return s

    rng, best, best_s = np.random.default_rng(seed), None, np.inf
    for _ in range(trials):
        cand = propose(rng)
        if (s := score(cand)) < best_s:
            best, best_s = cand, s
    if best is None:
        raise RuntimeError("No split contains both classes in every subset; change block_size or trials.")
    split = best[block_id]                                       # -1 empty, 0 training, 1 validation, 2 test
    train = valid & (split == 0) & (distance_transform_edt(~np.isin(split, [1, 2])) > buffer_px)
    val = valid & (split == 1) & (distance_transform_edt(split != 2) > buffer_px)
    test = valid & (split == 2)
    return {"train": train, "val": val, "test": test, "buffer": valid & ~(train | val | test),
            "block_id": block_id, "block_split": split, "score": float(best_s)}


def split_summary(split: dict, valid, y) -> pd.DataFrame:
    n_valid = valid.sum()
    t = pd.DataFrame({name: {"Pixels": int(split[k].sum()), "Share (%)": 100 * split[k].sum() / n_valid,
                             "Class-1 fraction": float(y[split[k]].mean())}
                      for name, k in (("Training", "train"), ("Buffer", "buffer"), ("Validation", "val"), ("Test", "test"))}).T
    t.index.name = "Subset"
    return t
