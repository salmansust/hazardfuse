import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from hazardfuse.split import block_split, split_summary


def site(rows=200, cols=260, seed=0):
    rng = np.random.default_rng(seed)
    h = gaussian_filter(rng.normal(size=(rows, cols)), 10)
    valid = np.ones((rows, cols), bool); valid[:5, :40] = False
    y = (h > np.quantile(h, 0.26)).astype(np.uint8)
    return valid, y


def test_split_properties():
    valid, y = site()
    s = block_split(valid, y, block_size=32, fractions=(0.7, 0.15, 0.15), buffer_px=8, trials=300, seed=42)
    tr, va, te = s["train"], s["val"], s["test"]
    assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
    assert ((tr | va | te | s["buffer"]) == valid).all()
    # buffer: no training pixel within 8 px of a held-out block, no validation pixel within 8 px of a test block
    assert distance_transform_edt(~np.isin(s["block_split"], [1, 2]))[tr].min() > 8
    assert distance_transform_edt(s["block_split"] != 2)[va].min() > 8
    held = np.isin(s["block_split"], [1, 2]) & valid
    assert abs(held.sum() / valid.sum() - 0.30) < 0.08
    for m in (tr, va, te):
        assert 0 < y[m].mean() < 1
    t = split_summary(s, valid, y)
    assert t["Pixels"].sum() == valid.sum()


def test_split_deterministic():
    valid, y = site()
    a = block_split(valid, y, trials=100, seed=42)
    b = block_split(valid, y, trials=100, seed=42)
    c = block_split(valid, y, trials=100, seed=7)
    assert (a["block_split"] == b["block_split"]).all()
    assert not (a["block_split"] == c["block_split"]).all()
