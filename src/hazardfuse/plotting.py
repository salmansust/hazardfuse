"""Figures. Fonts, sizes and colours come from the `figure` block of the configuration (configs/*.yaml)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# defaults; every key can be changed in the `figure` block of the configuration
STYLE = dict(font_family=["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
             font_size=9, title_size=10, label_size=9, tick_size=8, legend_size=8, text_color="black", dpi=300,
             formats=["png", "pdf", "svg"],
             colors=dict(cls=["#008000", "#d73027"], error=["#4575b4", "#fdae61", "#d73027", "#1a9641", "#e0e0e0"],
                         split=["#a6d96a", "#d9d9d9", "#fdae61", "#7b3294"], probability_cmap="RdYlGn_r",
                         train="#4d4d4d", val="#2a78d6", proposed="#2a78d6", grid="#e6e5e0",
                         groups=["#2a78d6", "#1a9850", "#e08214", "#7b3294", "#b2182b", "#8c510a", "#35978f",
                                 "#c51b7d", "#4d4d4d", "#bababa"]))
CLASS_LABELS = ("Lower HI (0)", "Higher HI (1)")
ERROR_LABELS = ("TN", "FP", "FN", "TP", "Not scored")
SPLIT_LABELS = ("Training", "Buffer", "Validation", "Test")


def apply_style(fig_cfg: dict | None = None) -> dict:
    """Merge the configuration's `figure` block into STYLE and set the matplotlib parameters."""
    cfg = dict(fig_cfg or {})
    colors = {**STYLE["colors"], **cfg.pop("colors", {})}
    if "class" in colors:
        colors["cls"] = colors.pop("class")
    STYLE.update(cfg)
    STYLE["colors"] = colors
    c = STYLE["text_color"]
    plt.rcParams.update({"font.family": "serif", "font.serif": STYLE["font_family"], "mathtext.fontset": "stix",
                         "font.size": STYLE["font_size"], "axes.titlesize": STYLE["title_size"],
                         "axes.labelsize": STYLE["label_size"], "xtick.labelsize": STYLE["tick_size"],
                         "ytick.labelsize": STYLE["tick_size"], "legend.fontsize": STYLE["legend_size"],
                         "pdf.fonttype": 42, "svg.fonttype": "none", "text.color": c, "axes.labelcolor": c,
                         "axes.titlecolor": c, "axes.edgecolor": c, "xtick.color": c, "ytick.color": c,
                         "axes.linewidth": 0.8, "axes.unicode_minus": False})
    return STYLE


def save(fig, out_dir, stem) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in STYLE["formats"]:
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.03,
                    **({"dpi": STYLE["dpi"]} if ext == "png" else {}))
    plt.close(fig)


def _map_axes(ax, title):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, loc="left", fontweight="bold")


def class_and_error_maps(prob, labels, valid, scope=None, threshold=0.5):
    """Class map (0, 1; 255 no data) and error map (0 TN, 1 FP, 2 FN, 3 TP, 4 not scored; 255 no data)."""
    scope = valid if scope is None else scope
    pred, truth = np.nan_to_num(prob) >= threshold, labels.astype(bool)
    cls = np.full(prob.shape, 255, np.uint8); cls[valid] = pred[valid]
    err = np.full(prob.shape, 255, np.uint8); err[valid] = 4
    for code, (t, p) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        err[scope & (truth == t) & (pred == p)] = code
    return cls, err


def fig_split(split, valid, out_dir):
    code = np.full(valid.shape, np.nan)
    for i, k in enumerate(("train", "buffer", "val", "test")):
        code[split[k]] = i
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.imshow(code, cmap=ListedColormap(STYLE["colors"]["split"]), vmin=0, vmax=3, interpolation="nearest")
    _map_axes(ax, "Spatial block split")
    ax.legend(handles=[Patch(fc=c, ec="black", lw=0.4, label=l) for c, l in zip(STYLE["colors"]["split"], SPLIT_LABELS)],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False)
    save(fig, out_dir, "Fig1_split")


def fig_maps(prob, labels, valid, test_mask, out_dir, stem, title, threshold=0.5):
    """(a) probability, (b) classes, (c) errors on the test blocks, (d) errors over the whole area."""
    col = STYLE["colors"]
    cls, err_all = class_and_error_maps(prob, labels, valid, threshold=threshold)
    _, err_test = class_and_error_maps(prob, labels, valid, scope=test_mask, threshold=threshold)
    h = 3.2 * prob.shape[0] / prob.shape[1] + 1.3                  # panel height follows the raster aspect
    fig, axs = plt.subplots(1, 4, figsize=(14, h), gridspec_kw=dict(wspace=0.28))
    fig.subplots_adjust(top=1 - 0.75 / h, bottom=0.55 / h, left=0.01, right=0.99)
    im = axs[0].imshow(np.ma.masked_where(~valid, prob), cmap=col["probability_cmap"], vmin=0, vmax=1,
                       interpolation="nearest")
    fig.colorbar(im, ax=axs[0], fraction=0.035, pad=0.02, label="P(higher HI)")
    axs[1].imshow(np.ma.masked_equal(cls, 255), cmap=ListedColormap(col["cls"]), vmin=0, vmax=1, interpolation="nearest")
    for ax, e in ((axs[2], err_test), (axs[3], err_all)):
        ax.imshow(np.ma.masked_equal(e, 255), cmap=ListedColormap(col["error"]), vmin=0, vmax=4, interpolation="nearest")
        ax.contour(test_mask.astype(float), levels=[0.5], colors="black", linewidths=0.6)
    for ax, t in zip(axs, ("(a) Probability", f"(b) Classes (threshold {threshold:g})", "(c) Errors, test blocks",
                           "(d) Errors, whole area")):
        _map_axes(ax, t)
    handles = ([Patch(fc=c, ec="black", lw=0.4, label=l) for c, l in zip(col["cls"], CLASS_LABELS)]
               + [Patch(fc=c, ec="black", lw=0.4, label=l) for c, l in zip(col["error"], ERROR_LABELS)])
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontweight="bold", y=1 - 0.2 / h)
    save(fig, out_dir, stem)


def fig_training_curves(runs, out_dir, stem="Fig3_training_curves"):
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    styles = ["-", "--", ":", "-."]
    for i, r in enumerate(runs):
        h = r["history"]
        ls = styles[i % len(styles)]
        ax.plot(np.arange(1, len(h["loss"]) + 1), h["loss"], ls, color=STYLE["colors"]["train"], lw=1,
                label=f"Training, seed {r['seed']}")
        ax.plot(np.arange(1, len(h["val_loss"]) + 1), h["val_loss"], ls, color=STYLE["colors"]["val"], lw=1,
                label=f"Validation, seed {r['seed']}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Binary cross-entropy (class-weighted)")
    ax.grid(color=STYLE["colors"]["grid"], lw=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)
    save(fig, out_dir, stem)


def fig_band_attention(mean, sd, groups, out_dir, stem="Fig4_band_attention"):
    """Mean band-attention weight per input band (sorted), coloured by feature group."""
    order = np.argsort(mean.values)[::-1]
    names = mean.index[order]
    gl = list(dict.fromkeys(groups[n] for n in names))
    gcol = {g: STYLE["colors"]["groups"][i % len(STYLE["colors"]["groups"])] for i, g in enumerate(gl)}
    fig, ax = plt.subplots(figsize=(7.2, 0.14 * len(names) + 1.0))
    yy = np.arange(len(names))
    ax.barh(yy, mean.values[order], xerr=None if sd is None else sd.values[order], color=[gcol[groups[n]] for n in names],
            height=0.75, error_kw=dict(lw=0.6, capsize=1.2))
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=STYLE["tick_size"] - 1.5); ax.invert_yaxis()
    ax.set_xlabel("Mean band-attention weight (test pixels; mean ± SD over seeds)")
    ax.grid(axis="x", color=STYLE["colors"]["grid"], lw=0.5); ax.set_axisbelow(True)
    ax.legend(handles=[Patch(fc=gcol[g], label=g) for g in gl], frameon=False, loc="lower right")
    save(fig, out_dir, stem)


def fig_ablation(delta, sd, out_dir, stem, metric="Macro_F1", xlabel="Δ Macro-F1 (variant − full), test blocks"):
    fig, ax = plt.subplots(figsize=(5.4, 0.36 * len(delta) + 1.0))
    yy = np.arange(len(delta))
    vals = delta[metric].values
    ax.barh(yy, vals, xerr=sd[metric].fillna(0).values, color=[STYLE["colors"]["proposed"] if v >= 0 else "#eb6834" for v in vals],
            height=0.6, error_kw=dict(lw=0.6, capsize=1.5))
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(yy); ax.set_yticklabels(delta.index); ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=STYLE["colors"]["grid"], lw=0.5); ax.set_axisbelow(True)
    save(fig, out_dir, stem)
