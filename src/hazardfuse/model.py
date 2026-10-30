"""HazardFuse architecture (Keras)."""
from __future__ import annotations

from tensorflow.keras import Model, layers


def build_hazardfuse(input_shape, dropout=0.30, band_attention=True, center_branch=True, spatial_branch=True,
                     fusion_gate=True, name="HazardFuse", **_ignored) -> Model:
    """
    HazardFuse: gated fusion of a center-pixel band-attention branch and a patch spatial-embedding branch.

    Center-pixel band-attention branch: bands of the center pixel, re-weighted by a band-attention gate
        (Dense B/2 ReLU -> Dense B sigmoid); the raw and attended vectors are concatenated and embedded (Dense 96).
    Patch spatial-embedding branch: Conv 3x3 (32) - BN - ReLU, Conv 3x3 (64) - BN - ReLU, max-pool,
        separable conv (96), global average pooling.
    Fusion: concatenation -> Dense 96, multiplied element-wise by a sigmoid fusion gate learned from the same
        concatenation -> dropout -> sigmoid output P(higher HI).
    The keyword switches remove one component each (architecture ablation).
    """
    h, w, bands = input_shape
    if h % 2 == 0 or w % 2 == 0:
        raise ValueError("HazardFuse needs an odd patch size (a defined center pixel).")
    if not (center_branch or spatial_branch):
        raise ValueError("At least one branch must be kept.")
    inp = layers.Input(input_shape, name="patch")
    branches = []

    if center_branch:                                       # center-pixel band-attention branch
        center = layers.Reshape((bands,), name="center_pixel")(
            layers.Cropping2D(((h // 2, h // 2), (w // 2, w // 2)), name="center_crop")(inp))
        if band_attention:
            a = layers.Dense(max(16, bands // 2), activation="relu", name="band_attention_hidden")(center)
            a = layers.Dense(bands, activation="sigmoid", name="band_attention")(a)
            px = layers.Concatenate(name="raw_and_attended")([center, layers.Multiply(name="attended_bands")([center, a])])
        else:
            px = center
        px = layers.Dense(96, activation="relu", name="center_embedding")(px)
        branches.append(layers.Dropout(dropout, name="center_dropout")(px))

    if spatial_branch:                                      # patch spatial-embedding branch
        s = inp
        for i, f in enumerate((32, 64), start=1):
            s = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"spatial_conv{i}")(s)
            s = layers.Activation("relu", name=f"spatial_relu{i}")(layers.BatchNormalization(name=f"spatial_bn{i}")(s))
        s = layers.MaxPooling2D(name="spatial_pool")(s)
        s = layers.SeparableConv2D(96, 3, padding="same", activation="relu", name="spatial_sepconv")(s)
        s = layers.GlobalAveragePooling2D(name="spatial_embedding")(s)
        branches.append(layers.Dropout(dropout, name="spatial_dropout")(s))

    fused = layers.Concatenate(name="fusion")(branches) if len(branches) == 2 else branches[0]
    joint = layers.Dense(96, activation="relu", name="joint_embedding")(fused)
    if fusion_gate and len(branches) == 2:
        gate = layers.Dense(96, activation="sigmoid", name="fusion_gate")(fused)
        joint = layers.Multiply(name="gated_fusion")([joint, gate])
    joint = layers.Dropout(dropout, name="joint_dropout")(joint)
    out = layers.Dense(1, activation="sigmoid", dtype="float32", name="p_higher_hi")(joint)
    return Model(inp, out, name=name)


def architecture_kwargs(model_cfg: dict) -> dict:
    keys = ("dropout", "band_attention", "center_branch", "spatial_branch", "fusion_gate")
    return {k: model_cfg[k] for k in keys if k in model_cfg}
