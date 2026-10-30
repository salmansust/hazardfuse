"""tf.data pipelines: a patch centred on every labelled pixel (reflect padding at the raster edge), and inference."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def padded(X, patch: int):
    """Reflect-padded copy of a (rows, cols, bands) stack as a TensorFlow constant."""
    p = patch // 2
    return tf.constant(np.pad(X, ((p, p), (p, p), (0, 0)), mode="reflect"))


def augment_patch(x):
    """Random 90-degree rotation and random horizontal/vertical flips."""
    x = tf.image.rot90(x, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    return tf.image.random_flip_up_down(tf.image.random_flip_left_right(x))


def patch_dataset(Xp, mask, labels, class_weights, patch, batch, training, seed, augment=True):
    """(patch, label, class weight) for every pixel in `mask`; the label is that of the center pixel."""
    r, c = np.nonzero(mask)
    y = labels[r, c].astype(np.float32)
    w = np.where(y == 1, class_weights[1], class_weights[0]).astype(np.float32)
    bands = int(Xp.shape[-1])
    ds = tf.data.Dataset.from_tensor_slices((r.astype(np.int32), c.astype(np.int32), y, w))
    if training:
        ds = ds.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)

    def _map(ri, ci, yi, wi):
        x = tf.slice(Xp, [ri, ci, 0], [patch, patch, bands])
        return (augment_patch(x) if (training and augment) else x), yi, wi

    return ds.map(_map, num_parallel_calls=AUTOTUNE).batch(batch).prefetch(AUTOTUNE)


def patch_batches(Xp, mask, patch, batch=4096):
    """Unlabelled patches for every pixel in `mask` (row-major order) and the pixel coordinates."""
    r, c = np.nonzero(mask)
    bands = int(Xp.shape[-1])
    ds = (tf.data.Dataset.from_tensor_slices((r.astype(np.int32), c.astype(np.int32)))
          .map(lambda ri, ci: tf.slice(Xp, [ri, ci, 0], [patch, patch, bands]), num_parallel_calls=AUTOTUNE)
          .batch(batch).prefetch(AUTOTUNE))
    return ds, (r, c)


def predict_map(model, Xp, mask, patch):
    """Probability map P(higher HI); NaN outside `mask`."""
    ds, (r, c) = patch_batches(Xp, mask, patch)
    prob = np.full(mask.shape, np.nan, np.float32)
    prob[r, c] = model.predict(ds, verbose=0).ravel()
    return prob


def band_attention_weights(model, Xp, mask, patch):
    """Mean output of the band-attention gate (one weight per input band) over the pixels in `mask`."""
    sub = tf.keras.Model(model.input, model.get_layer("band_attention").output)
    ds, _ = patch_batches(Xp, mask, patch)
    return sub.predict(ds, verbose=0).mean(axis=0)
