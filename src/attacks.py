"""Shared perturbation and evaluation helpers.

Every robustness script imports the PGD routine from here so that all reported
attacks use one identical implementation.
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

import config as C


def pgd(model, X, y_onehot, eps, mask, steps=None):
    """L-infinity PGD restricted to the features flagged by mask.

    The gradient is masked so that categorical and protocol fields are never
    perturbed, and the result is projected back into the eps-ball and clipped
    to the valid [0, 1] range.
    """
    steps = steps or getattr(C, "PGD_STEPS", 7)
    Xo = tf.convert_to_tensor(X, tf.float32)
    m = tf.constant(mask.astype("float32"))
    alpha = (2.5 * eps / steps) if steps > 0 else 0.0
    Xadv = tf.identity(Xo)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(Xadv)
            loss = loss_fn(y_onehot, model(Xadv, training=False))
        g = tape.gradient(loss, Xadv)
        Xadv = Xadv + alpha * tf.sign(g) * m
        Xadv = tf.clip_by_value(Xadv - Xo, -eps, eps) + Xo
        Xadv = tf.clip_by_value(Xadv, 0.0, 1.0)
    return Xadv.numpy()


def add_noise(X, std, mask, rng=None):
    """Zero-mean Gaussian noise on the masked features, clipped to [0, 1]."""
    rng = rng or np.random
    nz = rng.normal(0, std, X.shape)
    nz[:, ~mask] = 0.0
    return np.clip(X + nz, 0.0, 1.0)


def predict(model, X):
    return np.argmax(model(tf.convert_to_tensor(X, tf.float32),
                           training=False).numpy(), axis=1)


def accuracy(model, X, y):
    return float(accuracy_score(y, predict(model, X)))


def robustness_subset(X, y, n=None, seed=None):
    """Fixed random subset used for every robustness evaluation."""
    n = min(n or getattr(C, "ROBUST_SAMPLE", 5000), len(X))
    seed = C.RANDOM_STATE if seed is None else seed
    idx = np.random.RandomState(seed).choice(len(X), n, replace=False)
    return X[idx], y[idx]


def attack_index(label_encoder):
    classes = [str(c) for c in label_encoder.classes_]
    return classes.index("1") if "1" in classes else int(np.argmax(classes))


def extract_radius(std_grid, acc_curve, drop=0.05):
    """Largest test-noise level at which accuracy stays within drop of clean.

    Returns the radius and a flag marking values censored by the end of the grid.
    """
    clean = acc_curve[0]
    floor = clean - drop
    x, y = np.asarray(std_grid, float), np.asarray(acc_curve, float)
    if y[-1] >= floor:
        return float(x[-1]), True
    for i in range(1, len(x)):
        if y[i] < floor <= y[i - 1]:
            t = (y[i - 1] - floor) / (y[i - 1] - y[i] + 1e-12)
            return float(x[i - 1] + t * (x[i] - x[i - 1])), False
    return 0.0, False
