"""Robustness in the multiclass setting, reporting accuracy and macro-F1.

Macro-F1 is the informative metric here because accuracy is propped up by the
dominant normal class. Run once per dataset.

Produces: robustness_multiclass_<dataset>.json and the matching figure
Paper: Figure 10 and Supplementary Figure S7.
"""
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score

import config as C
import model as M
from data_preprocessing import preprocess
from attacks import pgd as _pgd


DATASET = None

STD_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
EPS_GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
VARIANTS = [("standard", 0.0), ("noise-aware", 0.1)]


def train_variant(data, n_classes, std):
    tf.keras.backend.clear_session()
    np.random.seed(C.RANDOM_STATE); tf.random.set_seed(C.RANDOM_STATE)
    M.TRAIN_NOISE_STD = float(std)
    mdl = M.build_model(data["X_train"].shape[1], n_classes)
    ytr = to_categorical(data["y_train"], n_classes); yva = to_categorical(data["y_val"], n_classes)
    es = EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0)
    mdl.fit(data["X_train"], ytr, validation_data=(data["X_val"], yva),
            epochs=C.EPOCHS, batch_size=C.BATCH_SIZE, callbacks=[es, rlr], verbose=2)
    return mdl


def get_model(data, n_classes, std, ds):
    fname = C.OUTPUT_DIR / (f"model_{ds}_multiclass.keras" if std == 0
                            else f"model_{ds}_multiclass_noise.keras")
    if fname.exists():
        print(f"   [load] {fname.name}")
        return tf.keras.models.load_model(fname, compile=False)
    print(f"   [train] {fname.name} (sigma={std})")
    mdl = train_variant(data, n_classes, std)
    mdl.save(fname)
    return mdl


def curves(model, Xs, ys, cmask, n_classes, steps):
    def metrics(Xin):
        pred = np.argmax(model(tf.constant(Xin, tf.float32), training=False).numpy(), 1)
        return float(accuracy_score(ys, pred)), float(f1_score(ys, pred, average="macro", zero_division=0))
    yoh = to_categorical(ys, n_classes)
    acc_n, f1_n = [], []
    for s in STD_GRID:
        nz = np.random.normal(0, s, Xs.shape); nz[:, ~cmask] = 0.0
        a, f = metrics(np.clip(Xs + nz, 0.0, 1.0)); acc_n.append(a); f1_n.append(f)
    acc_p, f1_p = [], []
    for e in EPS_GRID:
        Xadv = _pgd(model, Xs, yoh, e, cmask, steps) if e > 0 else Xs
        a, f = metrics(Xadv); acc_p.append(a); f1_p.append(f)
    return {"std_grid": STD_GRID, "noise_acc": acc_n, "noise_f1": f1_n,
            "eps_grid": EPS_GRID, "pgd_acc": acc_p, "pgd_f1": f1_p}


def main():
    if DATASET is not None:
        C.ACTIVE_DATASET = DATASET
    C.MODE = "multiclass"; C.K_FEATURES = 1000; C.LOSS_TYPE = "ce"
    ds = C.ACTIVE_DATASET
    np.random.seed(C.RANDOM_STATE); tf.random.set_seed(C.RANDOM_STATE)

    data = preprocess()
    n_classes = len(np.unique(data["y_train"])); cmask = data["continuous_mask"]
    steps = getattr(C, "PGD_STEPS", 7)
    N = min(getattr(C, "ROBUST_SAMPLE", 5000), len(data["X_test"]))
    sub = np.random.RandomState(C.RANDOM_STATE).choice(len(data["X_test"]), N, replace=False)
    Xs, ys = data["X_test"][sub], data["y_test"][sub]
    print(f">>> 多分类鲁棒性 | 数据集={ds} | 类别数={n_classes} | 子集 {N} | 连续 {int(cmask.sum())}/{len(cmask)}")

    out = {}
    for vname, std in VARIANTS:
        mdl = get_model(data, n_classes, std, ds)
        if mdl.input_shape[-1] != Xs.shape[1]:
            raise ValueError(f"{vname}: 维度 {mdl.input_shape[-1]} != {Xs.shape[1]};把 K_FEATURES 设回 1000。")
        out[vname] = curves(mdl, Xs, ys, cmask, n_classes, steps)
        c = out[vname]
        print(f"   {vname:11s}: cleanAcc={c['noise_acc'][0]:.4f} cleanF1={c['noise_f1'][0]:.4f} "
              f"| noise@0.1 acc={c['noise_acc'][STD_GRID.index(0.1)]:.3f} f1={c['noise_f1'][STD_GRID.index(0.1)]:.3f} "
              f"| PGD@0.02 acc={c['pgd_acc'][EPS_GRID.index(0.02)]:.3f} f1={c['pgd_f1'][EPS_GRID.index(0.02)]:.3f}")

    json.dump({"dataset": ds, "n_classes": n_classes, "n_eval": int(N), "models": out},
              open(C.OUTPUT_DIR / f"robustness_multiclass_{ds}.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)


    style = {"standard": "#c1969b", "noise-aware": "#4C72B0"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.0))
    for v in ["standard", "noise-aware"]:
        c = out[v]; col = style[v]
        a1.plot(c["std_grid"], c["noise_acc"], marker="o", color=col, lw=2, label=f"{v} — Accuracy")
        a1.plot(c["std_grid"], c["noise_f1"], marker="^", ls="--", color=col, lw=1.8, label=f"{v} — Macro-F1")
        a2.plot(c["eps_grid"], c["pgd_acc"], marker="o", color=col, lw=2, label=f"{v} — Accuracy")
        a2.plot(c["eps_grid"], c["pgd_f1"], marker="^", ls="--", color=col, lw=1.8, label=f"{v} — Macro-F1")
    a1.set_xlabel("Test Gaussian noise std"); a1.set_ylabel("Score"); a1.set_ylim(0, 1)
    a1.grid(alpha=0.25); a1.legend(fontsize=8); a1.set_title(f"Gaussian noise ({ds}, multiclass)", fontweight="bold")
    a2.set_xlabel("PGD budget epsilon"); a2.set_ylabel("Score"); a2.set_ylim(0, 1)
    a2.grid(alpha=0.25); a2.legend(fontsize=8); a2.set_title(f"PGD ({ds}, multiclass)", fontweight="bold")
    fig.tight_layout(); fig.savefig(C.OUTPUT_DIR / f"fig_robustness_multiclass_{ds}.png", dpi=300, bbox_inches="tight")
    print(f"\nsaved: robustness_multiclass_{ds}.json, fig_robustness_multiclass_{ds}.png")


if __name__ == "__main__":
    main()
