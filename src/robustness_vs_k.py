"""Robustness against the number of selected features, standard and noise-aware.

Produces: robustness_vs_k.json, fig_robustness_vs_k.png
Paper: Table 9 and Supplementary Figure S5.
"""
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score

import config as C
import model as M
from data_preprocessing import preprocess
from attacks import pgd as pgd_attack

KS = [10, 20, 49]
EPS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
STD = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
VARIANTS = [("standard", 0.0), ("noise-aware", 0.1)]
REP_EPS = 0.02


def train_one(data, n_classes, std):
    tf.keras.backend.clear_session()
    np.random.seed(C.RANDOM_STATE); tf.random.set_seed(C.RANDOM_STATE)
    M.TRAIN_NOISE_STD = std
    mdl = M.build_model(data["X_train"].shape[1], n_classes)
    ytr = to_categorical(data["y_train"], n_classes)
    yva = to_categorical(data["y_val"], n_classes)
    es = EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0)
    mdl.fit(data["X_train"], ytr, validation_data=(data["X_val"], yva),
            epochs=C.EPOCHS, batch_size=C.BATCH_SIZE, callbacks=[es, rlr], verbose=0)
    return mdl


def get_model(data, n_classes, std, K):
    fname = C.OUTPUT_DIR / (f"model_edge_iiot_binary_k{K}.keras" if std == 0
                            else f"model_edge_iiot_binary_k{K}_noise.keras")
    if fname.exists():
        print(f"   [load] {fname.name}")
        return tf.keras.models.load_model(fname, compile=False)
    print(f"   [train] {fname.name}  (sigma={std})")
    mdl = train_one(data, n_classes, std)
    mdl.save(fname)
    return mdl


def main():
    np.random.seed(C.RANDOM_STATE); tf.random.set_seed(C.RANDOM_STATE)
    steps = getattr(C, "PGD_STEPS", 7)
    results = {}
    for K in KS:
        C.K_FEATURES = K
        data = preprocess()
        n_classes = len(np.unique(data["y_train"]))
        Xte, yte, cmask = data["X_test"], data["y_test"], data["continuous_mask"]
        N = min(getattr(C, "ROBUST_SAMPLE", 5000), len(Xte))
        sub = np.random.RandomState(C.RANDOM_STATE).choice(len(Xte), N, replace=False)
        Xs, ys = Xte[sub], yte[sub]
        yoh = to_categorical(ys, n_classes)
        print(f"\n=== K={K} | eval on {N} samples | continuous {int(cmask.sum())}/{len(cmask)} ===")
        for vname, std in VARIANTS:
            mdl = get_model(data, n_classes, std, K)
            if mdl.input_shape[-1] != Xs.shape[1]:
                raise ValueError(f"{vname} K={K}: model dim {mdl.input_shape[-1]} != data {Xs.shape[1]}")
            clean = float(accuracy_score(ys, np.argmax(mdl.predict(Xs, verbose=0), 1)))
            pgd = []
            for e in EPS:
                Xadv = pgd_attack(mdl, Xs, yoh, e, cmask, steps)
                pgd.append(float(accuracy_score(ys, np.argmax(mdl.predict(Xadv, verbose=0), 1))))
            noise = []
            for s in STD:
                nz = np.random.normal(0, s, Xs.shape); nz[:, ~cmask] = 0.0
                Xn = np.clip(Xs + nz, 0.0, 1.0)
                noise.append(float(accuracy_score(ys, np.argmax(mdl.predict(Xn, verbose=0), 1))))
            results[f"K{K}_{vname}"] = {"clean": clean, "eps": EPS, "pgd": pgd, "std": STD, "noise": noise}
            print(f"   {vname:11s}: clean={clean:.4f} | PGD@{REP_EPS}={pgd[EPS.index(REP_EPS)]:.4f} "
                  f"| noise@0.1={noise[STD.index(0.1)]:.4f}")

    json.dump(results, open(C.OUTPUT_DIR / "robustness_vs_k.json", "w"), indent=2)


    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for vname, col, mk in [("standard", "#c1969b", "o"), ("noise-aware", "#6f8fb3", "^")]:
        yv = [results[f"K{K}_{vname}"]["pgd"][EPS.index(REP_EPS)] for K in KS]
        ax.plot(KS, yv, marker=mk, color=col, lw=2.2, label=f"{vname} (PGD ε={REP_EPS})")
    ax.set_xlabel("Number of selected features (K)")
    ax.set_ylabel(f"Accuracy under PGD attack (ε = {REP_EPS})")
    ax.set_xticks(KS); ax.set_ylim(0, 1.0); ax.grid(alpha=0.25); ax.legend(loc="center right")
    ax.set_title("Robustness vs. feature-selection size (Edge-IIoTset, binary)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(C.OUTPUT_DIR / "fig_robustness_vs_k.png", dpi=300, bbox_inches="tight")
    print("\nsaved robustness_vs_k.json + fig_robustness_vs_k.png  —— 把这两个发我")


if __name__ == "__main__":
    main()
