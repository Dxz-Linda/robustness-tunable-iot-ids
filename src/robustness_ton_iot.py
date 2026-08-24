"""Replication of the robustness analysis on ToN-IoT.

Switches the configuration to ToN-IoT binary and trains what is missing.

Produces: robustness_ton_iot.json, fig_robustness_ton_iot.png
Paper: Figure 9.
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


C.ACTIVE_DATASET = "ton_iot"
C.MODE = "binary"

C.K_FEATURES = 1000

EPS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
STD = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
VARIANTS = [("standard", 0.0), ("noise-aware", 0.1)]
STD_MODEL = "model_ton_iot_binary.keras"
NOISE_MODEL = "model_ton_iot_binary_noise.keras"


def train_one(data, n_classes, std):
    tf.keras.backend.clear_session()
    np.random.seed(C.RANDOM_STATE)
    tf.random.set_seed(C.RANDOM_STATE)
    M.TRAIN_NOISE_STD = std
    mdl = M.build_model(data["X_train"].shape[1], n_classes)
    ytr = to_categorical(data["y_train"], n_classes)
    yva = to_categorical(data["y_val"], n_classes)
    es = EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0)
    mdl.fit(data["X_train"], ytr, validation_data=(data["X_val"], yva),
            epochs=C.EPOCHS, batch_size=C.BATCH_SIZE, callbacks=[es, rlr], verbose=0)
    return mdl


def get_model(data, n_classes, std):
    fname = C.OUTPUT_DIR / (STD_MODEL if std == 0 else NOISE_MODEL)
    if fname.exists():
        print("   [load] " + fname.name)
        return tf.keras.models.load_model(fname, compile=False)
    print("   [train] " + fname.name + "  (sigma=" + str(std) + ")")
    mdl = train_one(data, n_classes, std)
    mdl.save(fname)
    return mdl


def main():
    np.random.seed(C.RANDOM_STATE)
    tf.random.set_seed(C.RANDOM_STATE)
    steps = getattr(C, "PGD_STEPS", 7)
    data = preprocess()
    n_classes = len(np.unique(data["y_train"]))
    Xte, yte, cmask = data["X_test"], data["y_test"], data["continuous_mask"]
    N = min(getattr(C, "ROBUST_SAMPLE", 5000), len(Xte))
    sub = np.random.RandomState(C.RANDOM_STATE).choice(len(Xte), N, replace=False)
    Xs, ys = Xte[sub], yte[sub]
    yoh = to_categorical(ys, n_classes)
    print("ToN-IoT binary | eval on " + str(N) + " samples | continuous "
          + str(int(cmask.sum())) + "/" + str(len(cmask)) + " features")

    results = {}
    for vname, std in VARIANTS:
        mdl = get_model(data, n_classes, std)
        clean = float(accuracy_score(ys, np.argmax(mdl.predict(Xs, verbose=0), 1)))
        pgd = []
        for e in EPS:
            Xadv = pgd_attack(mdl, Xs, yoh, e, cmask, steps)
            pgd.append(float(accuracy_score(ys, np.argmax(mdl.predict(Xadv, verbose=0), 1))))
        noise = []
        for s in STD:
            nz = np.random.normal(0, s, Xs.shape)
            nz[:, ~cmask] = 0.0
            Xn = np.clip(Xs + nz, 0.0, 1.0)
            noise.append(float(accuracy_score(ys, np.argmax(mdl.predict(Xn, verbose=0), 1))))
        results[vname] = {"clean": clean, "eps": EPS, "pgd": pgd, "std": STD, "noise": noise}
        print("   " + vname.ljust(11) + ": clean=" + format(clean, ".4f")
              + " | PGD@0.02=" + format(pgd[EPS.index(0.02)], ".4f")
              + " | noise@0.1=" + format(noise[STD.index(0.1)], ".4f"))

    json.dump(results, open(C.OUTPUT_DIR / "robustness_ton_iot.json", "w"), indent=2)

    C_ROSE, C_BLUE = "#c1969b", "#6f8fb3"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    a1.plot(EPS, results["standard"]["pgd"], marker="o", color=C_ROSE, lw=2, label="Standard")
    a1.plot(EPS, results["noise-aware"]["pgd"], marker="^", color=C_BLUE, lw=2, label="Noise-aware (sigma=0.1)")
    a1.set_xlabel("PGD budget epsilon")
    a1.set_ylabel("Accuracy under PGD")
    a1.set_ylim(0, 1)
    a1.grid(alpha=0.25)
    a1.legend()
    a1.set_title("PGD attack (ToN-IoT, binary)", fontweight="bold")
    a2.plot(STD, results["standard"]["noise"], marker="o", color=C_ROSE, lw=2, label="Standard")
    a2.plot(STD, results["noise-aware"]["noise"], marker="^", color=C_BLUE, lw=2, label="Noise-aware (sigma=0.1)")
    a2.set_xlabel("Test Gaussian noise sigma")
    a2.set_ylabel("Accuracy under noise")
    a2.set_ylim(0, 1)
    a2.grid(alpha=0.25)
    a2.legend()
    a2.set_title("Gaussian noise (ToN-IoT, binary)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(C.OUTPUT_DIR / "fig_robustness_ton_iot.png", dpi=300, bbox_inches="tight")
    print("\nsaved robustness_ton_iot.json + fig_robustness_ton_iot.png  -- send me these two")


if __name__ == "__main__":
    main()
