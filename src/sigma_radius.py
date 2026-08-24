"""Accuracy against test noise for models trained at a grid of noise levels.

Trains one model per sigma if it does not already exist, then extracts the
robustness radius from each curve. Works on either dataset through
config.ACTIVE_DATASET.

Produces: sigma_radius_<dataset>.json and two figures
Paper: Figure 13, Supplementary Figures S9 and S10.
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

SIGMAS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]


def sig_fname(sig):
    ds = C.ACTIVE_DATASET
    if sig == 0.0:
        return f"model_{ds}_binary.keras"
    return f"model_{ds}_binary_sig{int(round(sig * 100)):02d}.keras"


TEST_STD = [round(x, 3) for x in np.arange(0.0, 0.301, 0.025)]
RADIUS_DROP = 0.05
NOISE_REPEATS = getattr(C, "NOISE_REPEATS", 3)


def get_model(data, n_classes, sig):
    fname = C.OUTPUT_DIR / sig_fname(sig)
    if fname.exists():
        print(f"   [load] {fname.name}")
        return tf.keras.models.load_model(fname, compile=False)
    print(f"   [train] {fname.name}  (sigma={sig})")
    tf.keras.backend.clear_session()
    np.random.seed(C.RANDOM_STATE); tf.random.set_seed(C.RANDOM_STATE)
    M.TRAIN_NOISE_STD = float(sig)
    mdl = M.build_model(data["X_train"].shape[1], n_classes)
    ytr = to_categorical(data["y_train"], n_classes)
    yva = to_categorical(data["y_val"], n_classes)
    es = EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0)
    mdl.fit(data["X_train"], ytr, validation_data=(data["X_val"], yva),
            epochs=C.EPOCHS, batch_size=C.BATCH_SIZE, callbacks=[es, rlr], verbose=2)
    mdl.save(fname)
    return mdl


def noise_curve(model, X, y, cont_mask, std_list, repeats):
    accs = []
    for s in std_list:
        rep = []
        for _ in range(max(1, repeats)):
            nz = np.random.normal(0, s, X.shape); nz[:, ~cont_mask] = 0.0
            Xn = np.clip(X + nz, 0.0, 1.0)
            rep.append(accuracy_score(y, np.argmax(model.predict(Xn, verbose=0), axis=1)))
        accs.append(float(np.mean(rep)))
    return accs


def extract_radius(std_list, acc_list, drop):
    clean = acc_list[0]
    floor = clean - drop
    x = np.asarray(std_list, float); y = np.asarray(acc_list, float)
    if y[-1] >= floor:
        return float(x[-1]), True
    for i in range(1, len(x)):
        if y[i] < floor <= y[i - 1]:
            t = (y[i - 1] - floor) / (y[i - 1] - y[i] + 1e-12)
            return float(x[i - 1] + t * (x[i] - x[i - 1])), False
    return 0.0, False


def main():
    np.random.seed(C.RANDOM_STATE); tf.random.set_seed(C.RANDOM_STATE)
    C.K_FEATURES = 1000; C.LOSS_TYPE = "ce"; C.MODE = "binary"
    data = preprocess()
    n_classes = len(np.unique(data["y_train"]))
    cmask = data["continuous_mask"]
    X_test, y_test = data["X_test"], data["y_test"]
    N = min(getattr(C, "ROBUST_SAMPLE", 5000), len(X_test))
    sub = np.random.RandomState(C.RANDOM_STATE).choice(len(X_test), N, replace=False)
    Xs, ys = X_test[sub], y_test[sub]
    print(f">>> σ 网格实验 | 子集 {N} | 连续特征 {int(cmask.sum())}/{len(cmask)} | 测试档位 {TEST_STD}")

    curves, radii, cleans = {}, {}, {}
    for sig in SIGMAS:
        print(f"\n=== sigma = {sig} ===")
        mdl = get_model(data, n_classes, sig)
        if mdl.input_shape[-1] != Xs.shape[1]:
            raise ValueError(f"sigma={sig}: 模型维度 {mdl.input_shape[-1]} != 数据 {Xs.shape[1]};"
                             f" 请把 config 的 K_FEATURES 设回 1000。")
        acc = noise_curve(mdl, Xs, ys, cmask, TEST_STD, NOISE_REPEATS)
        r, censored = extract_radius(TEST_STD, acc, RADIUS_DROP)
        curves[str(sig)] = acc; radii[str(sig)] = r; cleans[str(sig)] = acc[0]
        print(f"   clean={acc[0]:.4f} | robustness radius r(σ)={r:.3f}"
              + ("  (全程未跌破阈值,取最大档位为下界)" if censored else ""))

    json.dump({"sigmas": SIGMAS, "test_std": TEST_STD, "curves": curves,
               "radius": radii, "clean_acc": cleans, "radius_drop": RADIUS_DROP, "n_eval": int(N)},
              open(C.OUTPUT_DIR / f"sigma_radius_{C.ACTIVE_DATASET}.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)


    plt.figure(figsize=(8.0, 5.2))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(SIGMAS)))
    for col, sig in zip(cmap, SIGMAS):
        lbl = "Standard (σ=0)" if sig == 0 else f"σ={sig}"
        plt.plot(TEST_STD, curves[str(sig)], marker="o", ms=4, color=col, label=lbl)
    plt.xlabel("Test-time Gaussian noise std (continuous features)"); plt.ylabel("Accuracy")
    plt.title("Accuracy vs. test noise for each training-noise level (Edge-IIoTset, binary)", fontweight="bold")
    plt.grid(alpha=0.3); plt.legend(title="Training noise")
    plt.tight_layout(); plt.savefig(C.OUTPUT_DIR / f"fig_sigma_curves_{C.ACTIVE_DATASET}.png", dpi=300, bbox_inches="tight")
    plt.close()


    xs = [s for s in SIGMAS if s > 0]
    ys_r = [radii[str(s)] for s in xs]
    plt.figure(figsize=(6.4, 6.0))
    plt.plot(xs, ys_r, marker="s", color="#4C72B0", lw=2, label="Measured radius r(σ)")
    lim = max(max(xs), max(ys_r)) * 1.1
    plt.plot([0, lim], [0, lim], ls="--", color="gray", label="r = σ (reference)")
    for x, y in zip(xs, ys_r):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    plt.xlabel("Training noise level σ")
    plt.ylabel(f"Robustness radius r(σ)  [acc ≥ clean − {RADIUS_DROP:.2f}]")
    plt.title("Robustness radius tracks the training-noise level", fontweight="bold")
    plt.xlim(0, lim); plt.ylim(0, lim); plt.grid(alpha=0.3); plt.legend(loc="upper left")
    plt.tight_layout(); plt.savefig(C.OUTPUT_DIR / f"fig_radius_vs_sigma_{C.ACTIVE_DATASET}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nsaved: sigma_radius.json, fig_sigma_curves.png, fig_radius_vs_sigma.png")


if __name__ == "__main__":
    main()
