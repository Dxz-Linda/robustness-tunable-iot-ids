"""Robustness radius over three training seeds, plus the evaluation-subset variance check.

Produces the radius table of Section 4.6.6 and the number quoted in Section 4.6.1
for how much the reported accuracy moves when the 5000-record subset is redrawn
without retraining. Resumable: finished seeds are skipped on restart.
"""
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score

import config as C
import model as M
from data_preprocessing import preprocess
from attacks import add_noise, predict, extract_radius

SEEDS = [42, 7, 123]
SIGMAS = [0.02, 0.05, 0.10, 0.15, 0.20]
TEST_STD = [round(x, 3) for x in np.arange(0.0, 0.301, 0.05)]
RADIUS_DROP = 0.05
SUBSET_REPEATS = 5
CKPT = "multiseed_radius_ckpt.json"

KNOWN = {0.0: "model_edge_iiot_binary.keras",
         0.05: "model_edge_iiot_binary_noise05.keras",
         0.10: "model_edge_iiot_binary_noise.keras"}


def train_variant(data, n_classes, sigma):
    tf.keras.backend.clear_session()
    M.TRAIN_NOISE_STD = float(sigma)
    mdl = M.build_model(data["X_train"].shape[1], n_classes)
    ytr = to_categorical(data["y_train"], n_classes)
    yva = to_categorical(data["y_val"], n_classes)
    es = EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE,
                       restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                            min_lr=1e-6, verbose=0)
    mdl.fit(data["X_train"], ytr, validation_data=(data["X_val"], yva),
            epochs=C.EPOCHS, batch_size=C.BATCH_SIZE, callbacks=[es, rlr], verbose=0)
    return mdl


def get_model(data, n_classes, sigma, seed):

    if seed == 42 and sigma in KNOWN and (C.OUTPUT_DIR / KNOWN[sigma]).exists():
        return tf.keras.models.load_model(C.OUTPUT_DIR / KNOWN[sigma], compile=False)
    return train_variant(data, n_classes, sigma)


def run_seed(seed):
    C.RANDOM_STATE = seed
    C.ACTIVE_DATASET = "edge_iiot"
    C.MODE = "binary"
    C.LOSS_TYPE = "ce"
    C.K_FEATURES = 1000
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"\n---- seed {seed} ----")

    data = preprocess()
    n_classes = len(np.unique(data["y_train"]))
    cmask = data["continuous_mask"]
    N = min(getattr(C, "ROBUST_SAMPLE", 5000), len(data["X_test"]))
    sub = np.random.RandomState(seed).choice(len(data["X_test"]), N, replace=False)
    Xs, ys = data["X_test"][sub], data["y_test"][sub]

    res = {"radius": {}}
    for sigma in SIGMAS:
        mdl = get_model(data, n_classes, sigma, seed)
        curve = [float(accuracy_score(ys, predict(mdl, add_noise(Xs, s, cmask))))
                 for s in TEST_STD]
        r, censored = extract_radius(TEST_STD, curve, RADIUS_DROP)
        res["radius"][str(sigma)] = {"clean": curve[0], "radius": r,
                                     "ratio": r / sigma, "censored": censored}
        print(f"    sigma={sigma}: clean={curve[0]:.4f} r={r:.3f} r/sigma={r/sigma:.2f}")

    if seed == SEEDS[0]:
        res["subset_var"] = {}
        for tag, sigma in [("standard", 0.0), ("noise01", 0.10)]:
            mdl = get_model(data, n_classes, sigma, seed)
            clean, noisy = [], []
            for rep in range(SUBSET_REPEATS):
                s2 = np.random.RandomState(1000 + rep).choice(len(data["X_test"]), N, replace=False)
                Xr, yr = data["X_test"][s2], data["y_test"][s2]
                clean.append(float(accuracy_score(yr, predict(mdl, Xr))))
                noisy.append(float(accuracy_score(yr, predict(mdl, add_noise(Xr, 0.1, cmask)))))
            res["subset_var"][tag] = {
                "clean_mean": float(np.mean(clean)), "clean_std": float(np.std(clean, ddof=1)),
                "noise01_mean": float(np.mean(noisy)), "noise01_std": float(np.std(noisy, ddof=1))}
    return res


def main():
    ck = C.OUTPUT_DIR / CKPT
    runs = json.load(open(ck, encoding="utf-8")) if ck.exists() else {}
    for s in SEEDS:
        if str(s) in runs:
            print(f"[skip] seed {s}")
            continue
        runs[str(s)] = run_seed(s)
        json.dump(runs, open(ck, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    used = {str(s): runs[str(s)] for s in SEEDS if str(s) in runs}
    agg = {}
    for sigma in SIGMAS:
        rs = [used[s]["radius"][str(sigma)]["radius"] for s in used]
        ra = [used[s]["radius"][str(sigma)]["ratio"] for s in used]
        cl = [used[s]["radius"][str(sigma)]["clean"] for s in used]
        agg[str(sigma)] = {
            "clean": {"mean": float(np.mean(cl)), "std": float(np.std(cl, ddof=1))},
            "radius": {"mean": float(np.mean(rs)), "std": float(np.std(rs, ddof=1))},
            "ratio": {"mean": float(np.mean(ra)), "std": float(np.std(ra, ddof=1))}}

    sv = used.get(str(SEEDS[0]), {}).get("subset_var", {})
    json.dump({"seeds": SEEDS, "per_seed": used, "aggregate": agg, "subset_var": sv},
              open(C.OUTPUT_DIR / "multiseed_radius.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    L = [f"Robustness radius over seeds={SEEDS}, mean +/- sample std", "",
         f"{'sigma':>8}{'clean accuracy':>22}{'r(sigma)':>22}{'r(sigma)/sigma':>22}", "-" * 74]
    for sigma in SIGMAS:
        a = agg[str(sigma)]
        L.append(f"{sigma:>8}"
                 f"{a['clean']['mean']:>14.4f}+/-{a['clean']['std']:.4f}"
                 f"{a['radius']['mean']:>14.4f}+/-{a['radius']['std']:.4f}"
                 f"{a['ratio']['mean']:>14.2f}+/-{a['ratio']['std']:.2f}")
    L += ["", f"Evaluation-subset variance, seed {SEEDS[0]}, {SUBSET_REPEATS} resamples, no retraining"]
    for tag, d in sv.items():
        L.append(f"  {tag}: clean={d['clean_mean']:.4f}+/-{d['clean_std']:.4f}  "
                 f"acc@noise0.1={d['noise01_mean']:.4f}+/-{d['noise01_std']:.4f}")
    txt = "\n".join(L)
    open(C.OUTPUT_DIR / "multiseed_radius_table.txt", "w", encoding="utf-8").write(txt)
    print("\n" + txt)
    print("\nsaved: multiseed_radius.json, multiseed_radius_table.txt")


if __name__ == "__main__":
    main()
