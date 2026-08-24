"""Accuracy-robustness trade-off over five seeds, with resume support.

Both variants are retrained for every seed, because the spread being measured
comes from training randomness and cannot be obtained by reusing one model.

Produces: multiseed_robustness.json, multiseed_robustness_table.txt
Paper: Table 10.
"""
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import config as C
import model as M
from data_preprocessing import preprocess
from attacks import pgd as _pgd

SEEDS = [42, 7, 123, 2024, 999]
CKPT = "multiseed_robustness_ckpt.json"
NOISE_STD_EVAL = [0.05, 0.1, 0.2]
PGD_EPS_EVAL = [0.02]
VARIANTS = [("standard", 0.0), ("noise-aware", 0.1)]


DO_RADIUS = False
RADIUS_SIGMAS = [0.05, 0.10, 0.20]
RADIUS_TEST_STD = [round(x, 3) for x in np.arange(0.0, 0.301, 0.05)]
RADIUS_DROP = 0.05


def train_variant(data, n_classes, std):
    tf.keras.backend.clear_session()
    M.TRAIN_NOISE_STD = float(std)
    mdl = M.build_model(data["X_train"].shape[1], n_classes)
    ytr = to_categorical(data["y_train"], n_classes); yva = to_categorical(data["y_val"], n_classes)
    es = EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0)
    mdl.fit(data["X_train"], ytr, validation_data=(data["X_val"], yva),
            epochs=C.EPOCHS, batch_size=C.BATCH_SIZE, callbacks=[es, rlr], verbose=0)
    return mdl


def eval_row(model, Xs, ys, cmask, a_idx, n_classes, steps):
    pred = np.argmax(model(tf.constant(Xs, tf.float32), training=False).numpy(), 1)
    clean = accuracy_score(ys, pred)
    pr, rc, _, _ = precision_recall_fscore_support((ys == a_idx).astype(int),
                                                   (pred == a_idx).astype(int),
                                                   average="binary", pos_label=1, zero_division=0)
    yoh = to_categorical(ys, n_classes)
    noise = {}
    for s in NOISE_STD_EVAL:
        nz = np.random.normal(0, s, Xs.shape); nz[:, ~cmask] = 0.0
        Xn = np.clip(Xs + nz, 0.0, 1.0)
        noise[str(s)] = float(accuracy_score(ys, np.argmax(model(tf.constant(Xn, tf.float32), training=False).numpy(), 1)))
    pgd = {}
    for e in PGD_EPS_EVAL:
        Xadv = _pgd(model, Xs, yoh, e, cmask, steps)
        pgd[str(e)] = float(accuracy_score(ys, np.argmax(model(tf.constant(Xadv, tf.float32), training=False).numpy(), 1)))
    return {"clean_acc": float(clean), "attack_recall": float(rc), "attack_precision": float(pr),
            "noise_acc": noise, "pgd_acc": pgd}


def run_seed(seed):
    C.RANDOM_STATE = seed; np.random.seed(seed); tf.random.set_seed(seed)
    C.LOSS_TYPE = "ce"; C.K_FEATURES = 1000
    print(f"\n############ SEED = {seed} ############")
    data = preprocess()
    le = data["label_encoder"]; classes = [str(c) for c in le.classes_]
    a_idx = classes.index("1") if "1" in classes else int(np.argmax(classes))
    n_classes = len(classes); cmask = data["continuous_mask"]
    steps = getattr(C, "PGD_STEPS", 7)
    N = min(getattr(C, "ROBUST_SAMPLE", 5000), len(data["X_test"]))
    sub = np.random.RandomState(seed).choice(len(data["X_test"]), N, replace=False)
    Xs, ys = data["X_test"][sub], data["y_test"][sub]

    res = {}
    for vname, std in VARIANTS:
        mdl = train_variant(data, n_classes, std)
        res[vname] = eval_row(mdl, Xs, ys, cmask, a_idx, n_classes, steps)
        r = res[vname]
        print(f"   {vname:11s}: clean={r['clean_acc']:.4f} rec={r['attack_recall']:.3f} "
              f"acc@0.1={r['noise_acc']['0.1']:.3f} PGD@0.02={r['pgd_acc']['0.02']:.3f}")
    return res


def aggregate(all_runs):
    variants = list(next(iter(all_runs.values())).keys())
    keys = ["clean_acc", "attack_recall", "attack_precision"]
    agg = {}
    for v in variants:
        agg[v] = {}
        for k in keys:
            vals = [all_runs[s][v][k] for s in all_runs]
            agg[v][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
        for grid, name in [(NOISE_STD_EVAL, "noise_acc"), (PGD_EPS_EVAL, "pgd_acc")]:
            agg[v][name] = {}
            for g in grid:
                vals = [all_runs[s][v][name][str(g)] for s in all_runs]
                agg[v][name][str(g)] = {"mean": float(np.mean(vals)),
                                        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
    return agg, variants


def write_table(agg, variants, seeds, path):
    L = [f"Table 7 (multi-seed) over seeds={seeds}  (mean +/- sample std, ddof=1)",
         "Edge-IIoTset, binary | CE | all features", "",
         f"{'Model':<20}{'CleanAcc':>16}{'AtkRecall':>16}{'Acc@0.05':>16}{'Acc@0.1':>16}{'Acc@0.2':>16}{'PGD@0.02':>16}",
         "-" * 116]
    for v in variants:
        a = agg[v]
        def cell(d): return f"{d['mean']:.4f}+/-{d['std']:.4f}"
        L.append(f"{v:<20}"
                 f"{cell(a['clean_acc']):>16}{cell(a['attack_recall']):>16}"
                 f"{cell(a['noise_acc']['0.05']):>16}{cell(a['noise_acc']['0.1']):>16}"
                 f"{cell(a['noise_acc']['0.2']):>16}{cell(a['pgd_acc']['0.02']):>16}")
    txt = "\n".join(L)
    open(path, "w", encoding="utf-8").write(txt)
    print("\n" + txt)


def main():
    p = C.OUTPUT_DIR / CKPT
    all_runs = json.load(open(p, encoding="utf-8")) if p.exists() else {}
    if all_runs:
        print(f">>> 续跑:已有种子 {sorted(int(s) for s in all_runs)},将跳过。删 {CKPT} 可全部重跑。")
    for s in SEEDS:
        if str(s) in all_runs:
            print(f"[skip] seed {s} 已完成")
            continue
        all_runs[str(s)] = run_seed(s)
        json.dump(all_runs, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    runs = {str(s): all_runs[str(s)] for s in SEEDS if str(s) in all_runs}
    agg, variants = aggregate(runs)
    json.dump({"seeds": SEEDS, "per_seed": runs, "aggregate": agg},
              open(C.OUTPUT_DIR / "multiseed_robustness.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    write_table(agg, variants, SEEDS, C.OUTPUT_DIR / "multiseed_robustness_table.txt")
    print("\nsaved: multiseed_robustness_ckpt.json, multiseed_robustness.json, multiseed_robustness_table.txt")


if __name__ == "__main__":
    main()
