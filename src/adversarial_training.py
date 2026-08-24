"""Madry-style PGD adversarial training compared against noise-aware training.

Uses a custom training loop because Keras fit does not support an inner attack.

Produces: model_edge_iiot_binary_advtrain.keras, advtrain_compare.json, advtrain_table.txt
Paper: Table 11 and Figure 8.
"""
import json
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import config as C
import model as M
from data_preprocessing import preprocess
from attacks import pgd as _pgd_eval


ADV_EPS = 0.03
ADV_STEPS = 7
EPOCHS_AT = 40
LR_AT = 1e-3
CLEAN_RATIO = 0.0
AT_MODEL = "model_edge_iiot_binary_advtrain.keras"
STD_MODEL = "model_edge_iiot_binary.keras"
NOISE_MODEL = "model_edge_iiot_binary_noise.keras"


STD_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
EPS_GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
REP_EPS = 0.02


@tf.function
def _pgd_train(model, x, yoh, eps, mask, steps, alpha):
    Xo = x
    Xadv = Xo + tf.random.uniform(tf.shape(Xo), -eps, eps) * mask
    Xadv = tf.clip_by_value(Xadv, 0.0, 1.0)
    lo = tf.keras.losses.CategoricalCrossentropy()
    for _ in tf.range(steps):
        with tf.GradientTape() as t:
            t.watch(Xadv)
            loss = lo(yoh, model(Xadv, training=False))
        g = t.gradient(loss, Xadv)
        Xadv = Xadv + alpha * tf.sign(g) * mask
        Xadv = tf.clip_by_value(Xadv - Xo, -eps, eps) + Xo
        Xadv = tf.clip_by_value(Xadv, 0.0, 1.0)
    return Xadv


def train_at(data, n_classes, cmask):
    tf.keras.backend.clear_session()
    np.random.seed(C.RANDOM_STATE); tf.random.set_seed(C.RANDOM_STATE)
    M.TRAIN_NOISE_STD = 0.0
    model = M.build_model(data["X_train"].shape[1], n_classes)
    opt = tf.keras.optimizers.Adam(LR_AT)
    lossfn = tf.keras.losses.CategoricalCrossentropy()
    mask = tf.constant(cmask.astype("float32"))
    alpha = tf.constant(2.5 * ADV_EPS / ADV_STEPS, tf.float32)
    eps = tf.constant(ADV_EPS, tf.float32)
    steps = tf.constant(ADV_STEPS, tf.int32)

    Xtr = data["X_train"].astype("float32"); ytr = to_categorical(data["y_train"], n_classes)
    Xva = data["X_val"].astype("float32"); yva = data["y_val"]
    n = len(Xtr); bs = C.BATCH_SIZE
    best_val, best_w = -1.0, None

    @tf.function
    def step(xb, yb, use_adv):
        xin = tf.cond(use_adv,
                      lambda: _pgd_train(model, xb, yb, eps, mask, steps, alpha),
                      lambda: xb)
        with tf.GradientTape() as tape:
            pred = model(xin, training=True)
            loss = lossfn(yb, pred)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    for ep in range(EPOCHS_AT):
        idx = np.random.permutation(n); t0 = time.time(); running = 0.0; nb = 0
        for s in range(0, n, bs):
            b = idx[s:s + bs]
            use_adv = tf.constant((CLEAN_RATIO == 0.0) or (np.random.rand() >= CLEAN_RATIO))
            l = step(tf.constant(Xtr[b]), tf.constant(ytr[b]), use_adv)
            running += float(l); nb += 1
        val_acc = accuracy_score(yva, np.argmax(model.predict(Xva, batch_size=bs, verbose=0), 1))
        print(f"  epoch {ep+1:2d}/{EPOCHS_AT} | adv-loss={running/max(nb,1):.4f} "
              f"| val clean acc={val_acc:.4f} | {time.time()-t0:.0f}s")
        if val_acc > best_val:
            best_val, best_w = val_acc, model.get_weights()
    if best_w is not None:
        model.set_weights(best_w)
    model.save(C.OUTPUT_DIR / AT_MODEL)
    return model


def eval_model(model, X, y, cmask, a_idx, n_classes, steps, repeats):
    yb = (y == a_idx).astype(int)
    pred = np.argmax(model(tf.constant(X, tf.float32), training=False).numpy(), 1)
    clean = accuracy_score(y, pred)
    pr, rc, f1, _ = precision_recall_fscore_support((y == a_idx).astype(int),
                                                    (pred == a_idx).astype(int),
                                                    average="binary", pos_label=1, zero_division=0)
    yoh = to_categorical(y, n_classes)
    noise = []
    for s in STD_GRID:
        acc = []
        for _ in range(max(1, repeats)):
            nz = np.random.normal(0, s, X.shape); nz[:, ~cmask] = 0.0
            Xn = np.clip(X + nz, 0.0, 1.0)
            acc.append(accuracy_score(y, np.argmax(model(tf.constant(Xn, tf.float32), training=False).numpy(), 1)))
        noise.append(float(np.mean(acc)))
    pgd = []
    for e in EPS_GRID:
        Xadv = _pgd_eval(model, X, yoh, e, cmask, steps)
        pgd.append(float(accuracy_score(y, np.argmax(model(tf.constant(Xadv, tf.float32), training=False).numpy(), 1))))
    return {"clean_acc": float(clean), "attack_recall": float(rc), "attack_precision": float(pr),
            "std_grid": STD_GRID, "noise": noise, "eps_grid": EPS_GRID, "pgd": pgd}


def main():
    np.random.seed(C.RANDOM_STATE); tf.random.set_seed(C.RANDOM_STATE)
    data = preprocess()
    le = data["label_encoder"]; classes = [str(c) for c in le.classes_]
    if len(classes) != 2:
        raise SystemExit("本脚本针对二分类;请把 config 设为 MODE='binary'。")
    a_idx = classes.index("1") if "1" in classes else int(np.argmax(classes))
    n_classes = len(classes); cmask = data["continuous_mask"]
    steps = getattr(C, "PGD_STEPS", 7); repeats = getattr(C, "NOISE_REPEATS", 3)


    at_path = C.OUTPUT_DIR / AT_MODEL
    if at_path.exists():
        print(f"[load] {AT_MODEL}")
        at = tf.keras.models.load_model(at_path, compile=False)
    else:
        print(f"[train] PGD-AT (eps={ADV_EPS}, steps={ADV_STEPS}, clean_ratio={CLEAN_RATIO})")
        at = train_at(data, n_classes, cmask)


    N = min(getattr(C, "ROBUST_SAMPLE", 5000), len(data["X_test"]))
    sub = np.random.RandomState(C.RANDOM_STATE).choice(len(data["X_test"]), N, replace=False)
    Xs, ys = data["X_test"][sub], data["y_test"][sub]

    out = {}
    out["Adversarial training (PGD)"] = eval_model(at, Xs, ys, cmask, a_idx, n_classes, steps, repeats)
    for name, fname in [("Standard", STD_MODEL), ("Noise-aware (σ=0.1)", NOISE_MODEL)]:
        p = C.OUTPUT_DIR / fname
        if p.exists():
            mdl = tf.keras.models.load_model(p, compile=False)
            out[name] = eval_model(mdl, Xs, ys, cmask, a_idx, n_classes, steps, repeats)
        else:
            print(f"[跳过] 未找到 {fname}")

    json.dump({"config": {"adv_eps": ADV_EPS, "adv_steps": ADV_STEPS, "clean_ratio": CLEAN_RATIO,
                          "n_eval": int(N)}, "models": out},
              open(C.OUTPUT_DIR / "advtrain_compare.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)


    def rep_noise(m, s): return m["noise"][m["std_grid"].index(s)]
    def rep_pgd(m, e): return m["pgd"][m["eps_grid"].index(e)]
    L = [f"AT vs noise-aware vs standard (Edge-IIoTset, binary) | n_eval={N}",
         f"AT config: eps={ADV_EPS}, steps={ADV_STEPS}, clean_ratio={CLEAN_RATIO}", "",
         f"{'Model':<28}{'CleanAcc':>9}{'AtkRec':>8}{'Acc@0.1':>9}{'Acc@0.2':>9}{'PGD@0.02':>10}",
         "-" * 73]
    for name, m in out.items():
        L.append(f"{name:<28}{m['clean_acc']:>9.4f}{m['attack_recall']:>8.3f}"
                 f"{rep_noise(m,0.1):>9.3f}{rep_noise(m,0.2):>9.3f}{rep_pgd(m,0.02):>10.3f}")
    txt = "\n".join(L)
    open(C.OUTPUT_DIR / "advtrain_table.txt", "w", encoding="utf-8").write(txt)
    print("\n" + txt)


    palette = {"Adversarial training (PGD)": ("#8c6bb1", "D"),
               "Noise-aware (σ=0.1)": ("#4C72B0", "^"),
               "Standard": ("#c1969b", "o")}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for name, m in out.items():
        col, mk = palette.get(name, ("#7F7F7F", "s"))
        a1.plot(m["std_grid"], m["noise"], marker=mk, color=col, lw=2, label=name)
        a2.plot(m["eps_grid"], m["pgd"], marker=mk, color=col, lw=2, label=name)
    a1.set_xlabel("Test Gaussian noise std"); a1.set_ylabel("Accuracy"); a1.set_ylim(0, 1)
    a1.grid(alpha=0.25); a1.legend(); a1.set_title("Robustness to Gaussian noise", fontweight="bold")
    a2.set_xlabel("PGD budget epsilon"); a2.set_ylabel("Accuracy"); a2.set_ylim(0, 1)
    a2.grid(alpha=0.25); a2.legend(); a2.set_title("Robustness to PGD", fontweight="bold")
    fig.tight_layout(); fig.savefig(C.OUTPUT_DIR / "fig_advtrain_compare.png", dpi=300, bbox_inches="tight")
    print("\nsaved: advtrain_compare.json, fig_advtrain_compare.png, advtrain_table.txt")


if __name__ == "__main__":
    main()
