"""Rule out gradient masking with stronger, transfer and gradient-free attacks.

The evaluated model is deterministic because the noise layer is removed at
inference, so the relevant diagnostics are stronger white-box attacks together
with black-box attacks that never touch a gradient.

Produces: robustness_adaptive.json, fig_adaptive_attacks.png
Paper: Figure 12.
"""
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import config as C
from data_preprocessing import preprocess

STD_MODEL = "model_edge_iiot_binary.keras"
NOISE_MODEL = "model_edge_iiot_binary_noise.keras"

REP_EPS = 0.02
EPS_LIST = [0.02, 0.05]
SANITY_EPS = 0.30

N_WHITEBOX = getattr(C, "ROBUST_SAMPLE", 5000)
N_BLACKBOX = 1000
PGD7_STEPS = getattr(C, "PGD_STEPS", 7)
STRONG_STEPS, STRONG_RESTARTS = 50, 10
TRANSFER_STEPS, TRANSFER_RESTARTS = 50, 5
SPSA_ITERS, SPSA_K, SPSA_C = 40, 8, 0.01
SQUARE_ITERS, SQUARE_SUBSET = 300, 2


def _probs(model, X):
    return model(tf.convert_to_tensor(X, tf.float32), training=False).numpy()


def _acc(model, X, y):
    return float((np.argmax(_probs(model, X), axis=1) == y).mean())


def _margin_loss(model, X, y):
    P = _probs(model, X); n = len(y); ar = np.arange(n)
    tp = P[ar, y]; P2 = P.copy(); P2[ar, y] = -1.0
    return P2.max(axis=1) - tp


def _pgd_core(model, x0, yoh, eps, mask, steps, alpha, rand_init):
    Xo = tf.convert_to_tensor(x0, tf.float32)
    m = tf.constant(mask.astype("float32"))
    if rand_init:
        d0 = tf.random.uniform(Xo.shape, -eps, eps) * m
        Xadv = tf.clip_by_value(Xo + d0, 0.0, 1.0)
    else:
        Xadv = tf.identity(Xo)
    lo = tf.keras.losses.CategoricalCrossentropy()
    for _ in range(steps):
        with tf.GradientTape() as t:
            t.watch(Xadv)
            loss = lo(yoh, model(Xadv, training=False))
        g = t.gradient(loss, Xadv)
        Xadv = Xadv + alpha * tf.sign(g) * m
        p = tf.clip_by_value(Xadv - Xo, -eps, eps)
        Xadv = tf.clip_by_value(Xo + p, 0.0, 1.0)
    return Xadv.numpy()


def pgd_weak_acc(model, X, y, yoh, eps, mask, steps=PGD7_STEPS):
    a = (2.5 * eps / steps) if steps > 0 else 0.0
    xadv = _pgd_core(model, X, yoh, eps, mask, steps, a, rand_init=False)
    return float((np.argmax(_probs(model, xadv), 1) == y).mean())


def pgd_strong_acc(model, X, y, yoh, eps, mask, steps=STRONG_STEPS, restarts=STRONG_RESTARTS):
    robust = np.ones(len(y), bool)
    a = (2.5 * eps / steps) if steps > 0 else 0.0
    for r in range(restarts):
        xadv = _pgd_core(model, X, yoh, eps, mask, steps, a, rand_init=(r > 0))
        robust &= (np.argmax(_probs(model, xadv), 1) == y)
    return float(robust.mean())


def transfer_acc(src, tgt, X, y, yoh, eps, mask, steps=TRANSFER_STEPS, restarts=TRANSFER_RESTARTS):
    robust = np.ones(len(y), bool)
    a = (2.5 * eps / steps) if steps > 0 else 0.0
    for r in range(restarts):
        xadv = _pgd_core(src, X, yoh, eps, mask, steps, a, rand_init=(r > 0))
        robust &= (np.argmax(_probs(tgt, xadv), 1) == y)
    return float(robust.mean())


def spsa_acc(model, X, y, eps, mask, iters=SPSA_ITERS, K=SPSA_K, c=SPSA_C, alpha=None):
    n, d = X.shape
    x0 = X.astype("float32")
    m = mask.astype("float32")[None, :]
    if alpha is None:
        alpha = 2.5 * eps / iters
    delta = np.zeros_like(x0)
    xadv = np.clip(x0 + delta, 0.0, 1.0)
    for _ in range(iters):
        U = np.random.choice([-1.0, 1.0], size=(K, n, d)).astype("float32") * m
        xp = np.clip(x0[None] + delta[None] + c * U, 0.0, 1.0)
        xm = np.clip(x0[None] + delta[None] - c * U, 0.0, 1.0)
        Lp = _margin_loss(model, xp.reshape(K * n, d), np.tile(y, K)).reshape(K, n)
        Lm = _margin_loss(model, xm.reshape(K * n, d), np.tile(y, K)).reshape(K, n)
        g = (((Lp - Lm) / (2 * c))[:, :, None] * U).mean(axis=0)
        delta = np.clip(delta + alpha * np.sign(g) * m, -eps, eps)
        xadv = np.clip(x0 + delta, 0.0, 1.0)
        delta = xadv - x0
    return float((np.argmax(_probs(model, xadv), 1) == y).mean())


def square_acc(model, X, y, eps, mask, iters=SQUARE_ITERS, subset=SQUARE_SUBSET):
    n, d = X.shape
    x0 = X.astype("float32")
    cont_idx = np.where(mask)[0]
    if len(cont_idx) == 0:
        return float((np.argmax(_probs(model, x0), 1) == y).mean())
    delta = np.zeros_like(x0)
    delta[:, cont_idx] = np.random.choice([-eps, eps], size=(n, len(cont_idx))).astype("float32")
    xadv = np.clip(x0 + delta, 0.0, 1.0); delta = xadv - x0
    best = _margin_loss(model, xadv, y)
    for _ in range(iters):
        cand = delta.copy()
        js = np.random.choice(cont_idx, size=min(subset, len(cont_idx)), replace=False)
        cand[:, js] = np.random.choice([-eps, eps], size=(n, len(js))).astype("float32")
        xc = np.clip(x0 + cand, 0.0, 1.0); cand = xc - x0
        L = _margin_loss(model, xc, y)
        imp = L > best
        delta[imp] = cand[imp]; best[imp] = L[imp]
    xadv = np.clip(x0 + delta, 0.0, 1.0)
    return float((np.argmax(_probs(model, xadv), 1) == y).mean())


def main():
    np.random.seed(C.RANDOM_STATE); tf.random.set_seed(C.RANDOM_STATE)
    data = preprocess()
    n_classes = len(np.unique(data["y_train"]))
    Xte, yte, cmask = data["X_test"], data["y_test"], data["continuous_mask"]
    from tensorflow.keras.utils import to_categorical

    std = tf.keras.models.load_model(C.OUTPUT_DIR / STD_MODEL, compile=False)
    noi = tf.keras.models.load_model(C.OUTPUT_DIR / NOISE_MODEL, compile=False)
    for m, nm in [(std, "standard"), (noi, "noise-aware")]:
        if m.input_shape[-1] != Xte.shape[1]:
            raise ValueError(f"{nm}: 模型维度 {m.input_shape[-1]} != 数据 {Xte.shape[1]};把 K_FEATURES 设回 1000。")

    rng = np.random.RandomState(C.RANDOM_STATE)
    Nw = min(N_WHITEBOX, len(Xte)); subw = rng.choice(len(Xte), Nw, replace=False)
    Xw, yw = Xte[subw], yte[subw]; yohw = to_categorical(yw, n_classes)
    Nb = min(N_BLACKBOX, len(Xte)); subb = rng.choice(len(Xte), Nb, replace=False)
    Xb, yb = Xte[subb], yte[subb]; yohb = to_categorical(yb, n_classes)
    print(f">>> 白盒子集 {Nw} | 黑盒子集 {Nb} | 连续特征 {int(cmask.sum())}/{len(cmask)}")

    results = {"models": {}, "config": {
        "rep_eps": REP_EPS, "eps_list": EPS_LIST, "sanity_eps": SANITY_EPS,
        "strong_pgd": [STRONG_STEPS, STRONG_RESTARTS], "spsa": [SPSA_ITERS, SPSA_K, SPSA_C],
        "square": [SQUARE_ITERS, SQUARE_SUBSET], "n_whitebox": Nw, "n_blackbox": Nb}}

    for name, model, src in [("standard", std, noi), ("noise-aware", noi, std)]:
        print(f"\n==================  {name}  ==================")
        clean = _acc(model, Xw, yw)
        per_eps = {}
        for eps in EPS_LIST:
            print(f"  -- eps = {eps} --")
            pw = pgd_weak_acc(model, Xw, yw, yohw, eps, cmask)
            ps = pgd_strong_acc(model, Xw, yw, yohw, eps, cmask)
            tr = transfer_acc(src, model, Xw, yw, yohw, eps, cmask)
            sp = spsa_acc(model, Xb, yb, eps, cmask)
            sq = square_acc(model, Xb, yb, eps, cmask)
            worst = min(pw, ps, tr, sp, sq)
            per_eps[str(eps)] = {"pgd_weak": pw, "pgd_strong": ps, "transfer": tr,
                                 "spsa": sp, "square": sq, "worst_case": worst}
            print(f"     PGD7={pw:.4f} | PGD50x10={ps:.4f} | transfer={tr:.4f} "
                  f"| SPSA={sp:.4f} | Square={sq:.4f} | WORST={worst:.4f}")

        s_ps = pgd_strong_acc(model, Xw, yw, yohw, SANITY_EPS, cmask, steps=100, restarts=5)
        s_sq = square_acc(model, Xb, yb, SANITY_EPS, cmask, iters=500)
        print(f"  [健全性 eps={SANITY_EPS}] PGD100x5={s_ps:.4f} | Square={s_sq:.4f}  (应接近多数类基线)")
        results["models"][name] = {"clean": clean, "per_eps": per_eps,
                                   "sanity": {"eps": SANITY_EPS, "pgd_strong": s_ps, "square": s_sq}}

    json.dump(results, open(C.OUTPUT_DIR / "robustness_adaptive.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)


    attacks = ["pgd_weak", "pgd_strong", "transfer", "spsa", "square"]
    labels = ["PGD (7-step)", "PGD (50×10)", "Transfer", "SPSA (black-box)", "Random search (black-box)"]
    x = np.arange(len(attacks)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    vs = [results["models"]["standard"]["per_eps"][str(REP_EPS)][a] for a in attacks]
    vn = [results["models"]["noise-aware"]["per_eps"][str(REP_EPS)][a] for a in attacks]
    ax.bar(x - w / 2, vs, w, color="#c1969b", edgecolor="#5a5a5a", lw=0.4, label="Standard")
    ax.bar(x + w / 2, vn, w, color="#6f8fb3", edgecolor="#5a5a5a", lw=0.4, label="Noise-aware (σ=0.1)")
    for xi, v in zip(x - w / 2, vs):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    for xi, v in zip(x + w / 2, vn):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.axhspan(0, 0, color="none")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(f"Accuracy under attack (ε = {REP_EPS})"); ax.set_ylim(0, 1.0); ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper right")
    ax.set_title("White-box, transfer, and gradient-free attacks (Edge-IIoTset, binary)", fontweight="bold")
    fig.tight_layout(); fig.savefig(C.OUTPUT_DIR / "fig_adaptive_attacks.png", dpi=300, bbox_inches="tight")
    print("\nsaved robustness_adaptive.json + fig_adaptive_attacks.png  —— 把这两个发我")


if __name__ == "__main__":
    main()
