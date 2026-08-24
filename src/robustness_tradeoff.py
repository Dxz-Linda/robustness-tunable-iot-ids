"""Accuracy under test-time noise for the standard, noise-aware and forest models.

All models see the same noisy records at every noise level, so the comparison is
exact rather than approximate.

Produces: robustness_tradeoff.json, fig_robustness_tradeoff.png
Paper: Figure 3 and Supplementary Figure S6.
"""
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import config as C
from data_preprocessing import preprocess

NOISE_REPEATS = getattr(C, "NOISE_REPEATS", 3)
STD_LIST = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]


DEEP_MODELS = [
    ("Ours (standard)",           "model_edge_iiot_binary.keras",         ("o", "-", "#C44E52")),
    ("Ours (noise-aware σ=0.05)", "model_edge_iiot_binary_noise05.keras", ("D", "-", "#55A868")),
    ("Ours (noise-aware σ=0.1)",  "model_edge_iiot_binary_noise.keras",   ("^", "-", "#4C72B0")),
]


def compare_noise(models, X, y, cont_mask, std_list, repeats):
    curves = {name: [] for name in models}
    for std in std_list:
        per = {name: [] for name in models}
        for _ in range(max(1, repeats)):
            noise = np.random.normal(0, std, X.shape)
            noise[:, ~cont_mask] = 0.0
            X_noisy = np.clip(X + noise, 0.0, 1.0)
            for name, fn in models.items():
                per[name].append(accuracy_score(y, fn(X_noisy)))
        for name in models:
            curves[name].append(float(np.mean(per[name])))
        print("  std={:.3f} -> ".format(std)
              + " | ".join(f"{n}={curves[n][-1]:.4f}" for n in models))
    return curves


def make_deep_predict(model):
    return lambda Xb: np.argmax(model.predict(Xb, verbose=0), axis=1)


def main():
    np.random.seed(C.RANDOM_STATE)
    tf.random.set_seed(C.RANDOM_STATE)

    data = preprocess()
    X_test, y_test = data["X_test"], data["y_test"]
    cont_mask = data["continuous_mask"]

    N = min(getattr(C, "ROBUST_SAMPLE", 5000), len(X_test))
    rng = np.random.RandomState(C.RANDOM_STATE)
    sub = rng.choice(len(X_test), size=N, replace=False)
    X_sub, y_sub = X_test[sub], y_test[sub]
    print(f">>> 权衡对比 | 子集 {N} | 连续特征 {int(cont_mask.sum())}/{len(cont_mask)} | "
          f"每档重复 {NOISE_REPEATS}")

    models, styles = {}, {}


    print("\n[训练 RandomForest]")
    rf = RandomForestClassifier(n_jobs=-1, random_state=C.RANDOM_STATE, **C.RF_PARAMS)
    rf.fit(data["X_train"], data["y_train"])
    models["Random Forest"] = lambda Xb: rf.predict(Xb)
    styles["Random Forest"] = ("s", "-", "#7F7F7F")


    for name, fname, st in DEEP_MODELS:
        path = C.OUTPUT_DIR / fname
        if not path.exists():
            print(f"[跳过] 找不到 {fname} —— 略过 {name}(还没训练?)")
            continue
        mdl = tf.keras.models.load_model(path, compile=False)
        if mdl.input_shape[-1] != X_sub.shape[1]:
            print(f"[跳过] {fname} 期望 {mdl.input_shape[-1]} 维,当前 {X_sub.shape[1]} 维 "
                  f"—— 把 config 的 K_FEATURES 设回 1000 再跑")
            continue
        models[name] = make_deep_predict(mdl)
        styles[name] = st
        print(f"[已载入] {name}  <- {fname}")

    print("\n== 抗噪曲线(同一带噪样本喂所有模型) ==")
    curves = compare_noise(models, X_sub, y_sub, cont_mask, STD_LIST, NOISE_REPEATS)

    json.dump({"std_list": STD_LIST, "curves": curves, "n_eval": int(N),
               "n_continuous": int(cont_mask.sum()), "n_features": int(len(cont_mask))},
              open(C.OUTPUT_DIR / "robustness_tradeoff.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)


    order = [n for n, _, _ in DEEP_MODELS if n in curves] + ["Random Forest"]
    plt.figure(figsize=(7.8, 5.0))
    for name in order:
        mk, ls, col = styles[name]
        plt.plot(STD_LIST, curves[name], marker=mk, linestyle=ls, color=col,
                 linewidth=2.2 if name.startswith("Ours") else 1.8, label=name)
    plt.xlabel("Test-time Gaussian noise std (continuous features)")
    plt.ylabel("Accuracy")
    plt.title("Robustness vs. training-noise level (Edge-IIoTset, binary)", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="center right")
    plt.ylim(0.33, 0.97)
    plt.tight_layout()
    plt.savefig(C.OUTPUT_DIR / "fig_robustness_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("\n图已存 outputs/fig_robustness_tradeoff.png ；数值已存 outputs/robustness_tradeoff.json")


if __name__ == "__main__":
    main()
