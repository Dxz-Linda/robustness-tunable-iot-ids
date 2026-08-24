"""Gaussian-noise and PGD curves for one trained model.

Requires: model_<RUN_TAG>.keras from train.py, run with the same configuration
Produces: robustness_<RUN_TAG>_noise.png, robustness_<RUN_TAG>_pgd.png
Paper: the per-model curves behind Figure 4.
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import config as C
from attacks import pgd as pgd_attack

PGD_STEPS = getattr(C, "PGD_STEPS", 10)
NOISE_REPEATS = getattr(C, "NOISE_REPEATS", 5)


def load_trained_model():
    from data_preprocessing import preprocess  # noqa
    path = C.OUTPUT_DIR / f"model_{C.RUN_TAG}.keras"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}。请先用相同的 ACTIVE_DATASET / MODE / EXP_NOTE 跑一次 train.py。")
    return tf.keras.models.load_model(path, compile=False)


def gaussian_noise_test(model, X_test, y_test, std_list, cont_mask, repeats):
    results = []
    for std in std_list:
        accs = []
        for _ in range(max(1, repeats)):
            noise = np.random.normal(0, std, X_test.shape)
            noise[:, ~cont_mask] = 0.0
            X_noisy = np.clip(X_test + noise, 0, 1)
            y_pred = np.argmax(model.predict(X_noisy, verbose=0), axis=1)
            accs.append(accuracy_score(y_test, y_pred))
        avg = float(np.mean(accs))
        results.append(avg)
        print(f"  std={std:.3f} -> accuracy={avg:.4f} (平均 {repeats} 次)")
    return results


def pgd_test(model, X_test, y_test, eps_list, cont_mask, steps):
    from tensorflow.keras.utils import to_categorical
    n_classes = model.output_shape[-1]
    y_oh = to_categorical(y_test, n_classes)
    results = []
    for eps in eps_list:
        X_adv = pgd_attack(model, X_test, y_oh, eps, cont_mask, steps)
        y_pred = np.argmax(model.predict(X_adv, verbose=0), axis=1)
        acc = accuracy_score(y_test, y_pred)
        results.append(acc)
        print(f"  epsilon={eps:.3f} -> accuracy={acc:.4f} (PGD, {steps} 步)")
    return results


def plot_curve(x, y, xlabel, title, path):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel); plt.ylabel("Accuracy"); plt.title(title)
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()


def main():
    from data_preprocessing import preprocess
    np.random.seed(C.RANDOM_STATE)
    tf.random.set_seed(C.RANDOM_STATE)
    print(f">>> 鲁棒性实验 | RUN_TAG={C.RUN_TAG}")
    data = preprocess()
    model = load_trained_model()
    X_test, y_test = data["X_test"], data["y_test"]
    cont_mask = data["continuous_mask"]

    N = min(getattr(C, "ROBUST_SAMPLE", 5000), len(X_test))
    rng = np.random.RandomState(C.RANDOM_STATE)
    sub = rng.choice(len(X_test), size=N, replace=False)
    X_test, y_test = X_test[sub], y_test[sub]
    print(f">>> 鲁棒性在 {N} 个随机测试样本上评估 | "
          f"只扰动连续特征 {int(cont_mask.sum())}/{len(cont_mask)} | "
          f"PGD 步数={PGD_STEPS} | 噪声重复={NOISE_REPEATS}")

    print("\n== 高斯噪声鲁棒性(不同标准差,多次取平均) ==")
    std_list = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    noise_acc = gaussian_noise_test(model, X_test, y_test, std_list, cont_mask, NOISE_REPEATS)
    plot_curve(std_list, noise_acc, "Noise std",
               f"Robustness to Gaussian Noise ({C.RUN_TAG})",
               C.OUTPUT_DIR / f"robustness_{C.RUN_TAG}_noise.png")

    print("\n== PGD 对抗鲁棒性 ==")
    eps_list = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
    adv_acc = pgd_test(model, X_test, y_test, eps_list, cont_mask, PGD_STEPS)
    plot_curve(eps_list, adv_acc, "Perturbation epsilon",
               f"Robustness to PGD Attack ({C.RUN_TAG})",
               C.OUTPUT_DIR / f"robustness_{C.RUN_TAG}_pgd.png")

    print(f"\n鲁棒性曲线已存到 {C.OUTPUT_DIR.resolve()}(文件名带 {C.RUN_TAG})")


if __name__ == "__main__":
    main()
