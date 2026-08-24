"""Recover attack recall by lowering the decision threshold, without losing robustness.

The threshold is selected on the validation split and reported on the test
split, so the operating point is not tuned to the test set.

Produces: threshold_sweep.json and two figures
Paper: Figures 14 and 15.
"""
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import config as C
from data_preprocessing import preprocess
from attacks import pgd as pgd_attack

NOISE_MODEL = "model_edge_iiot_binary_noise.keras"
TARGET_RECALL = 0.72
THRESHOLDS = np.round(np.arange(0.02, 0.99, 0.01), 3)
EPS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
STD = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]


def attack_index(label_encoder):
    classes = [str(c) for c in label_encoder.classes_]
    idx = classes.index("1") if "1" in classes else int(np.argmax(classes))
    print(f"[检查] 类别顺序={classes} -> 认定攻击类列号 attack_idx={idx}(请确认无误)")
    return idx


def prob_attack(model, X, a_idx):
    return model.predict(X, batch_size=C.BATCH_SIZE, verbose=0)[:, a_idx]


def attack_metrics(y_bin, p_attack, thr):
    y_pred = (p_attack >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_bin, y_pred, average="binary", pos_label=1, zero_division=0)
    return {"thr": float(thr), "precision": float(prec), "recall": float(rec),
            "f1": float(f1), "accuracy": float(accuracy_score(y_bin, y_pred))}


def sweep(y_bin, p_attack):
    return [attack_metrics(y_bin, p_attack, float(t)) for t in THRESHOLDS]


def select_threshold_on_val(rows_val, target_recall):
    cand = [r for r in rows_val if r["recall"] >= target_recall]
    if cand:
        return max(cand, key=lambda r: r["thr"])
    print(f"[提示] 验证集上没有阈值能达到召回 {target_recall};改选召回最高的阈值。")
    return max(rows_val, key=lambda r: r["recall"])


def robustness_at_threshold(model, X, y_enc, a_idx, mask, thr, n_classes, steps, repeats):
    y_bin = (y_enc == a_idx).astype(int)
    y_oh = to_categorical(y_enc, n_classes)
    noise_acc = []
    for s in STD:
        accs = []
        for _ in range(max(1, repeats)):
            nz = np.random.normal(0, s, X.shape); nz[:, ~mask] = 0.0
            Xn = np.clip(X + nz, 0.0, 1.0)
            accs.append(accuracy_score(y_bin, (prob_attack(model, Xn, a_idx) >= thr).astype(int)))
        noise_acc.append(float(np.mean(accs)))
    pgd_acc = []
    for e in EPS:
        Xadv = pgd_attack(model, X, y_oh, e, mask, steps)
        pgd_acc.append(float(accuracy_score(y_bin, (prob_attack(model, Xadv, a_idx) >= thr).astype(int))))
    return noise_acc, pgd_acc


def main():
    np.random.seed(C.RANDOM_STATE); tf.random.set_seed(C.RANDOM_STATE)
    steps = getattr(C, "PGD_STEPS", 7); repeats = getattr(C, "NOISE_REPEATS", 3)

    data = preprocess()
    le = data["label_encoder"]
    n_classes = len(le.classes_)
    if n_classes != 2:
        raise SystemExit(f"本脚本只针对二分类;当前类别数={n_classes}。请把 config 设为 MODE='binary'。")
    a_idx = attack_index(le)
    cmask = data["continuous_mask"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    y_val_bin = (y_val == a_idx).astype(int)
    y_test_bin = (y_test == a_idx).astype(int)

    noise = tf.keras.models.load_model(C.OUTPUT_DIR / NOISE_MODEL, compile=False)


    p_val = prob_attack(noise, X_val, a_idx)
    rows_val = sweep(y_val_bin, p_val)
    tstar_row_val = select_threshold_on_val(rows_val, TARGET_RECALL)
    tstar = tstar_row_val["thr"]
    val_050 = attack_metrics(y_val_bin, p_val, 0.5)


    p_test = prob_attack(noise, X_test, a_idx)
    rows_test = sweep(y_test_bin, p_test)
    test_tstar = attack_metrics(y_test_bin, p_test, tstar)
    test_050 = attack_metrics(y_test_bin, p_test, 0.5)

    print("\n=== 阈值在验证集上选、在测试集上报(Edge-IIoTset, binary, noise-aware σ=0.1) ===")
    print(f"[验证集] 选出 t*={tstar:.2f}: 召回={tstar_row_val['recall']:.3f} 精度={tstar_row_val['precision']:.3f} "
          f"F1={tstar_row_val['f1']:.3f}  (目标召回 ≥ {TARGET_RECALL})")
    print(f"[测试集] 默认 thr=0.50: 召回={test_050['recall']:.3f} 精度={test_050['precision']:.3f} "
          f"F1={test_050['f1']:.3f} 准确率={test_050['accuracy']:.3f}")
    print(f"[测试集] 固定 t*={tstar:.2f}: 召回={test_tstar['recall']:.3f} 精度={test_tstar['precision']:.3f} "
          f"F1={test_tstar['f1']:.3f} 准确率={test_tstar['accuracy']:.3f}")
    print(f"  -> 泛化检查:验证召回 {tstar_row_val['recall']:.3f} vs 测试召回 {test_tstar['recall']:.3f}"
          f"(两者接近 = 阈值未过拟合)")


    N = min(getattr(C, "ROBUST_SAMPLE", 5000), len(X_test))
    sub = np.random.RandomState(C.RANDOM_STATE).choice(len(X_test), N, replace=False)
    Xs, y_enc_s = X_test[sub], y_test[sub]
    print(f"\n[鲁棒性] 在 {N} 个随机测试样本上,分别用默认阈值 0.50 与 t*={tstar:.2f} 评估...")
    noise_def, pgd_def = robustness_at_threshold(noise, Xs, y_enc_s, a_idx, cmask, 0.5, n_classes, steps, repeats)
    noise_tgt, pgd_tgt = robustness_at_threshold(noise, Xs, y_enc_s, a_idx, cmask, tstar, n_classes, steps, repeats)

    json.dump({"attack_idx": a_idx, "target_recall": TARGET_RECALL,
               "selected_on": "validation", "t_star": tstar,
               "val_sweep": rows_val, "val_at_tstar": tstar_row_val, "val_at_0.5": val_050,
               "test_sweep": rows_test, "test_at_tstar": test_tstar, "test_at_0.5": test_050,
               "robustness": {"std_grid": STD, "eps_grid": EPS,
                              "default_threshold": {"thr": 0.5, "noise_acc": noise_def, "pgd_acc": pgd_def},
                              "target_threshold": {"thr": tstar, "noise_acc": noise_tgt, "pgd_acc": pgd_tgt}}},
              open(C.OUTPUT_DIR / "threshold_sweep.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)


    thr_a = [r["thr"] for r in rows_test]; pr = [r["precision"] for r in rows_test]
    rc = [r["recall"] for r in rows_test]; f1 = [r["f1"] for r in rows_test]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax1.plot(thr_a, pr, color="#4C72B0", label="Attack precision")
    ax1.plot(thr_a, rc, color="#C44E52", label="Attack recall")
    ax1.plot(thr_a, f1, color="#55A868", label="Attack F1")
    ax1.axvline(0.5, ls=":", color="gray"); ax1.axvline(tstar, ls="--", color="black")
    ax1.text(tstar, 0.03, f" t*={tstar:.2f} (val-selected)", rotation=90, va="bottom", fontsize=8)
    ax1.set_xlabel("Decision threshold on P(attack)  [test set]"); ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.02); ax1.grid(alpha=0.25); ax1.legend(loc="lower center")
    ax1.set_title("Threshold vs. attack precision / recall / F1 (test)", fontweight="bold")
    ax2.plot(rc, pr, color="#4C72B0")
    ax2.scatter([test_050["recall"]], [test_050["precision"]], color="gray", zorder=5, label="default (thr=0.5)")
    ax2.scatter([test_tstar["recall"]], [test_tstar["precision"]], color="black", zorder=5,
                label=f"t*={tstar:.2f} (val-selected)")
    ax2.set_xlabel("Attack recall"); ax2.set_ylabel("Attack precision")
    ax2.set_xlim(0, 1.02); ax2.set_ylim(0, 1.02); ax2.grid(alpha=0.25); ax2.legend(loc="lower left")
    ax2.set_title("Precision-recall trade-off (noise-aware, σ=0.1, test)", fontweight="bold")
    fig.tight_layout(); fig.savefig(C.OUTPUT_DIR / "fig_threshold_pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


    fig, (b1, b2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    b1.plot(STD, noise_def, marker="o", color="#7F7F7F", label="default thr=0.50")
    b1.plot(STD, noise_tgt, marker="^", color="#2C7BB6", label=f"t*={tstar:.2f} (val-selected)")
    b1.set_xlabel("Test Gaussian noise std"); b1.set_ylabel("Accuracy"); b1.set_ylim(0, 1)
    b1.grid(alpha=0.25); b1.legend(); b1.set_title("Robustness to Gaussian noise at two thresholds", fontweight="bold")
    b2.plot(EPS, pgd_def, marker="o", color="#7F7F7F", label="default thr=0.50")
    b2.plot(EPS, pgd_tgt, marker="^", color="#2C7BB6", label=f"t*={tstar:.2f} (val-selected)")
    b2.set_xlabel("PGD budget epsilon"); b2.set_ylabel("Accuracy"); b2.set_ylim(0, 1)
    b2.grid(alpha=0.25); b2.legend(); b2.set_title("Robustness to PGD at two thresholds", fontweight="bold")
    fig.tight_layout(); fig.savefig(C.OUTPUT_DIR / "fig_threshold_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nsaved: threshold_sweep.json, fig_threshold_pr_curve.png, fig_threshold_robustness.png")


if __name__ == "__main__":
    main()
