"""Train and evaluate the proposed detector on the current configuration.

Produces: model_<RUN_TAG>.keras, metrics_<RUN_TAG>.json, confusion matrix, report
Paper: Table 3, Figure 2, and the confusion matrices in the Supplementary Material.
"""
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (classification_report, confusion_matrix,
                             matthews_corrcoef, accuracy_score,
                             precision_recall_fscore_support)
import matplotlib.pyplot as plt
import seaborn as sns

import config as C
from data_preprocessing import preprocess
from model import build_model


def plot_history(history, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history.history["loss"], label="train")
    ax[0].plot(history.history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history.history["accuracy"], label="train")
    ax[1].plot(history.history["val_accuracy"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_confusion(cm, classes, path):
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.title(f"Confusion Matrix ({C.RUN_TAG})")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def evaluate(model, X_test, y_test, classes, tag):
    t0 = time.time()
    y_prob = model.predict(X_test, batch_size=C.BATCH_SIZE, verbose=0)
    infer_time = (time.time() - t0) / len(X_test) * 1000
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    report = classification_report(y_test, y_pred, target_names=classes,
                                   zero_division=0, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== [{tag}] 评估结果 =====")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Macro-Precision: {prec:.4f}")
    print(f"Macro-Recall  : {rec:.4f}")
    print(f"Macro-F1      : {f1:.4f}")
    print(f"MCC           : {mcc:.4f}")
    print(f"每样本推理时延 : {infer_time:.4f} ms")
    print("\n--- 逐类报告 ---\n", report)

    plot_confusion(cm, classes, C.OUTPUT_DIR / f"confusion_{tag}.png")
    with open(C.OUTPUT_DIR / f"report_{tag}.txt", "w", encoding="utf-8") as f:
        f.write(report)

    return {"accuracy": acc, "macro_precision": prec, "macro_recall": rec,
            "macro_f1": f1, "mcc": mcc, "infer_ms_per_sample": infer_time}


def main():
    np.random.seed(C.RANDOM_STATE)
    tf.random.set_seed(C.RANDOM_STATE)

    print(f"\n>>> 运行配置: 数据集={C.ACTIVE_DATASET} | 模式={C.MODE} | "
          f"特征上限={C.K_FEATURES} | 损失={C.LOSS_TYPE}")
    print(f">>> 输出标签(RUN_TAG): {C.RUN_TAG}\n")

    data = preprocess()
    classes = [str(c) for c in data["label_encoder"].classes_]
    n_classes = len(classes)
    n_features = data["X_train"].shape[1]

    y_train_oh = to_categorical(data["y_train"], n_classes)
    y_val_oh = to_categorical(data["y_val"], n_classes)


    sample_weight = None
    if getattr(C, "USE_CLASS_WEIGHT", False):
        cw = compute_class_weight("balanced",
                                  classes=np.unique(data["y_train"]),
                                  y=data["y_train"])
        cw_map = {c: w for c, w in zip(np.unique(data["y_train"]), cw)}
        sample_weight = np.array([cw_map[y] for y in data["y_train"]], dtype="float32")
        print(f">>> 已启用类别加权 class_weight={{ {', '.join(f'{int(c)}:{w:.2f}' for c,w in cw_map.items())} }}")

    model = build_model(n_features, n_classes)
    model.summary()
    n_params = model.count_params()

    es = EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE,
                       restore_best_weights=True)

    rlr = ReduceLROnPlateau(monitor="val_loss",
                            factor=getattr(C, "REDUCE_LR_FACTOR", 0.5),
                            patience=getattr(C, "REDUCE_LR_PATIENCE", 4),
                            min_lr=1e-6, verbose=1)

    history = model.fit(
        data["X_train"], y_train_oh,
        validation_data=(data["X_val"], y_val_oh),
        sample_weight=sample_weight,
        epochs=C.EPOCHS, batch_size=C.BATCH_SIZE,
        callbacks=[es, rlr], verbose=2)

    plot_history(history, C.OUTPUT_DIR / f"training_history_{C.RUN_TAG}.png")

    metrics = evaluate(model, data["X_test"], data["y_test"], classes, tag=C.RUN_TAG)
    metrics["n_params"] = int(n_params)
    metrics["model_size_MB"] = round(n_params * 4 / (1024 ** 2), 3)
    metrics["mode"] = C.MODE
    metrics["dataset"] = C.ACTIVE_DATASET
    metrics["n_features"] = int(n_features)

    print(f"\n[轻量化] 参数量={n_params:,} | 估算模型大小={metrics['model_size_MB']} MB")

    model.save(C.OUTPUT_DIR / f"model_{C.RUN_TAG}.keras")
    with open(C.OUTPUT_DIR / f"metrics_{C.RUN_TAG}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n全部结果已存到 {C.OUTPUT_DIR.resolve()}(文件名带 {C.RUN_TAG})")


if __name__ == "__main__":
    main()
