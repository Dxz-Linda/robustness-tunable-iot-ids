"""Four baselines under identical preprocessing, splits and training budget.

Produces: baselines_<RUN_TAG>.json
Paper: the CNN, BiLSTM, CNN+BiLSTM and random-forest rows of Table 5.
"""
import json
import time
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             matthews_corrcoef)

import config as C
from data_preprocessing import preprocess


def build_cnn(n_features, n_classes):
    inp = layers.Input(shape=(n_features,), name="input_features")
    x = layers.Reshape((n_features, 1))(inp)
    x = layers.Conv1D(C.CONV1_FILTERS, C.KERNEL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(C.CONV2_FILTERS, C.KERNEL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(C.DROPOUT)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(C.DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(C.DROPOUT)(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)
    return models.Model(inp, out, name="Baseline_CNN")


def build_bilstm(n_features, n_classes):
    inp = layers.Input(shape=(n_features,), name="input_features")
    x = layers.Reshape((n_features, 1))(inp)
    x = layers.Bidirectional(layers.LSTM(C.LSTM_UNITS, return_sequences=True))(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(C.DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(C.DROPOUT)(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)
    return models.Model(inp, out, name="Baseline_BiLSTM")


def build_cnn_bilstm(n_features, n_classes):
    inp = layers.Input(shape=(n_features,), name="input_features")
    x = layers.Reshape((n_features, 1))(inp)
    x = layers.Conv1D(C.CONV1_FILTERS, C.KERNEL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(C.CONV2_FILTERS, C.KERNEL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(C.DROPOUT)(x)
    x = layers.Bidirectional(layers.LSTM(C.LSTM_UNITS, return_sequences=True))(x)


    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(C.DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(C.DROPOUT)(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)
    return models.Model(inp, out, name="Baseline_CNN_BiLSTM_NoAttn")


def compile_model(model):
    opt = optimizers.Adam(learning_rate=C.LEARNING_RATE)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def metrics_from_preds(y_true, y_pred, infer_ms, n_params, extra=None):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    m = {
        "accuracy": float(acc),
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "mcc": float(mcc),
        "infer_ms_per_sample": float(infer_ms),
        "n_params": (int(n_params) if n_params is not None else None),
        "model_size_MB": (round(n_params * 4 / (1024 ** 2), 3)
                          if n_params is not None else None),
    }
    if extra:
        m.update(extra)
    return m


def train_and_eval_keras(build_fn, name, data, n_classes):
    tf.keras.backend.clear_session()
    np.random.seed(C.RANDOM_STATE)
    tf.random.set_seed(C.RANDOM_STATE)

    n_features = data["X_train"].shape[1]
    model = compile_model(build_fn(n_features, n_classes))
    n_params = int(model.count_params())
    print(f"\n----- 训练基线: {name}  (参数量 {n_params:,}) -----")

    y_tr = to_categorical(data["y_train"], n_classes)
    y_va = to_categorical(data["y_val"], n_classes)

    es = EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE,
                       restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss",
                            factor=getattr(C, "REDUCE_LR_FACTOR", 0.5),
                            patience=getattr(C, "REDUCE_LR_PATIENCE", 4),
                            min_lr=1e-6, verbose=0)

    model.fit(data["X_train"], y_tr,
              validation_data=(data["X_val"], y_va),
              epochs=C.EPOCHS, batch_size=C.BATCH_SIZE,
              callbacks=[es, rlr], verbose=2)

    t0 = time.time()
    y_prob = model.predict(data["X_test"], batch_size=C.BATCH_SIZE, verbose=0)
    infer_ms = (time.time() - t0) / len(data["X_test"]) * 1000
    y_pred = np.argmax(y_prob, axis=1)
    return metrics_from_preds(data["y_test"], y_pred, infer_ms, n_params)


def train_and_eval_rf(data):
    print("\n----- 训练基线: RandomForest -----")
    clf = RandomForestClassifier(n_jobs=-1, random_state=C.RANDOM_STATE, **C.RF_PARAMS)
    clf.fit(data["X_train"], data["y_train"])

    t0 = time.time()
    y_pred = clf.predict(data["X_test"])
    infer_ms = (time.time() - t0) / len(data["X_test"]) * 1000


    n_nodes = int(sum(est.tree_.node_count for est in clf.estimators_))
    extra = {"n_estimators": int(clf.n_estimators), "n_nodes": n_nodes}
    return metrics_from_preds(data["y_test"], y_pred, infer_ms, n_params=None, extra=extra)


def _fmt_params(m):
    if m.get("n_params") is not None:
        return f"{m['n_params']:,}"
    if "n_estimators" in m:
        return f"{m['n_estimators']} trees / {m.get('n_nodes', '?')} nodes"
    return "N/A"


def print_comparison_table(results):
    rows = []

    main_path = C.OUTPUT_DIR / f"metrics_{C.RUN_TAG}.json"
    if main_path.exists():
        with open(main_path, encoding="utf-8") as f:
            ours = json.load(f)
        rows.append(("Ours (CNN-BiLSTM-Attn)", ours))
    else:
        print(f"\n[提示] 没找到主跑指标 {main_path.name},对比表只列四个基线。"
              f" 想要 Ours 那一行,请先用主配置跑一次 train.py。")

    order = ["CNN", "BiLSTM", "CNN_BiLSTM (NoAttn)", "RandomForest"]
    for name in order:
        if name in results:
            rows.append((name, results[name]))


    header = (f"{'Model':<24}{'Acc':>9}{'MacroF1':>10}{'MCC':>9}"
              f"{'Infer(ms)':>11}  {'Params/Complexity'}")
    bar = "=" * max(len(header), 60)
    line = "-" * len(header)
    print("\n" + bar)
    print(f"Baseline comparison (RUN_TAG = {C.RUN_TAG})")
    print(bar)
    print(header)
    print(line)
    for label, m in rows:
        print(f"{label:<24}"
              f"{m['accuracy']:>9.4f}"
              f"{m['macro_f1']:>10.4f}"
              f"{m['mcc']:>9.4f}"
              f"{m['infer_ms_per_sample']:>11.4f}"
              f"  {_fmt_params(m)}")
    print(line)


def main():
    np.random.seed(C.RANDOM_STATE)
    tf.random.set_seed(C.RANDOM_STATE)

    print(f">>> 基线对比 | 数据集={C.ACTIVE_DATASET} | 模式={C.MODE} | RUN_TAG={C.RUN_TAG}")
    print(">>> 基线一律用 CE(不读 LOSS_TYPE);请确认 config 为主配置"
          "(edge_iiot / binary / class weight 关 / 不加噪 / LR 1e-3)。\n")

    data = preprocess()
    classes = [str(c) for c in data["label_encoder"].classes_]
    n_classes = len(classes)

    results = {}
    results["CNN"] = train_and_eval_keras(build_cnn, "CNN", data, n_classes)
    results["BiLSTM"] = train_and_eval_keras(build_bilstm, "BiLSTM", data, n_classes)
    results["CNN_BiLSTM (NoAttn)"] = train_and_eval_keras(
        build_cnn_bilstm, "CNN_BiLSTM (NoAttn)", data, n_classes)
    results["RandomForest"] = train_and_eval_rf(data)

    out_path = C.OUTPUT_DIR / f"baselines_{C.RUN_TAG}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n四个基线指标已存到 {out_path.resolve()}")

    print_comparison_table(results)
    print("\n提示:表中 'CNN_BiLSTM (NoAttn)' 一行 = 注意力消融结果(第 4.3 节可直接引用)。")


if __name__ == "__main__":
    main()
