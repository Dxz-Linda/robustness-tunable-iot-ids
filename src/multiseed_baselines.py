"""Proposed model and four baselines over five seeds, with resume support.

Produces: multiseed_baselines.json, multiseed_table.txt
Paper: Table 5, and the paired comparisons in the first rows of Table 7.
"""
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import config as C
import model as M
import baselines as B
from data_preprocessing import preprocess

SEEDS = [42, 7, 123, 2024, 999]
CKPT = "multiseed_checkpoint.json"
METRICS = ["accuracy", "macro_f1", "mcc", "macro_precision", "macro_recall"]


def set_all_seeds(s):
    C.RANDOM_STATE = s
    np.random.seed(s)
    tf.random.set_seed(s)


def train_ours(data, n_classes):
    tf.keras.backend.clear_session()
    M.TRAIN_NOISE_STD = 0.0
    mdl = M.build_model(data["X_train"].shape[1], n_classes)
    n_params = int(mdl.count_params())
    ytr = to_categorical(data["y_train"], n_classes)
    yva = to_categorical(data["y_val"], n_classes)
    es = EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=getattr(C, "REDUCE_LR_FACTOR", 0.5),
                            patience=getattr(C, "REDUCE_LR_PATIENCE", 4), min_lr=1e-6, verbose=0)
    mdl.fit(data["X_train"], ytr, validation_data=(data["X_val"], yva),
            epochs=C.EPOCHS, batch_size=C.BATCH_SIZE, callbacks=[es, rlr], verbose=2)
    t0 = time.time()
    y_prob = mdl.predict(data["X_test"], batch_size=C.BATCH_SIZE, verbose=0)
    infer_ms = (time.time() - t0) / len(data["X_test"]) * 1000
    y_pred = np.argmax(y_prob, axis=1)
    return B.metrics_from_preds(data["y_test"], y_pred, infer_ms, n_params)


def run_one_seed(seed):
    set_all_seeds(seed)
    C.LOSS_TYPE = "ce"; C.K_FEATURES = 1000
    print(f"\n############ SEED = {seed} ############")
    data = preprocess()
    n_classes = len(np.unique(data["y_train"]))
    res = {}
    res["Ours (CNN-BiLSTM-Attn)"] = train_ours(data, n_classes)
    res["CNN"]                    = B.train_and_eval_keras(B.build_cnn, "CNN", data, n_classes)
    res["BiLSTM"]                 = B.train_and_eval_keras(B.build_bilstm, "BiLSTM", data, n_classes)
    res["CNN+BiLSTM (NoAttn)"]    = B.train_and_eval_keras(B.build_cnn_bilstm, "CNN+BiLSTM (NoAttn)", data, n_classes)
    res["Random Forest"]          = B.train_and_eval_rf(data)
    return res


def load_ckpt():
    p = C.OUTPUT_DIR / CKPT
    if p.exists():
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            print(f"[警告] {CKPT} 读不动,忽略并从头跑。")
    return {}


def save_ckpt(d):
    json.dump(d, open(C.OUTPUT_DIR / CKPT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def aggregate(all_runs):
    models = list(next(iter(all_runs.values())).keys())
    agg = {}
    for mdl in models:
        agg[mdl] = {}
        for met in METRICS:
            vals = [all_runs[s][mdl][met] for s in all_runs]
            agg[mdl][met] = {"mean": float(np.mean(vals)),
                             "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                             "values": [float(v) for v in vals]}
    return agg, models


def write_table(agg, models, seeds, path):
    order = [m for m in ["Ours (CNN-BiLSTM-Attn)", "CNN+BiLSTM (NoAttn)", "CNN", "BiLSTM", "Random Forest"]
             if m in models]
    L = [f"Multi-seed comparison over seeds={seeds}  (mean +/- sample std, ddof=1)",
         "Edge-IIoTset, binary | CE loss | all 49 features | standard (no noise)", ""]
    hdr = f"{'Model':<26}{'Accuracy':>20}{'Macro-F1':>20}{'MCC':>20}"
    L += [hdr, "-" * len(hdr)]
    for m in order:
        a, f, c = agg[m]["accuracy"], agg[m]["macro_f1"], agg[m]["mcc"]
        L.append(f"{m:<26}"
                 f"{a['mean']:>12.4f}+/-{a['std']:.4f}"
                 f"{f['mean']:>12.4f}+/-{f['std']:.4f}"
                 f"{c['mean']:>12.4f}+/-{c['std']:.4f}")
    if "Ours (CNN-BiLSTM-Attn)" in agg and "CNN+BiLSTM (NoAttn)" in agg:
        L += ["", "Attention gain (Ours - NoAttn), paired per seed:"]
        for met in ["accuracy", "mcc"]:
            o = agg["Ours (CNN-BiLSTM-Attn)"][met]["values"]
            n = agg["CNN+BiLSTM (NoAttn)"][met]["values"]
            diffs = [a - b for a, b in zip(o, n)]
            sd = np.std(diffs, ddof=1) if len(diffs) > 1 else 0.0
            L.append(f"  d_{met}: mean={np.mean(diffs):+.4f}  std={sd:.4f}  per-seed={[round(x,4) for x in diffs]}")
        L += ["",
              "解读:若 d_accuracy 的 mean 明显大于 std(例如 mean/std >= 2),注意力增益稳健、不是波动;",
              "若相当,则 §4.4 改成'注意力带来小幅但方向一致的提升',或考虑加宽 key_dim/heads。"]
    txt = "\n".join(L)
    open(path, "w", encoding="utf-8").write(txt)
    print("\n" + txt)


def main():
    all_runs = load_ckpt()
    done = [int(s) for s in all_runs.keys()]
    if done:
        print(f">>> 续跑:checkpoint 里已有种子 {sorted(done)},将跳过。删掉 {CKPT} 可强制全部重跑。")

    for s in SEEDS:
        if str(s) in all_runs:
            print(f"\n[skip] seed {s} 已完成,跳过。")
            continue
        all_runs[str(s)] = run_one_seed(s)
        save_ckpt(all_runs)

    runs_used = {str(s): all_runs[str(s)] for s in SEEDS if str(s) in all_runs}
    agg, models = aggregate(runs_used)
    json.dump({"seeds": SEEDS, "per_seed": runs_used, "aggregate": agg},
              open(C.OUTPUT_DIR / "multiseed_baselines.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    write_table(agg, models, SEEDS, C.OUTPUT_DIR / "multiseed_table.txt")
    print("\nsaved: multiseed_checkpoint.json, multiseed_baselines.json, multiseed_table.txt")


if __name__ == "__main__":
    main()
