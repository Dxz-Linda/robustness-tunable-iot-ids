"""Four architecture variants under both training regimes, five seeds each.

Answers whether any architectural component contributes robustness rather than
only clean accuracy.

Produces: ablation_robustness.json, ablation_robustness_table.txt
Paper: Table 6, and the paired tests fed to paired_tests_robust.py.
"""
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score

import config as C
from data_preprocessing import preprocess
from attacks import pgd

SEEDS = [42, 7, 123, 2024, 999]


DATA_SEED = None
EVAL_SEED = 42
NOISE_EVAL = 0.1
PGD_EVAL = 0.02
TRAIN_SIGMA = 0.1
CKPT = "ablation_robustness_ckpt.json"


MODEL_CACHE_DIR = C.OUTPUT_DIR / "ablation_models"
MODEL_NAME_TEMPLATE = "ab_{tag}_{training}_s{seed}.keras"
VARIANT_TAG = {
    "BiLSTM only": "BiLSTMonly",
    "CNN only": "CNNonly",
    "CNN+BiLSTM (no attn.)": "CNNBiLSTMnoattn",
    "Full (+attention)": "Fullattention",
}


ONLY_SEEDS = None
ONLY_VARIANTS = None


TABLE4_CLEAN = {
    "BiLSTM only": (0.8996, 0.0115),
    "CNN only": (0.9139, 0.0020),
    "CNN+BiLSTM (no attn.)": (0.9215, 0.0022),
    "Full (+attention)": (0.9227, 0.0006),
}


EXPECTED_PARAMS = {
    "BiLSTM only": 42178,
    "CNN only": 34114,
    "CNN+BiLSTM (no attn.)": 132930,
    "Full (+attention)": 199234,
}


def _head(x, n_classes, drop):
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(getattr(C, "DENSE_UNITS", 64), activation="relu")(x)
    x = layers.Dropout(drop)(x)
    return layers.Dense(n_classes, activation="softmax", name="output")(x)


def _stem(n_features, noise_std):
    inp = layers.Input(shape=(n_features,), name="input_features")
    x = layers.Reshape((n_features, 1))(inp)
    if noise_std and noise_std > 0:
        x = layers.GaussianNoise(noise_std, name="train_noise")(x)
    return inp, x


def _conv_trunk(x, drop):
    x = layers.Conv1D(64, 3, padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(128, 3, padding="same", use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(drop)(x)
    return x


def build_bilstm_only(n_features, n_classes, noise_std=0.0, drop=None):
    drop = getattr(C, "DROPOUT", 0.3) if drop is None else drop
    inp, x = _stem(n_features, noise_std)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    return models.Model(inp, _head(x, n_classes, drop), name="BiLSTM_only")


def build_cnn_only(n_features, n_classes, noise_std=0.0, drop=None):
    drop = getattr(C, "DROPOUT", 0.3) if drop is None else drop
    inp, x = _stem(n_features, noise_std)
    x = _conv_trunk(x, drop)
    return models.Model(inp, _head(x, n_classes, drop), name="CNN_only")


def build_cnn_bilstm(n_features, n_classes, noise_std=0.0, drop=None):
    drop = getattr(C, "DROPOUT", 0.3) if drop is None else drop
    inp, x = _stem(n_features, noise_std)
    x = _conv_trunk(x, drop)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    return models.Model(inp, _head(x, n_classes, drop), name="CNN_BiLSTM")


def build_full(n_features, n_classes, noise_std=0.0, drop=None):
    drop = getattr(C, "DROPOUT", 0.3) if drop is None else drop
    inp, x = _stem(n_features, noise_std)
    x = _conv_trunk(x, drop)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    a = layers.MultiHeadAttention(num_heads=4, key_dim=32, name="mhsa")(x, x)
    x = layers.Add()([x, a])
    x = layers.LayerNormalization()(x)
    return models.Model(inp, _head(x, n_classes, drop), name="Full_attention")


VARIANTS = [
    ("BiLSTM only", build_bilstm_only),
    ("CNN only", build_cnn_only),
    ("CNN+BiLSTM (no attn.)", build_cnn_bilstm),
    ("Full (+attention)", build_full),
]
TRAININGS = [("standard", 0.0), ("noise-aware", TRAIN_SIGMA)]


def verify_architectures(n_features, n_classes):
    print(">>> 结构参数量校验（对照论文 Table 4）")
    bad = []
    for vname, builder in VARIANTS:
        tf.keras.backend.clear_session()
        m = builder(n_features, n_classes, noise_std=0.0)
        got, want = m.count_params(), EXPECTED_PARAMS[vname]
        flag = "ok" if got == want else "不一致，差 %+d" % (got - want)
        print("    %-24s %10s  期望 %10s  %s"
              % (vname, "{:,}".format(got), "{:,}".format(want), flag))
        if got != want:
            bad.append(vname)
    if bad:
        raise SystemExit(
            "\n参数量与论文 Table 4 不一致：%s\n"
            "请检查 config.py 里的 DENSE_UNITS 是否为 64，以及输入特征数是否为 49。\n"
            "确认无误后再跑，否则 Table 12 与 Table 4 的对照没有意义。" % ", ".join(bad))
    print("    全部一致，继续。\n")


def cached_model_path(vname, tname, seed):
    if MODEL_CACHE_DIR is None or DATA_SEED is not None:
        return None
    tag = VARIANT_TAG.get(vname)
    if tag is None:
        return None
    from pathlib import Path
    return Path(MODEL_CACHE_DIR) / MODEL_NAME_TEMPLATE.format(
        tag=tag, training=tname, seed=seed)


def train_one(builder, data, n_classes, noise_std, seed):
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.random.set_seed(seed)

    n_features = data["X_train"].shape[1]
    mdl = builder(n_features, n_classes, noise_std=noise_std)
    mdl.compile(optimizer=optimizers.Adam(learning_rate=C.LEARNING_RATE),
                loss="categorical_crossentropy", metrics=["accuracy"])
    ytr = to_categorical(data["y_train"], n_classes)
    yva = to_categorical(data["y_val"], n_classes)
    es = EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE,
                       restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                            min_lr=1e-6, verbose=0)
    mdl.fit(data["X_train"], ytr, validation_data=(data["X_val"], yva),
            epochs=C.EPOCHS, batch_size=C.BATCH_SIZE, callbacks=[es, rlr], verbose=0)
    return mdl


def evaluate(mdl, data, sub_idx, cmask, n_classes, eval_seed):
    yp = np.argmax(mdl.predict(data["X_test"], batch_size=C.BATCH_SIZE, verbose=0), axis=1)
    clean = float(accuracy_score(data["y_test"], yp))

    Xs, ys = data["X_test"][sub_idx], data["y_test"][sub_idx]
    yoh = to_categorical(ys, n_classes)

    rng = np.random.RandomState(eval_seed)
    accs = []
    for _ in range(getattr(C, "NOISE_REPEATS", 3)):
        nz = rng.normal(0, NOISE_EVAL, Xs.shape)
        nz[:, ~cmask] = 0.0
        Xn = np.clip(Xs + nz, 0.0, 1.0)
        accs.append(accuracy_score(ys, np.argmax(mdl.predict(Xn, verbose=0), axis=1)))
    noise_acc = float(np.mean(accs))

    Xadv = pgd(mdl, Xs, yoh, PGD_EVAL, cmask)
    pgd_acc = float(accuracy_score(ys, np.argmax(mdl.predict(Xadv, verbose=0), axis=1)))
    return clean, noise_acc, pgd_acc


def prepare_data(seed_for_data):
    C.RANDOM_STATE = seed_for_data
    C.LOSS_TYPE = "ce"
    C.K_FEATURES = 1000
    np.random.seed(seed_for_data)
    tf.random.set_seed(seed_for_data)
    return preprocess()


def main():
    fixed_data = DATA_SEED is not None
    prepared_for = None
    if fixed_data:
        print(">>> 数据固定在 DATA_SEED = %d，所有训练种子共用同一份抽样与划分。" % DATA_SEED)
        print("    跨种子标准差只反映训练随机性，但干净准确率不再与 Table 4 逐格可比。")
        data = prepare_data(DATA_SEED)
    else:
        print(">>> DATA_SEED 为 None，每个训练种子重新抽样与划分，")
        print("    与 multiseed_baselines.py 完全同口径，干净准确率可与 Table 4 直接对照。")
        data = prepare_data(SEEDS[0])
        prepared_for = SEEDS[0]

    n_classes = len(np.unique(data["y_train"]))
    n_features = data["X_train"].shape[1]
    cmask = data["continuous_mask"]
    print(">>> 特征数 %d（连续 %d） | train=%d val=%d test=%d"
          % (n_features, int(cmask.sum()), len(data["X_train"]),
             len(data["X_val"]), len(data["X_test"])))

    verify_architectures(n_features, n_classes)

    N = min(getattr(C, "ROBUST_SAMPLE", 5000), len(data["X_test"]))
    sub_idx = np.random.RandomState(EVAL_SEED).choice(len(data["X_test"]), N, replace=False)
    print(">>> 鲁棒性评估子集 %d 条，固定在 EVAL_SEED = %d\n" % (N, EVAL_SEED))

    p = C.OUTPUT_DIR / CKPT
    ck = json.load(open(p, encoding="utf-8")) if p.exists() else {}
    if ck:
        print(">>> 续跑：已有 %d 条记录，将跳过。若这是上一版留下的 ckpt 请先改名备份。\n"
              % len(ck))

    run_seeds = ONLY_SEEDS or SEEDS
    run_variants = [(v, b) for v, b in VARIANTS
                    if ONLY_VARIANTS is None or v in ONLY_VARIANTS]
    if ONLY_SEEDS or ONLY_VARIANTS:
        print(">>> 子集模式：种子 %s，变体 %s\n"
              % (run_seeds, [v for v, _ in run_variants]))

    for seed in run_seeds:
        print("\n############ TRAIN SEED = %d ############" % seed)
        if not fixed_data and seed != prepared_for:
            data = prepare_data(seed)
            prepared_for = seed
            N = min(getattr(C, "ROBUST_SAMPLE", 5000), len(data["X_test"]))
            sub_idx = np.random.RandomState(EVAL_SEED).choice(
                len(data["X_test"]), N, replace=False)
        for vname, builder in run_variants:
            for tname, sigma in TRAININGS:
                key = "%s|%s|%s" % (seed, vname, tname)
                if key in ck:
                    print("  [skip] %s | %s" % (vname, tname))
                    continue
                cpath = cached_model_path(vname, tname, seed)
                if cpath is not None and cpath.exists():
                    print("  [load ] %s | %s | seed %d  <- %s" % (vname, tname, seed, cpath.name))
                    mdl = tf.keras.models.load_model(cpath, compile=False)
                    if mdl.count_params() != EXPECTED_PARAMS[vname]:
                        raise SystemExit(
                            "缓存模型 %s 的参数量 %d 与预期 %d 不符，命名映射可能对错了，"
                            "请核对 VARIANT_TAG。"
                            % (cpath.name, mdl.count_params(), EXPECTED_PARAMS[vname]))
                else:
                    print("  [train] %s | %s | seed %d" % (vname, tname, seed))
                    mdl = train_one(builder, data, n_classes, sigma, seed)
                clean, noise_acc, pgd_acc = evaluate(mdl, data, sub_idx, cmask,
                                                     n_classes, EVAL_SEED)
                ck[key] = {"clean_acc": clean,
                           "noise_%s" % NOISE_EVAL: noise_acc,
                           "pgd_%s" % PGD_EVAL: pgd_acc,
                           "n_params": int(mdl.count_params())}
                print("          clean=%.4f noise@%.1f=%.4f PGD@%.2f=%.4f"
                      % (clean, NOISE_EVAL, noise_acc, PGD_EVAL, pgd_acc))
                json.dump(ck, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
                tf.keras.backend.clear_session()


    metrics = ["clean_acc", "noise_%s" % NOISE_EVAL, "pgd_%s" % PGD_EVAL]
    agg = {}
    for vname, _ in VARIANTS:
        for tname, _ in TRAININGS:
            rows = [ck["%s|%s|%s" % (s, vname, tname)] for s in SEEDS
                    if "%s|%s|%s" % (s, vname, tname) in ck]
            if not rows:
                continue
            e = {"n_params": rows[0]["n_params"]}
            for m in metrics:
                v = [r[m] for r in rows]
                e[m] = {"mean": float(np.mean(v)),
                        "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                        "values": [float(x) for x in v]}
            agg["%s|%s" % (vname, tname)] = e

    json.dump({"seeds": SEEDS, "data_seed": DATA_SEED, "eval_seed": EVAL_SEED,
               "per_run": ck, "aggregate": agg},
              open(C.OUTPUT_DIR / "ablation_robustness.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    L = ["Table 12 消融变体在扰动下的表现 (Edge-IIoTset, binary), seeds=%s" % SEEDS,
         "mean +/- sample std, ddof=1",
         "抽样与划分固定在 DATA_SEED=%d，鲁棒性子集固定在 EVAL_SEED=%d，"
         % (DATA_SEED, EVAL_SEED),
         "所以下表的标准差只反映训练随机性，不含抽样与评估随机性。", "",
         "%-24s%-17s%9s%18s%18s%18s"
         % ("Variant", "Training", "Params", "CleanAcc",
            "Acc@noise%.1f" % NOISE_EVAL, "Acc@PGD%.2f" % PGD_EVAL),
         "-" * 104]
    for vname, _ in VARIANTS:
        for tname, sigma in TRAININGS:
            k = "%s|%s" % (vname, tname)
            if k not in agg:
                continue
            e = agg[k]
            lab = "standard" if sigma == 0 else "sigma = %.1f" % sigma
            L.append("%-24s%-17s%9s%18s%18s%18s" % (
                vname, lab, "{:,}".format(e["n_params"]),
                "%.4f+/-%.4f" % (e["clean_acc"]["mean"], e["clean_acc"]["std"]),
                "%.4f+/-%.4f" % (e[metrics[1]]["mean"], e[metrics[1]]["std"]),
                "%.4f+/-%.4f" % (e[metrics[2]]["mean"], e[metrics[2]]["std"])))

    L += ["", "与论文 Table 4 干净准确率的自动对照（只对 standard 那四行）："]
    for vname, _ in VARIANTS:
        k = "%s|standard" % vname
        if k not in agg or vname not in TABLE4_CLEAN:
            continue
        got_m, got_s = agg[k]["clean_acc"]["mean"], agg[k]["clean_acc"]["std"]
        ref_m, ref_s = TABLE4_CLEAN[vname]
        d = got_m - ref_m
        verdict = "一致" if abs(d) <= 2 * max(got_s, ref_s, 1e-4) else "偏差偏大，需要解释"
        L.append("  %-24s 本表 %.4f+/-%.4f   Table 4 %.4f+/-%.4f   差 %+.4f   %s"
                 % (vname, got_m, got_s, ref_m, ref_s, d, verdict))
    L += ["",
          "口径说明：DATA_SEED=%s。若为 None，本表与 multiseed_baselines.py 完全同口径，"
          % DATA_SEED,
          "上面四行应当全部落在两倍标准差以内；若为固定值，干净准确率本就不必逐格相等，",
          "此时应在 Table 12 加脚注说明本实验固定了抽样与划分以隔离训练随机性。", "",
          "读法：",
          "  标准训练那四行若在噪声与 PGD 下全部大幅下滑，说明没有任何架构模块带来鲁棒性；",
          "  加噪训练那四行若全部保持高位且组内差异远小于两种训练之间的差距，说明训练方式压倒架构。",
          "  接着跑 paired_tests_robust.py 出配对检验，把这两条从目视升级成统计结论。"]
    txt = "\n".join(L)
    open(C.OUTPUT_DIR / "ablation_robustness_table.txt", "w", encoding="utf-8").write(txt)
    print("\n" + txt)
    print("\nsaved: ablation_robustness_ckpt.json, ablation_robustness.json, "
          "ablation_robustness_table.txt")


if __name__ == "__main__":
    main()
