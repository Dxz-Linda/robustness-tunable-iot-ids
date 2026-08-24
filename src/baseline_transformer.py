"""Parameter-matched compact transformer encoder trained under the same protocol.

Requires: nothing beyond the datasets
Produces: transformer_baseline.json, transformer_baseline_table.txt
Paper: the transformer row of Table 5 and all of Section 4.3.1.
"""
import json
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

import config as C
from data_preprocessing import preprocess
from attacks import pgd

SEEDS = [42, 7, 123, 2024, 999]
NOISE_SEEDS = [42, 7, 123]
RUN_NOISE_AWARE = True


DATA_SEED = None
EVAL_SEED = 42
TARGET_PARAMS = 199234
N_BLOCKS = 2
N_HEADS = 4
TRAIN_SIGMA = 0.1
CKPT = "transformer_baseline_ckpt.json"
NOISE_EVAL = 0.1
PGD_EVAL = 0.02


class AddPositionEmbedding(layers.Layer):

    def __init__(self, length, depth, **kwargs):
        super().__init__(**kwargs)
        self.length = int(length)
        self.depth = int(depth)

    def build(self, input_shape):
        self.pos = self.add_weight(
            name="position_embedding",
            shape=(1, self.length, self.depth),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True)
        super().build(input_shape)

    def call(self, x):
        return x + self.pos

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"length": self.length, "depth": self.depth})
        return cfg


def build_transformer(n_features, n_classes, d_model, d_ff,
                      n_blocks=N_BLOCKS, n_heads=N_HEADS, drop=None, noise_std=0.0):
    drop = C.DROPOUT if drop is None else drop
    inp = layers.Input(shape=(n_features,), name="input_features")
    x = layers.Reshape((n_features, 1))(inp)
    if noise_std and noise_std > 0:
        x = layers.GaussianNoise(noise_std, name="train_noise")(x)
    x = layers.Dense(d_model, name="token_embedding")(x)
    x = AddPositionEmbedding(n_features, d_model, name="position_embedding")(x)

    for i in range(n_blocks):
        a = layers.MultiHeadAttention(num_heads=n_heads,
                                      key_dim=max(d_model // n_heads, 1),
                                      name="mhsa_%d" % i)(x, x)
        x = layers.LayerNormalization(name="ln_a_%d" % i)(
            layers.Add()([x, layers.Dropout(drop)(a)]))
        f = layers.Dense(d_ff, activation="relu")(x)
        f = layers.Dense(d_model)(f)
        x = layers.LayerNormalization(name="ln_f_%d" % i)(
            layers.Add()([x, layers.Dropout(drop)(f)]))

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(getattr(C, "DENSE_UNITS", 64), activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)
    return models.Model(inp, out, name="Compact_Transformer")


def pick_config(n_features, n_classes, target=TARGET_PARAMS, require_wide_ffn=True):
    best = None
    for d_model in [32, 48, 64, 80, 96, 112, 128]:
        if d_model % N_HEADS != 0:
            continue
        for d_ff in [64, 96, 128, 192, 256, 320, 384]:
            if require_wide_ffn and d_ff < d_model:
                continue
            tf.keras.backend.clear_session()
            m = build_transformer(n_features, n_classes, d_model, d_ff)
            n = m.count_params()
            gap = abs(n - target)
            if best is None or gap < best[0]:
                best = (gap, d_model, d_ff, n)
    _, d_model, d_ff, n = best
    print(">>> 选中配置: d_model=%d, d_ff=%d, blocks=%d, heads=%d -> %s 参数（目标 %s，差 %+d）"
          % (d_model, d_ff, N_BLOCKS, N_HEADS, "{:,}".format(n),
             "{:,}".format(target), n - target))
    tf.keras.backend.clear_session()
    return d_model, d_ff, n


def prepare_data(seed_for_data):
    C.RANDOM_STATE = seed_for_data
    C.LOSS_TYPE = "ce"
    C.K_FEATURES = 1000
    np.random.seed(seed_for_data)
    tf.random.set_seed(seed_for_data)
    return preprocess()


def run_seed(seed, data, n_classes, cmask, sub_idx, d_model, d_ff, noise_std):
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    tf.random.set_seed(seed)

    n_features = data["X_train"].shape[1]
    mdl = build_transformer(n_features, n_classes, d_model, d_ff, noise_std=noise_std)
    mdl.compile(optimizer=optimizers.Adam(learning_rate=C.LEARNING_RATE),
                loss="categorical_crossentropy", metrics=["accuracy"])
    ytr = to_categorical(data["y_train"], n_classes)
    yva = to_categorical(data["y_val"], n_classes)
    es = EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE,
                       restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                            min_lr=1e-6, verbose=0)
    mdl.fit(data["X_train"], ytr, validation_data=(data["X_val"], yva),
            epochs=C.EPOCHS, batch_size=C.BATCH_SIZE, callbacks=[es, rlr], verbose=2)

    t0 = time.time()
    yprob = mdl.predict(data["X_test"], batch_size=C.BATCH_SIZE, verbose=0)
    batched_ms = (time.time() - t0) / len(data["X_test"]) * 1000
    yp = np.argmax(yprob, axis=1)
    res = {"accuracy": float(accuracy_score(data["y_test"], yp)),
           "macro_f1": float(f1_score(data["y_test"], yp, average="macro", zero_division=0)),
           "mcc": float(matthews_corrcoef(data["y_test"], yp)),
           "batched_ms_per_sample": float(batched_ms),
           "n_params": int(mdl.count_params())}

    Xs, ys = data["X_test"][sub_idx], data["y_test"][sub_idx]
    yoh = to_categorical(ys, n_classes)
    rng = np.random.RandomState(EVAL_SEED)
    accs = []
    for _ in range(getattr(C, "NOISE_REPEATS", 3)):
        nz = rng.normal(0, NOISE_EVAL, Xs.shape)
        nz[:, ~cmask] = 0.0
        Xn = np.clip(Xs + nz, 0.0, 1.0)
        accs.append(accuracy_score(ys, np.argmax(mdl.predict(Xn, verbose=0), axis=1)))
    res["noise_%s" % NOISE_EVAL] = float(np.mean(accs))
    Xadv = pgd(mdl, Xs, yoh, PGD_EVAL, cmask)
    res["pgd_%s" % PGD_EVAL] = float(
        accuracy_score(ys, np.argmax(mdl.predict(Xadv, verbose=0), axis=1)))

    print("   acc=%.4f f1=%.4f mcc=%.4f noise@%.1f=%.4f PGD@%.2f=%.4f"
          % (res["accuracy"], res["macro_f1"], res["mcc"],
             NOISE_EVAL, res["noise_%s" % NOISE_EVAL],
             PGD_EVAL, res["pgd_%s" % PGD_EVAL]))
    return res


def selftest():
    print(">>> 结构自检")
    m = build_transformer(49, 2, 112, 192)
    m.summary()
    y = m(tf.zeros((8, 49)))
    print("    前向输出形状:", tuple(y.shape), " 应为 (8, 2)")
    assert tuple(y.shape) == (8, 2)
    names = [l.name for l in m.layers]
    assert "position_embedding" in names, "位置嵌入层没有进入模型图"
    npos = int(np.prod(m.get_layer("position_embedding").pos.shape))
    print("    位置嵌入参数量:", npos, " 应为 49 x 112 = 5488")
    assert npos == 49 * 112
    m2 = build_transformer(49, 2, 112, 192, noise_std=TRAIN_SIGMA)
    assert m2.count_params() == m.count_params(), "噪声层不应引入参数"
    print("    自检通过，位置嵌入形状正确且参数已被计入。")


def main():
    if "--selftest" in sys.argv:
        selftest()
        return

    fixed_data = DATA_SEED is not None
    if fixed_data:
        print(">>> 数据固定在 DATA_SEED = %d。注意这与 multiseed_baselines.py 不同口径，"
              % DATA_SEED)
        print("    这样跑出来的 standard 臂不宜直接填进 Table 4。")
        data = prepare_data(DATA_SEED)
    else:
        print(">>> DATA_SEED 为 None，每个训练种子重新抽样与划分，")
        print("    与 multiseed_baselines.py 同口径，standard 臂可以直接填进 Table 4。")
        data = prepare_data(SEEDS[0])
    n_classes = len(np.unique(data["y_train"]))
    n_features = data["X_train"].shape[1]
    cmask = data["continuous_mask"]

    d_model, d_ff, n_par = pick_config(n_features, n_classes)

    N = min(getattr(C, "ROBUST_SAMPLE", 5000), len(data["X_test"]))
    sub_idx = np.random.RandomState(EVAL_SEED).choice(len(data["X_test"]), N, replace=False)
    print(">>> 鲁棒性评估子集 %d 条，固定在 EVAL_SEED = %d" % (N, EVAL_SEED))

    arms = [("standard", 0.0, SEEDS)]
    if RUN_NOISE_AWARE:
        arms.append(("noise-aware", TRAIN_SIGMA, NOISE_SEEDS))

    p = C.OUTPUT_DIR / CKPT
    ck = json.load(open(p, encoding="utf-8")) if p.exists() else {}
    if ck:
        print(">>> 续跑：已有 %d 条记录，将跳过。若这是上一版留下的 ckpt 请先改名备份。"
              % len(ck))

    for tname, sigma, seed_list in arms:
        for s in seed_list:
            key = "%s|%s" % (s, tname)
            if key in ck:
                print("[skip] seed %s | %s" % (s, tname))
                continue
            print("\n############ SEED = %d | %s ############" % (s, tname))
            if not fixed_data:
                data = prepare_data(s)
                N = min(getattr(C, "ROBUST_SAMPLE", 5000), len(data["X_test"]))
                sub_idx = np.random.RandomState(EVAL_SEED).choice(
                    len(data["X_test"]), N, replace=False)
            ck[key] = run_seed(s, data, n_classes, cmask, sub_idx, d_model, d_ff, sigma)
            json.dump(ck, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    keys = ["accuracy", "macro_f1", "mcc", "noise_%s" % NOISE_EVAL,
            "pgd_%s" % PGD_EVAL, "batched_ms_per_sample"]
    agg = {}
    for tname, sigma, seed_list in arms:
        e = {}
        for k in keys:
            v = [ck["%s|%s" % (s, tname)][k] for s in seed_list
                 if "%s|%s" % (s, tname) in ck]
            if not v:
                continue
            e[k] = {"mean": float(np.mean(v)),
                    "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                    "values": [float(x) for x in v]}
        e["seeds"] = seed_list
        agg[tname] = e

    json.dump({"data_seed": DATA_SEED, "eval_seed": EVAL_SEED,
               "arch": {"d_model": d_model, "d_ff": d_ff, "n_blocks": N_BLOCKS,
                        "n_heads": N_HEADS, "n_params": n_par},
               "per_run": ck, "aggregate": agg},
              open(C.OUTPUT_DIR / "transformer_baseline.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    L = ["参数量匹配的紧凑 Transformer 基线 (Edge-IIoTset, binary)",
         "架构: d_model=%d, d_ff=%d, blocks=%d, heads=%d, 参数量=%s（主模型 199,234）"
         % (d_model, d_ff, N_BLOCKS, N_HEADS, "{:,}".format(n_par)),
         ("每个种子重新抽样与划分，与 multiseed_baselines.py 同口径，可直接填进 Table 4。"
          if DATA_SEED is None else
          "数据固定在 DATA_SEED=%d，与 Table 4 其它行不同口径，不要直接填进 Table 4。"
          % DATA_SEED), ""]
    for tname, sigma, seed_list in arms:
        if tname not in agg or not agg[tname].get("accuracy"):
            continue
        L.append("--- %s 臂，seeds=%s ---" % (tname, seed_list))
        for k in keys:
            if k in agg[tname]:
                L.append("  %-24s%.4f +/- %.4f"
                         % (k, agg[tname][k]["mean"], agg[tname][k]["std"]))
        L.append("")
    L += ["Table 4 新增一行用 standard 臂的 accuracy、macro_f1、mcc 与参数量。",
          "Section 4.3.1 的鲁棒性对照用两个臂的 noise 与 pgd 两列。", "",
          "读法：",
          "  若 standard 臂的准确率与 MCC 落在主模型的种子间波动范围内，",
          "  就照 Section 3.5 与 4.3.1 的拟稿写明在该输入长度与该规模下，",
          "  混合方式的选择不决定干净准确率；若 Transformer 明显更好，须如实报告，",
          "  并把主模型定位成同等精度下更省的选择。",
          "  若 standard 臂在噪声与 PGD 下同样崩塌而 noise-aware 臂同样保持高位，",
          "  则可在 Section 4.6.1 补一句：脆弱性与其消除在一个结构上完全无关的",
          "  注意力模型上复现，进一步支持它属于任务而不属于某个模型族。"]
    txt = "\n".join(L)
    open(C.OUTPUT_DIR / "transformer_baseline_table.txt", "w", encoding="utf-8").write(txt)
    print("\n" + txt)
    print("\nsaved: transformer_baseline_ckpt.json, transformer_baseline.json, "
          "transformer_baseline_table.txt")


if __name__ == "__main__":
    main()
