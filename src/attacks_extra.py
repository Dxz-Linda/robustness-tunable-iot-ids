"""Adaptive attack and direct manipulation of attack indicator fields.

Indicator fields are ranked on the training split by how much more often they
are non-zero under attack than under normal traffic. The saliency ranking is
also run as a negative control.

Produces: attacks_extra.json, attacks_extra_table.txt
Paper: Table 14 and Section 4.6.8.
"""
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

import config as C
from data_preprocessing import preprocess
from attacks import pgd as pgd_plain

STD_MODEL = "model_edge_iiot_binary.keras"
NOISE_MODEL = "model_edge_iiot_binary_noise.keras"
TRAIN_SIGMA = 0.1

EPS = 0.02
PGD_STEPS = getattr(C, "PGD_STEPS", 7)
N_EOT = 20
N_EVAL = 2000
FLIP_K = [1, 3, 5, 10]
ZERO_FRAC_SPARSE = 0.7
EPSZ = 1e-9


def resolve_feature_names(data):
    try:
        import pandas as pd
        cfg = C.DATASETS[C.ACTIVE_DATASET]
        cols = list(pd.read_csv(cfg["csv_path"], nrows=1, low_memory=False).columns)
        drop = set(cfg["drop_cols"]) | {cfg["label_binary"], cfg["label_multiclass"]}
        kept = [c for c in cols if c not in drop]
        idx = data.get("selected_idx")
        if idx is None:
            return None
        names = [kept[j] for j in idx]
        if len(names) != data["X_test"].shape[1]:
            print("      [提示] 列名还原长度对不上，退回特征编号。")
            return None
        return names
    except Exception as e:
        print("      [提示] 列名还原失败，退回特征编号: %s" % e)
        return None


def acc_of(model, X, y):
    p = model(tf.convert_to_tensor(X, tf.float32), training=False).numpy()
    return float(accuracy_score(y, np.argmax(p, axis=1)))


def eot_pgd(model, X, yoh, eps, mask, sigma, steps=PGD_STEPS, n_eot=N_EOT):
    Xo = tf.convert_to_tensor(X, tf.float32)
    m = tf.constant(mask.astype("float32"))
    alpha = (2.5 * eps / steps) if steps > 0 else 0.0
    Xadv = tf.identity(Xo)
    lo = tf.keras.losses.CategoricalCrossentropy()
    yoh_t = tf.convert_to_tensor(yoh, tf.float32)
    for _ in range(steps):
        acc_g = tf.zeros_like(Xadv)
        for _ in range(n_eot):
            noise = tf.random.normal(tf.shape(Xadv), stddev=sigma) * m
            with tf.GradientTape() as t:
                t.watch(Xadv)
                xi = tf.clip_by_value(Xadv + noise, 0.0, 1.0)
                loss = lo(yoh_t, model(xi, training=False))
            acc_g += t.gradient(loss, Xadv)
        Xadv = Xadv + alpha * tf.sign(acc_g / n_eot) * m
        p = tf.clip_by_value(Xadv - Xo, -eps, eps)
        Xadv = tf.clip_by_value(Xo + p, 0.0, 1.0)
    return Xadv.numpy()


def saliency_attack_class(model, X, a_idx):
    Xt = tf.convert_to_tensor(X, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(Xt)
        prob = model(Xt, training=False)
        atk = prob[:, a_idx]
    g = tape.gradient(atk, Xt)
    return tf.reduce_mean(tf.abs(g), axis=0).numpy()


def indicator_score(X, y, a_idx, is_sparse):
    active = X > EPSZ
    m_atk = (y == a_idx)
    if m_atk.sum() == 0 or (~m_atk).sum() == 0:
        raise ValueError("攻击类标签 %r 在数据里找不到正负两类" % a_idx)
    p_atk = active[m_atk].mean(axis=0)
    p_nrm = active[~m_atk].mean(axis=0)
    score = (p_atk - p_nrm).astype(np.float64)
    score[~is_sparse] = -np.inf
    score[X.std(axis=0) < EPSZ] = -np.inf
    return score, p_atk, p_nrm


def flip_values(X, y, a_idx, order):
    vals = []
    m_atk = (y == a_idx)
    for j in order:
        v = X[m_atk & (X[:, j] > EPSZ), j]
        vals.append(float(np.median(v)) if v.size else 1.0)
    return np.clip(np.asarray(vals, dtype=np.float64), 0.0, 1.0)


def flip_up(X, order, vals, k):
    Xa = X.copy()
    n_changed = 0
    for j, v in zip(order[:k], vals[:k]):
        rows = Xa[:, j] <= EPSZ
        n_changed += int(rows.sum())
        Xa[rows, j] = v
    return Xa, n_changed


def flip_down(X, order, k):
    Xa = X.copy()
    n_changed = 0
    for j in order[:k]:
        rows = Xa[:, j] > EPSZ
        n_changed += int(rows.sum())
        Xa[rows, j] = 0.0
    return Xa, n_changed


def run_flip_suite(mdl, X, y, a_idx, order, vals, tag, verbose=True):
    out = {"overall_up": {}, "on_normal_up": {}, "overall_down": {}, "on_attack_down": {}}
    normal_rows = (y != a_idx)
    attack_rows = ~normal_rows

    for k in FLIP_K:
        Xa, n1 = flip_up(X, order, vals, k)
        if n1 == 0:
            raise AssertionError("[%s] k=%d 的抬升翻转没有改动任何取值，攻击是空的" % (tag, k))
        out["overall_up"][str(k)] = acc_of(mdl, Xa, y)
        out["on_normal_up"][str(k)] = acc_of(mdl, Xa[normal_rows], y[normal_rows])

        Xb, n2 = flip_down(X, order, k)
        out["overall_down"][str(k)] = acc_of(mdl, Xb, y)
        out["on_attack_down"][str(k)] = (acc_of(mdl, Xb[attack_rows], y[attack_rows])
                                         if attack_rows.sum() else float("nan"))
        if verbose:
            print("    k=%-3d 抬升改动 %6d 个取值 | overall_up=%.4f on_normal_up=%.4f "
                  "|| 压零改动 %6d 个取值 | overall_down=%.4f on_attack_down=%.4f"
                  % (k, n1, out["overall_up"][str(k)], out["on_normal_up"][str(k)],
                     n2, out["overall_down"][str(k)], out["on_attack_down"][str(k)]))
    return out


def main():
    np.random.seed(C.RANDOM_STATE)
    tf.random.set_seed(C.RANDOM_STATE)
    data = preprocess()
    Xtr, ytr = data["X_train"], data["y_train"]
    Xte, yte = data["X_test"], data["y_test"]
    cmask = data["continuous_mask"]
    le = data["label_encoder"]
    classes = [str(c) for c in le.classes_]
    a_idx = classes.index("1") if "1" in classes else int(np.argmax(classes))
    n_classes = len(np.unique(ytr))
    print(">>> 类别顺序=%s，攻击类列号 a_idx=%d" % (classes, a_idx))

    rng = np.random.RandomState(C.RANDOM_STATE)
    Nb = min(getattr(C, "ROBUST_SAMPLE", 5000), len(Xte))
    subb = rng.choice(len(Xte), Nb, replace=False)
    Xb, yb = Xte[subb], yte[subb]
    Ne = min(N_EVAL, len(Xte))
    sube = rng.choice(len(Xte), Ne, replace=False)
    Xe, ye = Xte[sube], yte[sube]
    yohe = to_categorical(ye, n_classes)
    print(">>> EOT 子集 %d | 翻转攻击子集 %d | 连续特征 %d/%d"
          % (Ne, Nb, int(cmask.sum()), len(cmask)))

    models = {}
    for name, fname in [("Standard", STD_MODEL), ("Noise-aware (sigma=0.1)", NOISE_MODEL)]:
        p = C.OUTPUT_DIR / fname
        if not p.exists():
            print("[跳过] 找不到 %s" % fname)
            continue
        mdl = tf.keras.models.load_model(p, compile=False)
        if mdl.input_shape[-1] != Xte.shape[1]:
            raise ValueError("%s 输入维 %d != 数据 %d，请把 K_FEATURES 设回 1000。"
                             % (fname, mdl.input_shape[-1], Xte.shape[1]))
        models[name] = mdl
    if "Standard" not in models:
        raise SystemExit("必须有标准模型才能计算显著性排序。")


    zero_frac_tr = (Xtr <= EPSZ).mean(axis=0)
    is_sparse = cmask & (zero_frac_tr >= ZERO_FRAC_SPARSE)
    print(">>> 稀疏近二值特征 %d 个（按训练集零值比例 >= %.1f）"
          % (int(is_sparse.sum()), ZERO_FRAC_SPARSE))


    score, p_atk, p_nrm = indicator_score(Xtr, ytr, a_idx, is_sparse)
    valid = np.flatnonzero(np.isfinite(score))
    order_ind = valid[np.argsort(-score[valid])]
    fn = resolve_feature_names(data)
    print("\n>>> 排序一，数据驱动的攻击指示位，按 P(active|attack) - P(active|normal) 降序")
    print("    %-5s %-26s %9s %12s %12s" % ("idx", "feature", "score", "P(act|atk)", "P(act|nrm)"))
    for j in order_ind[:max(FLIP_K)]:
        nm = str(fn[j])[:26] if fn is not None else ""
        print("    %-5d %-26s %9.3f %12.3f %12.3f" % (j, nm, score[j], p_atk[j], p_nrm[j]))
    weak = int((score[order_ind[:max(FLIP_K)]] < 0.05).sum())
    if weak:
        print("    提示：其中 %d 个的指示强度低于 0.05，说明可用的强指示位不足 %d 个，"
              "正文要如实说明这一点。" % (weak, max(FLIP_K)))
    vals_ind = flip_values(Xtr, ytr, a_idx, order_ind)


    idx3000 = rng.choice(len(Xte), min(3000, len(Xte)), replace=False)
    sal = saliency_attack_class(models["Standard"], Xte[idx3000], a_idx)
    order_sal = np.array([int(j) for j in np.argsort(sal)[::-1] if is_sparse[j]])
    vals_sal = flip_values(Xtr, ytr, a_idx, order_sal)
    print("\n>>> 排序二，标准模型显著性排序，前 10: %s" % order_sal[:10].tolist())
    print("    这一套只作对照。若它选出的特征在训练数据里对标签几乎没有指示性，")
    print("    翻转它们不产生效果就属于预期之内，而不是模型鲁棒的证据。")

    ORDERINGS = [("indicator", order_ind, vals_ind), ("saliency", order_sal, vals_sal)]


    results = {}
    for name, mdl in models.items():
        print("\n================ %s ================" % name)
        entry = {"clean_on_flip_subset": acc_of(mdl, Xb, yb),
                 "clean_on_eot_subset": acc_of(mdl, Xe, ye)}
        print("  clean on flip subset (%d) = %.4f" % (Nb, entry["clean_on_flip_subset"]))
        print("  clean on eot  subset (%d) = %.4f" % (Ne, entry["clean_on_eot_subset"]))

        entry["pgd_7step"] = acc_of(mdl, pgd_plain(mdl, Xe, yohe, EPS, cmask), ye)
        print("  PGD %d-step (eps=%s)  = %.4f" % (PGD_STEPS, EPS, entry["pgd_7step"]))

        entry["eot_pgd"] = acc_of(mdl, eot_pgd(mdl, Xe, yohe, EPS, cmask, TRAIN_SIGMA), ye)
        print("  EOT-PGD (%d draws)    = %.4f" % (N_EOT, entry["eot_pgd"]))

        entry["flip"] = {}
        for tag, order, vals in ORDERINGS:
            print("  --- 稀疏翻转，排序 = %s ---" % tag)
            entry["flip"][tag] = run_flip_suite(mdl, Xb, yb, a_idx, order, vals, tag)
        results[name] = entry


    cfg = {"eps": EPS, "pgd_steps": PGD_STEPS, "n_eot": N_EOT,
           "train_sigma": TRAIN_SIGMA, "flip_k": FLIP_K,
           "n_eot_eval": Ne, "n_flip_eval": Nb, "attack_class_index": a_idx,
           "zero_frac_sparse": ZERO_FRAC_SPARSE,
           "n_sparse": int(is_sparse.sum()),
           "order_indicator_top10": order_ind[:10].tolist(),
           "order_saliency_top10": order_sal[:10].tolist(),
           "indicator_scores_top10": [float(score[j]) for j in order_ind[:10]],
           "flip_values_top10": [float(v) for v in vals_ind[:10]],
           "feature_names": (list(fn) if fn is not None else None),
           "order_indicator_names_top10": ([str(fn[j]) for j in order_ind[:10]]
                                           if fn is not None else None),
           "order_saliency_names_top10": ([str(fn[j]) for j in order_sal[:10]]
                                          if fn is not None else None)}
    json.dump({"config": cfg, "models": results},
              open(C.OUTPUT_DIR / "attacks_extra.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    names = list(results)
    L = ["Table 10 自适应攻击与特征操纵攻击 (Edge-IIoTset, binary), eps=%s" % EPS, "",
         "口径说明：",
         "  clean 分两行，因为 PGD 与 EOT 用 %d 条子集，翻转攻击用 %d 条子集，" % (Ne, Nb),
         "  同一列必须与同一个 clean 对照，否则会出现攻击后准确率高于干净准确率的假象。",
         "  overall_up     在全部样本上把指示位的零值抬到攻击流量的典型取值，报整体准确率；",
         "  on_normal_up   只在正常样本上抬升，报正常样本仍被判正常的比例，即假阳性攻击；",
         "  overall_down   在全部样本上把指示位压回零，报整体准确率；",
         "  on_attack_down 只在攻击样本上压零，报攻击样本仍被判为攻击的比例，即逃逸攻击。",
         "  排序 indicator 为数据驱动，正文主用；排序 saliency 为上一版口径，只作对照。", "",
         "%-52s" % "Attack" + "".join("%26s" % n for n in names),
         "-" * (52 + 26 * len(names))]

    rows = [("Clean, flip subset (%d)" % Nb, lambda e: e["clean_on_flip_subset"]),
            ("Clean, EOT subset (%d)" % Ne, lambda e: e["clean_on_eot_subset"]),
            ("PGD, %d steps" % PGD_STEPS, lambda e: e["pgd_7step"]),
            ("EOT-PGD, %d draws at sigma=%s" % (N_EOT, TRAIN_SIGMA), lambda e: e["eot_pgd"])]
    for tag, _, _ in ORDERINGS:
        for field, label in [("overall_up", "overall, raise"),
                             ("on_normal_up", "on normal, raise"),
                             ("overall_down", "overall, zero out"),
                             ("on_attack_down", "on attack, zero out")]:
            for k in FLIP_K:
                rows.append(("Flip [%s] %s, %d feature(s)" % (tag, label, k),
                             (lambda tg, fd, kk: (lambda e: e["flip"][tg][fd][str(kk)]))(tag, field, k)))
    for label, getter in rows:
        L.append("%-52s" % label + "".join("%26.4f" % getter(results[n]) for n in names))

    L += ["", "读法：",
          "  EOT-PGD 与七步 PGD 在加噪模型上应当接近，表示攻击者知道防御细节也无优势。",
          "  排序 indicator 下若标准模型的 on_normal_up 明显下降而加噪模型基本不动，",
          "  就直接验证了 Section 4.6.1 的机制：脆弱性由零值稀疏指示位被激活所承载。",
          "  排序 indicator 下若 on_attack_down 大幅下降，说明攻击者只要抹掉少数几个字段",
          "  就能逃逸，这是比假阳性更值得写进正文的现实威胁。",
          "  若排序 indicator 下两个模型都几乎不动，而排序 saliency 下同样不动，",
          "  则结论是深度模型的脆弱性并非由单个指示位承载，而是由多个特征上的",
          "  微小改动累加承载。此时不要硬套原机制，改用 margin_radius 的裕度与梯度口径，",
          "  并把本实验作为一个阴性对照如实报出，它同样是对机制的有价值约束。"]
    txt = "\n".join(L)
    open(C.OUTPUT_DIR / "attacks_extra_table.txt", "w", encoding="utf-8").write(txt)
    print("\n" + txt)
    print("\nsaved: attacks_extra.json, attacks_extra_table.txt")


if __name__ == "__main__":
    main()
