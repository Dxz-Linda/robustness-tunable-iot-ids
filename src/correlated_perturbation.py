"""Re-run the attack under perturbation models that respect feature correlations.

Compares independent perturbation against a correlation-grouped attacker and
against one confined to the principal subspace of the continuous block.

Produces: correlated_perturbation.json, correlated_perturbation_table.txt
Paper: Table 13 and Section 4.6.7.
"""
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import config as C
from data_preprocessing import preprocess
from attacks import pgd as pgd_independent

STD_MODEL = "model_edge_iiot_binary.keras"
NOISE_MODEL = "model_edge_iiot_binary_noise.keras"

EPS_LIST = [0.02]
PCA_VAR = 0.95
GROUP_CORR = 0.7
STEPS = getattr(C, "PGD_STEPS", 7)
SUB_RESTARTS = 5


def acc_of(model, X, y):
    p = model(tf.convert_to_tensor(X, tf.float32), training=False).numpy()
    return float(accuracy_score(y, np.argmax(p, axis=1)))


def perturb_subspace(model, X, y, eps, cont_idx, pca, restarts=SUB_RESTARTS, rng=None):
    rng = rng or np.random.RandomState(C.RANDOM_STATE)
    n = len(X)
    robust = np.ones(n, dtype=bool)
    comps = pca.components_
    for _ in range(restarts):
        z = rng.normal(size=(n, comps.shape[0]))
        d = z @ comps
        scale = np.abs(d).max(axis=1, keepdims=True)
        d = d / np.maximum(scale, 1e-12) * eps
        Xp = X.copy()
        Xp[:, cont_idx] = np.clip(X[:, cont_idx] + d, 0.0, 1.0)
        pred = np.argmax(model(tf.convert_to_tensor(Xp, tf.float32),
                               training=False).numpy(), axis=1)
        robust &= (pred == y)
    return float(robust.mean())


def build_groups(Xc, thr=GROUP_CORR):
    with np.errstate(invalid="ignore", divide="ignore"):
        Cm = np.corrcoef(Xc, rowvar=False)
    Cm = np.nan_to_num(Cm, nan=0.0)
    np.fill_diagonal(Cm, 0.0)
    adj = csr_matrix((np.abs(Cm) > thr).astype(int))
    n_comp, labels = connected_components(adj, directed=False)
    return [np.where(labels == k)[0] for k in range(n_comp)], Cm


def pgd_grouped(model, X, yoh, eps, cont_idx, groups_local, steps=STEPS):
    Xo = X.astype("float32")
    Xadv = Xo.copy()
    alpha = (2.5 * eps / steps) if steps > 0 else 0.0
    lo = tf.keras.losses.CategoricalCrossentropy()
    yoh_t = tf.convert_to_tensor(yoh, tf.float32)
    for _ in range(steps):
        xt = tf.convert_to_tensor(Xadv, tf.float32)
        with tf.GradientTape() as t:
            t.watch(xt)
            loss = lo(yoh_t, model(xt, training=False))
        g = t.gradient(loss, xt).numpy()
        step = np.zeros_like(Xadv)
        for grp in groups_local:
            cols = cont_idx[grp]
            s = np.sign(g[:, cols].sum(axis=1, keepdims=True))
            step[:, cols] = alpha * s
        Xadv = np.clip(Xadv + step, Xo - eps, Xo + eps)
        Xadv = np.clip(Xadv, 0.0, 1.0)
    return Xadv


def main():
    np.random.seed(C.RANDOM_STATE)
    tf.random.set_seed(C.RANDOM_STATE)
    data = preprocess()
    Xtr = data["X_train"]
    Xte, yte = data["X_test"], data["y_test"]
    cmask = data["continuous_mask"]
    cont_idx = np.where(cmask)[0]
    n_classes = len(np.unique(data["y_train"]))

    N = min(getattr(C, "ROBUST_SAMPLE", 5000), len(Xte))
    sub = np.random.RandomState(C.RANDOM_STATE).choice(len(Xte), N, replace=False)
    Xs, ys = Xte[sub], yte[sub]
    yoh = to_categorical(ys, n_classes)
    print(f">>> 子集 {N} | 连续特征 {len(cont_idx)}/{len(cmask)}")


    Xc_tr = Xtr[:, cont_idx]
    groups_local, Cm = build_groups(Xc_tr)
    iu = np.triu_indices_from(Cm, k=1)
    absc = np.abs(Cm[iu])
    mean_abs = float(absc.mean())
    frac_05 = float((absc > 0.5).mean())
    frac_07 = float((absc > GROUP_CORR).mean())
    n_groups = len(groups_local)
    n_multi = sum(1 for g in groups_local if len(g) > 1)
    print(f">>> 连续特征相关性: 平均 |corr| = {mean_abs:.4f} | "
          f"|corr|>0.5 的比例 = {frac_05*100:.1f}% | |corr|>{GROUP_CORR} 的比例 = {frac_07*100:.1f}%")
    print(f">>> 分组结果: {n_groups} 组，其中 {n_multi} 组含多于一个特征 "
          f"（攻击者自由度从 {len(cont_idx)} 降到 {n_groups}）")


    pca = PCA(n_components=PCA_VAR, svd_solver="full").fit(Xc_tr)
    print(f">>> 主子空间维度 = {pca.n_components_} / {len(cont_idx)} "
          f"（保留 {PCA_VAR*100:.0f}% 方差）")


    models = {}
    for name, fname in [("Standard", STD_MODEL), ("Noise-aware (sigma=0.1)", NOISE_MODEL)]:
        p = C.OUTPUT_DIR / fname
        if not p.exists():
            print(f"[跳过] 找不到 {fname}")
            continue
        mdl = tf.keras.models.load_model(p, compile=False)
        if mdl.input_shape[-1] != Xs.shape[1]:
            raise ValueError(f"{fname} 输入维 {mdl.input_shape[-1]} != 数据 {Xs.shape[1]}，"
                             f"请把 K_FEATURES 设回 1000。")
        models[name] = mdl

    results = {}
    for name, mdl in models.items():
        print(f"\n================ {name} ================")
        entry = {"clean": acc_of(mdl, Xs, ys)}
        print(f"  clean = {entry['clean']:.4f}")
        for eps in EPS_LIST:
            a_ind = float(accuracy_score(
                ys, np.argmax(mdl(tf.convert_to_tensor(
                    pgd_independent(mdl, Xs, yoh, eps, cmask), tf.float32),
                    training=False).numpy(), axis=1)))
            a_sub = perturb_subspace(mdl, Xs, ys, eps, cont_idx, pca)
            Xg = pgd_grouped(mdl, Xs, yoh, eps, cont_idx, groups_local)
            a_grp = acc_of(mdl, Xg, ys)
            entry[str(eps)] = {"independent": a_ind, "subspace": a_sub, "grouped": a_grp}
            print(f"  eps={eps}: independent={a_ind:.4f} | subspace={a_sub:.4f} | grouped={a_grp:.4f}")
        results[name] = entry

    out = {"correlation_stats": {"mean_abs_corr": mean_abs,
                                 "frac_abs_corr_above_0.5": frac_05,
                                 f"frac_abs_corr_above_{GROUP_CORR}": frac_07,
                                 "n_continuous": int(len(cont_idx)),
                                 "n_groups": int(n_groups),
                                 "n_multi_feature_groups": int(n_multi),
                                 "pca_dim": int(pca.n_components_),
                                 "pca_variance_kept": PCA_VAR},
           "eps_list": EPS_LIST, "n_eval": int(N), "models": results}
    json.dump(out, open(C.OUTPUT_DIR / "correlated_perturbation.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    L = ["尊重相关结构的扰动模型 (Edge-IIoTset, binary)", "",
         "Section 3.7 需要的相关性统计：",
         f"  连续特征之间的平均绝对相关系数 : {mean_abs:.4f}        <-- 正文填这个",
         f"  绝对相关系数超过 0.5 的特征对比例 : {frac_05*100:.1f}%   <-- 正文填这个", "",
         "Table 14 用的数字（攻击者自由度 / 各模型准确率）：", ""]
    for eps in EPS_LIST:
        L += [f"  预算 eps = {eps}",
              f"    {'Perturbation model':<24}{'DoF':>6}" +
              "".join(f"{n:>26}" for n in results)]
        for key, dof in [("independent", len(cont_idx)),
                         ("subspace", pca.n_components_),
                         ("grouped", n_groups)]:
            row = f"    {key:<24}{dof:>6}"
            for n in results:
                row += f"{results[n][str(eps)][key]:>26.4f}"
            L.append(row)
        L.append("")
    L += ["读法：若 standard 一行在 subspace 与 grouped 下仍然远低于 clean，",
          "就说明崩塌不是独立扰动这一不现实假设的产物，可以照 Section 4.6.8 的措辞写；",
          "若 grouped 下 standard 明显回升，则在正文如实写明分组约束显著削弱了攻击者，",
          "并把独立扰动结果明确定位成攻击者能力的上界。"]
    txt = "\n".join(L)
    open(C.OUTPUT_DIR / "correlated_perturbation_table.txt", "w", encoding="utf-8").write(txt)
    print("\n" + txt)
    print("\nsaved: correlated_perturbation.json, correlated_perturbation_table.txt")


if __name__ == "__main__":
    main()
