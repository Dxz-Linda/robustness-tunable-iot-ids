"""Why the random forest collapses: importance and split thresholds.

Collects the split thresholds the ensemble places on sparse indicator features
and reports what fraction of them lie below the attack budget.

Produces: rf_mechanism.json, rf_mechanism_summary.txt
Paper: Section 4.6.1. Run make_figures.py afterwards to draw Figure 5.
"""
import json
import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

import config as C
from data_preprocessing import preprocess, load_and_clean


ZERO_FRAC_SPARSE = 0.7
PERM_REPEATS = 10
PERM_SUBSET = 20000
ATTACK_BUDGET = 0.02
STD_MODEL = "model_edge_iiot_binary.keras"


RF_PARAMS = dict(n_jobs=-1, random_state=C.RANDOM_STATE, **C.RF_PARAMS)


def get_feature_names(selected_idx):
    try:
        cfg = C.DATASETS[C.ACTIVE_DATASET]
        X_raw, _ = load_and_clean(cfg)
        all_names = list(X_raw.columns)
        return [all_names[i] for i in selected_idx]
    except Exception as e:
        print(f"[提示] 取列名失败（{e}），改用 f0/f1/... 编号，不影响数字结果。")
        return [f"f{i}" for i in range(len(selected_idx))]


def deep_saliency(model, X):
    Xt = tf.convert_to_tensor(X, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(Xt)
        prob = model(Xt, training=False)
        pred_prob = tf.reduce_max(prob, axis=1)
    g = tape.gradient(pred_prob, Xt)
    return tf.reduce_mean(tf.abs(g), axis=0).numpy()


def collect_thresholds(rf, feature_idx):
    out = []
    for est in rf.estimators_:
        t = est.tree_
        sel = t.feature == feature_idx
        if sel.any():
            out.extend(t.threshold[sel].tolist())
    return np.asarray(out, dtype=float)


def main():
    np.random.seed(C.RANDOM_STATE)
    tf.random.set_seed(C.RANDOM_STATE)
    print(">>> 随机森林机制分析 | 数据集=%s | 模式=%s" % (C.ACTIVE_DATASET, C.MODE))

    data = preprocess()
    Xtr, ytr = data["X_train"], data["y_train"]
    Xte, yte = data["X_test"], data["y_test"]
    cmask = data["continuous_mask"]
    names = get_feature_names(data["selected_idx"])
    n_feat = Xtr.shape[1]


    zero_frac = (Xte == 0).mean(axis=0)
    is_sparse = cmask & (zero_frac >= ZERO_FRAC_SPARSE)
    sparse_idx = np.where(is_sparse)[0]
    print(f">>> 特征数={n_feat} | 连续={int(cmask.sum())} | 稀疏近二值={len(sparse_idx)}")


    print("\n[1/4] 训练随机森林 ...")
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(Xtr, ytr)
    print(f"      clean test accuracy = {rf.score(Xte, yte):.4f}")


    print("[2/4] 计算 impurity-based 与 permutation 重要性 ...")
    imp_gini = rf.feature_importances_
    m = min(PERM_SUBSET, len(Xte))
    sub = np.random.RandomState(C.RANDOM_STATE).choice(len(Xte), m, replace=False)
    perm = permutation_importance(rf, Xte[sub], yte[sub], n_repeats=PERM_REPEATS,
                                  random_state=C.RANDOM_STATE, n_jobs=-1)
    imp_perm = perm.importances_mean

    share_gini = float(imp_gini[sparse_idx].sum() / max(imp_gini.sum(), 1e-12))
    pos_perm = np.clip(imp_perm, 0, None)
    share_perm = float(pos_perm[sparse_idx].sum() / max(pos_perm.sum(), 1e-12))
    print(f"      稀疏指示位占总重要性: gini={share_gini*100:.1f}%  perm={share_perm*100:.1f}%")


    rho_gini = rho_perm = None
    sal = None
    p_std = C.OUTPUT_DIR / STD_MODEL
    if p_std.exists():
        print("[3/4] 与标准深度模型的显著性排序对齐 ...")
        std_model = tf.keras.models.load_model(p_std, compile=False)
        if std_model.input_shape[-1] == n_feat:
            idx3000 = np.random.RandomState(C.RANDOM_STATE).choice(
                len(Xte), min(3000, len(Xte)), replace=False)
            sal = deep_saliency(std_model, Xte[idx3000])
            rho_gini = float(spearmanr(imp_gini, sal).correlation)
            rho_perm = float(spearmanr(imp_perm, sal).correlation)
            print(f"      Spearman rho(深度显著性, RF gini)={rho_gini:.3f}  "
                  f"rho(深度显著性, RF perm)={rho_perm:.3f}")
        else:
            print(f"      [跳过] 模型输入维 {std_model.input_shape[-1]} != 数据 {n_feat}，"
                  f"请把 K_FEATURES 设回 1000。")
    else:
        print(f"[3/4] [跳过] 找不到 {STD_MODEL}，无法做深度模型对照。")


    print("[4/4] 统计稀疏指示位上的分裂阈值 ...")
    per_feature = []
    all_thr = []
    for j in sparse_idx:
        thr = collect_thresholds(rf, int(j))
        if len(thr) == 0:
            continue
        all_thr.append(thr)
        per_feature.append({
            "feature_index": int(j), "feature_name": names[j],
            "n_splits": int(len(thr)),
            "median_threshold": float(np.median(thr)),
            "frac_below_budget": float((thr < ATTACK_BUDGET).mean()),
        })
    all_thr = np.concatenate(all_thr) if all_thr else np.array([])
    if len(all_thr):
        med_all = float(np.median(all_thr))
        frac_below = float((all_thr < ATTACK_BUDGET).mean())
        print(f"      稀疏位分裂总数={len(all_thr)} | 中位阈值={med_all:.4f} | "
              f"低于 {ATTACK_BUDGET} 的比例={frac_below*100:.1f}%")
    else:
        med_all, frac_below = float("nan"), float("nan")
        print("      [警告] 森林没有在稀疏指示位上分裂，请检查 ZERO_FRAC_SPARSE 设置。")


    out = {
        "rf_params": {k: v for k, v in RF_PARAMS.items() if k != "n_jobs"},
        "rf_clean_accuracy": float(rf.score(Xte, yte)),
        "n_features": int(n_feat),
        "n_continuous": int(cmask.sum()),
        "n_sparse_near_binary": int(len(sparse_idx)),
        "sparse_share_of_importance": {"impurity": share_gini, "permutation": share_perm},
        "spearman_vs_deep_saliency": {"impurity": rho_gini, "permutation": rho_perm},
        "attack_budget": ATTACK_BUDGET,
        "split_thresholds_on_sparse": {
            "n_splits_total": int(len(all_thr)),
            "median_threshold": med_all,
            "fraction_below_budget": frac_below,
            "per_feature": per_feature,
        },
        "importance_impurity": imp_gini.tolist(),
        "importance_permutation": imp_perm.tolist(),
        "deep_saliency": (sal.tolist() if sal is not None else None),
        "feature_names": names,
        "is_sparse": is_sparse.tolist(),
    }
    json.dump(out, open(C.OUTPUT_DIR / "rf_mechanism.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    L = ["随机森林机制分析 (Edge-IIoTset, binary)", "",
         f"RF clean accuracy              : {out['rf_clean_accuracy']:.4f}",
         f"稀疏近二值特征数 / 连续 / 全部  : {len(sparse_idx)} / {int(cmask.sum())} / {n_feat}",
         f"稀疏位占 impurity 重要性        : {share_gini*100:.1f}%   <-- 正文填这个",
         f"稀疏位占 permutation 重要性     : {share_perm*100:.1f}%   <-- 正文填这个",
         f"Spearman rho(深度显著性, perm)  : {rho_perm}              <-- 正文填这个",
         f"稀疏位分裂中位阈值              : {med_all:.4f}           <-- 正文填这个",
         f"低于攻击预算 {ATTACK_BUDGET} 的分裂比例  : {frac_below*100:.1f}%   <-- 正文填这个", "",
         "逐特征明细（按低于预算的比例降序）："]
    for r in sorted(per_feature, key=lambda d: -d["frac_below_budget"]):
        L.append(f"  {r['feature_name']:<28} splits={r['n_splits']:<6} "
                 f"median={r['median_threshold']:.4f}  below={r['frac_below_budget']*100:.1f}%")
    txt = "\n".join(L)
    open(C.OUTPUT_DIR / "rf_mechanism_summary.txt", "w", encoding="utf-8").write(txt)
    print("\n" + txt)

    print("\nsaved: rf_mechanism.json, rf_mechanism_summary.txt")
    print("run make_figures.py to draw Figure 5 from this json")


if __name__ == "__main__":
    main()
