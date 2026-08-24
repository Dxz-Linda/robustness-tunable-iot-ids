"""Randomized hyperparameter search for the random-forest baseline.

Scored on the fixed validation split so that no test record influences the
choice. Read the summary before changing anything: if the selected and default
configurations agree, the reported forest numbers stand as they are.

Produces: rf_tuning.json, rf_tuning_summary.txt
Paper: Section 3.8.
"""
import json
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score

import config as C
from data_preprocessing import preprocess

N_ITER = 60
SCORING = "matthews_corrcoef"

SEARCH_SPACE = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [None, 10, 20, 30, 50],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf":  [1, 2, 4, 8],
    "max_features":      ["sqrt", "log2", 0.3, 0.5],
}

DEFAULT_PARAMS = dict(C.RF_PARAMS)


def evaluate(params, data):
    rf = RandomForestClassifier(n_jobs=-1, random_state=C.RANDOM_STATE, **params)
    rf.fit(data["X_train"], data["y_train"])
    yp = rf.predict(data["X_test"])
    return {"accuracy": float(accuracy_score(data["y_test"], yp)),
            "macro_f1": float(f1_score(data["y_test"], yp, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(data["y_test"], yp))}


def main():
    np.random.seed(C.RANDOM_STATE)
    print(">>> 随机森林超参数搜索 | 数据集=%s | 模式=%s" % (C.ACTIVE_DATASET, C.MODE))
    data = preprocess()


    X_tv = np.vstack([data["X_train"], data["X_val"]])
    y_tv = np.concatenate([data["y_train"], data["y_val"]])
    fold = np.concatenate([-np.ones(len(data["X_train"]), dtype=int),
                           np.zeros(len(data["X_val"]), dtype=int)])
    ps = PredefinedSplit(fold)

    t0 = time.time()
    search = RandomizedSearchCV(
        RandomForestClassifier(n_jobs=-1, random_state=C.RANDOM_STATE),
        SEARCH_SPACE, n_iter=N_ITER, scoring=SCORING, cv=ps,
        random_state=C.RANDOM_STATE, n_jobs=1, verbose=2, refit=False)
    search.fit(X_tv, y_tv)
    elapsed = time.time() - t0

    best = search.best_params_
    print(f"\n最优配置（验证集 {SCORING}={search.best_score_:.4f}）: {best}")

    print("\n[对照] 在测试集上评估 最优配置 与 baselines.py 默认配置 ...")
    m_best = evaluate(best, data)
    m_def = evaluate(DEFAULT_PARAMS, data)

    d_acc = m_best["accuracy"] - m_def["accuracy"]
    d_mcc = m_best["mcc"] - m_def["mcc"]
    need_rerun = (abs(d_acc) >= 0.002) or (abs(d_mcc) >= 0.005)

    cv = search.cv_results_
    ranked = sorted(
        [{"params": cv["params"][i], "val_score": float(cv["mean_test_score"][i])}
         for i in range(len(cv["params"]))],
        key=lambda d: -d["val_score"])

    json.dump({"search_space": {k: [str(x) for x in v] for k, v in SEARCH_SPACE.items()},
               "n_iter": N_ITER, "scoring": SCORING, "selected_on": "validation split",
               "seed": C.RANDOM_STATE, "elapsed_sec": elapsed,
               "best_params": {k: str(v) for k, v in best.items()},
               "best_val_score": float(search.best_score_),
               "test_metrics_best": m_best, "test_metrics_default": m_def,
               "need_rerun_paper_numbers": bool(need_rerun),
               "all_candidates_ranked": ranked},
              open(C.OUTPUT_DIR / "rf_tuning.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    L = ["随机森林超参数搜索 (Edge-IIoTset, binary)", "",
         f"搜索方法      : RandomizedSearchCV, {N_ITER} 个候选",
         f"评分          : {SCORING}，在固定的验证集上（PredefinedSplit），测试集不参与",
         f"随机种子      : {C.RANDOM_STATE}",
         f"耗时          : {elapsed/60:.1f} 分钟", "",
         "搜索空间："]
    for k, v in SEARCH_SPACE.items():
        L.append(f"  {k:<20}{v}")
    L += ["",
          f"最优配置      : {best}",
          f"验证集 {SCORING} : {search.best_score_:.4f}", "",
          "测试集对照（这一段决定要不要重跑论文数字）：",
          f"  最优配置    acc={m_best['accuracy']:.4f}  macroF1={m_best['macro_f1']:.4f}  mcc={m_best['mcc']:.4f}",
          f"  默认配置    acc={m_def['accuracy']:.4f}  macroF1={m_def['macro_f1']:.4f}  mcc={m_def['mcc']:.4f}",
          f"  差值        d_acc={d_acc:+.4f}  d_mcc={d_mcc:+.4f}", ""]
    if need_rerun:
        L += ["  >>> 差异显著。请把最优配置写回 baselines.py / robustness_rf.py /",
              "      robustness_tradeoff.py / rf_mechanism.py 的 RandomForestClassifier，",
              "      并重跑 Table 4、Table 7、Figure 4、Figure 8、Figure 10 中的 RF 数字。"]
    else:
        L += ["  >>> 差异在噪声范围内。论文里 RF 的所有数字保持不变，",
              "      在 Section 3.8 写明搜索确认了当前配置已接近最优即可。"]
    L += ["", "验证集分数前 10 的候选："]
    for r in ranked[:10]:
        L.append(f"  {r['val_score']:.4f}  {r['params']}")
    txt = "\n".join(L)
    open(C.OUTPUT_DIR / "rf_tuning_summary.txt", "w", encoding="utf-8").write(txt)
    print("\n" + txt)
    print("\nsaved: rf_tuning.json, rf_tuning_summary.txt")


if __name__ == "__main__":
    main()
