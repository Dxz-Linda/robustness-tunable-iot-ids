"""Evidence that no test statistics leak into training.

Training values fall inside [0, 1] by construction while a small fraction of
test values fall outside it, which is only possible if the scaler never saw the
test split.

Produces: preprocessing_integrity_<dataset>.json
Paper: Section 3.2.
"""
import json
import numpy as np

import config as C
from data_preprocessing import preprocess


def main():
    ds = C.ACTIVE_DATASET
    data = preprocess()
    Xtr, Xva, Xte = data["X_train"], data["X_val"], data["X_test"]
    cmask = data["continuous_mask"]
    n_feat = Xtr.shape[1]


    tr_in01 = float(((Xtr >= -1e-9) & (Xtr <= 1 + 1e-9)).mean())
    te_in01 = float(((Xte >= -1e-9) & (Xte <= 1 + 1e-9)).mean())
    te_below = float((Xte < -1e-9).mean())
    te_above = float((Xte > 1 + 1e-9).mean())


    cont = Xte[:, cmask]
    cont_oob = float(((cont < -1e-9) | (cont > 1 + 1e-9)).mean()) if cmask.any() else 0.0
    cont_max_over = float(max(0.0, cont.max() - 1.0)) if cmask.any() else 0.0
    cont_min_under = float(min(0.0, cont.min())) if cmask.any() else 0.0


    n_cont = int(cmask.sum()); n_cat = int((~cmask).sum())

    report = {
        "dataset": ds,
        "n_selected_features": n_feat,
        "n_continuous": n_cont, "n_categorical": n_cat,
        "leakage_check": {
            "note": "scaler & MI-selector fit on TRAIN only (see data_preprocessing.preprocess)",
            "train_fraction_within_[0,1]": round(tr_in01, 6),
            "test_fraction_within_[0,1]": round(te_in01, 6),
            "test_fraction_below_0": round(te_below, 6),
            "test_fraction_above_1": round(te_above, 6),
        },
        "continuous_out_of_range": {
            "fraction_oob": round(cont_oob, 6),
            "max_value_over_1": round(cont_max_over, 6),
            "min_value_under_0": round(cont_min_under, 6),
            "note": "robustness scripts clip x to [0,1] after perturbation, so oob values only matter at the boundary",
        },
    }

    print("=" * 64)
    print(f"预处理完整性报告 | dataset = {ds}")
    print("=" * 64)
    print(f"选中特征数: {n_feat}  (连续 {n_cont} / 类别 {n_cat})")
    print(f"[无泄漏] 训练集落在 [0,1] 的比例: {tr_in01:.4%}  (应≈100%,因 scaler 在训练集拟合)")
    print(f"         测试集落在 [0,1] 的比例: {te_in01:.4%}  (略<100% 证明 scaler 未见测试集)")
    print(f"         测试集 <0 比例: {te_below:.4%} | >1 比例: {te_above:.4%}")
    print(f"[连续特征越界] 比例: {cont_oob:.4%} | 最大超出: +{cont_max_over:.4f} | 最小低于: {cont_min_under:.4f}")
    print("         鲁棒性评测统一 clip 到 [0,1],故越界仅在边界处影响,不改变'ε=量程比例'的解释。")
    print("=" * 64)

    json.dump(report, open(C.OUTPUT_DIR / f"preprocessing_integrity_{ds}.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"saved: preprocessing_integrity_{ds}.json")


if __name__ == "__main__":
    main()
