"""Verify that the 200,000-record stratified subsample preserves class shares.

Produces: sampling_check.json, sampling_check_summary.txt
Paper: Section 3.1.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import chisquare
from sklearn.model_selection import train_test_split

import config as C

MODES = ["binary", "multiclass"]


def analyse(y_full, max_rows, seed, tag):
    n_full = len(y_full)
    idx = np.arange(n_full)
    try:
        idx_s, _, y_s, _ = train_test_split(
            idx, y_full, train_size=max_rows, stratify=y_full, random_state=seed)
        method = "sklearn train_test_split, stratify on the label, shuffle enabled"
    except ValueError:
        rng = np.random.RandomState(seed)
        idx_s = rng.choice(n_full, size=max_rows, replace=False)
        y_s = y_full.iloc[idx_s]
        method = "fallback simple random sampling, stratification not feasible"

    full_counts = y_full.value_counts()
    samp_counts = pd.Series(y_s).value_counts()
    classes = list(full_counts.index)

    rows, max_dev, max_dev_class = [], 0.0, None
    obs, exp = [], []
    for c in classes:
        f_share = float(full_counts[c] / n_full)
        s_n = int(samp_counts.get(c, 0))
        s_share = s_n / len(y_s)
        dev = abs(s_share - f_share) * 100.0
        if dev > max_dev:
            max_dev, max_dev_class = dev, str(c)
        rows.append({"class": str(c), "full_n": int(full_counts[c]),
                     "full_share_pct": f_share * 100, "sample_n": s_n,
                     "sample_share_pct": s_share * 100, "abs_dev_pp": dev})
        obs.append(s_n)
        exp.append(f_share * len(y_s))

    exp = np.asarray(exp, float)
    exp = exp * (sum(obs) / exp.sum())
    chi2, p = chisquare(f_obs=np.asarray(obs, float), f_exp=exp)

    rows_sorted = sorted(rows, key=lambda d: d["full_n"])
    rarest = rows_sorted[:2]

    return {"tag": tag, "method": method, "seed": seed,
            "n_full": n_full, "n_sample": int(len(y_s)),
            "max_abs_deviation_pp": max_dev, "max_deviation_class": max_dev_class,
            "chi2": float(chi2), "p_value": float(p),
            "degrees_of_freedom": len(classes) - 1,
            "rarest_classes": rarest, "per_class": rows}


def main():
    if C.ACTIVE_DATASET != "edge_iiot":
        print("[提示] 本脚本针对 Edge-IIoTset。已临时把 ACTIVE_DATASET 切到 edge_iiot。")
    cfg = C.DATASETS["edge_iiot"]
    max_rows = cfg["max_rows"]
    seed = C.RANDOM_STATE

    cols = [cfg["label_multiclass"], cfg["label_binary"]]
    print(f">>> 读取标签列 {cols} ...")
    df = pd.read_csv(cfg["csv_path"], usecols=cols, low_memory=False)
    print(f">>> 全量行数 = {len(df)} | 目标抽样 = {max_rows}")

    results = {}
    for mode in MODES:
        col = cfg["label_multiclass"] if mode == "multiclass" else cfg["label_binary"]
        y_full = df[col].astype(str).str.strip()
        print(f"\n=== 分层依据 = {col}（MODE='{mode}'） ===")
        r = analyse(y_full, max_rows, seed, tag=mode)
        results[mode] = r
        print(f"    最大份额偏差 = {r['max_abs_deviation_pp']:.4f} 个百分点 "
              f"（类别 {r['max_deviation_class']}）")
        print(f"    卡方 = {r['chi2']:.3f}, df = {r['degrees_of_freedom']}, p = {r['p_value']:.4f}")


    y_mc = df[cfg["label_multiclass"]].astype(str).str.strip()
    y_bin = df[cfg["label_binary"]].astype(str).str.strip()
    idx = np.arange(len(df))
    idx_s, _, _, _ = train_test_split(idx, y_bin, train_size=max_rows,
                                      stratify=y_bin, random_state=seed)
    mc_full = y_mc.value_counts() / len(y_mc)
    mc_samp = y_mc.iloc[idx_s].value_counts() / len(idx_s)
    cross_dev = float(max(abs(mc_samp.get(c, 0.0) - mc_full[c]) * 100 for c in mc_full.index))
    print(f"\n=== 交叉检查：按 binary 分层时，multiclass 最大份额偏差 = "
          f"{cross_dev:.4f} 个百分点 ===")
    results["cross_check_binary_strat_multiclass_dev_pp"] = cross_dev

    json.dump(results, open(C.OUTPUT_DIR / "sampling_check.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    L = ["Edge-IIoTset 分层抽样验证", "",
         f"抽样工具  : sklearn.model_selection.train_test_split，train_size={max_rows}，"
         f"stratify=标签，random_state={seed}",
         f"全量行数  : {len(df)}", ""]
    for mode in MODES:
        r = results[mode]
        col = cfg["label_multiclass"] if mode == "multiclass" else cfg["label_binary"]
        L += [f"--- 分层依据 {col}（MODE='{mode}'） ---",
              f"  最大绝对份额偏差 : {r['max_abs_deviation_pp']:.4f} 个百分点，"
              f"出现在类别 {r['max_deviation_class']}   <-- 正文填这个",
              f"  卡方拟合优度     : chi2={r['chi2']:.3f}, df={r['degrees_of_freedom']}, "
              f"p={r['p_value']:.4f}   <-- 正文填这个"]
        for rr in r["rarest_classes"]:
            L.append(f"  最稀有类 {rr['class']:<18} 全量 {rr['full_n']} 条 -> "
                     f"样本 {rr['sample_n']} 条   <-- 正文填这个")
        L.append("")
    L += [f"交叉检查：按 binary 标签分层时 multiclass 的最大份额偏差 = {cross_dev:.4f} 个百分点。",
          "若这个值很小，可在正文写明两种模式下的抽样都保持了细粒度类别比例；",
          "若较大，则需在 Section 3.1 说明二分类与多分类实验各自按对应标签分层抽样。", "",
          "逐类明细（multiclass 分层）："]
    for r in results["multiclass"]["per_class"]:
        L.append(f"  {r['class']:<22} full {r['full_share_pct']:>6.3f}%  "
                 f"sample {r['sample_share_pct']:>6.3f}%  dev {r['abs_dev_pp']:.4f} pp")
    txt = "\n".join(L)
    open(C.OUTPUT_DIR / "sampling_check_summary.txt", "w", encoding="utf-8").write(txt)
    print("\n" + txt)
    print("\nsaved: sampling_check.json, sampling_check_summary.txt")


if __name__ == "__main__":
    main()
