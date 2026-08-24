"""Paired tests across seed-matched runs, replacing visual error-bar overlap.

Requires: multiseed_baselines.json and multiseed_robustness.json
Produces: paired_tests.json, paired_tests_summary.txt
Paper: the first rows of Table 7 and the argument in Section 4.4.
"""
import json
import numpy as np
from scipy import stats

import config as C

BASE_JSON = "multiseed_baselines.json"
ROB_JSON = "multiseed_robustness.json"

NAME_OURS = "Ours (CNN-BiLSTM-Attn)"
NAME_NOATT = "CNN+BiLSTM (NoAttn)"
NAME_RF = "Random Forest"


def paired_test(a, b, label):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = a - b
    n = len(d)
    out = {"label": label, "n_pairs": int(n),
           "values_a": a.tolist(), "values_b": b.tolist(), "diffs": d.tolist(),
           "mean_diff": float(d.mean()),
           "sd_diff": float(d.std(ddof=1)) if n > 1 else 0.0}
    if n < 2 or d.std(ddof=1) == 0:
        out.update({"t": None, "p_ttest": None, "ci95": [None, None],
                    "cohens_dz": None, "p_wilcoxon": None})
        return out
    t, p = stats.ttest_rel(a, b)
    ci = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=stats.sem(d))
    dz = float(d.mean() / d.std(ddof=1))
    try:
        _, pw = stats.wilcoxon(a, b)
    except Exception:
        pw = None
    out.update({"t": float(t), "p_ttest": float(p),
                "ci95": [float(ci[0]), float(ci[1])],
                "cohens_dz": dz,
                "p_wilcoxon": (float(pw) if pw is not None else None)})
    return out


def fmt(r, unit="points", scale=100.0):
    if r["t"] is None:
        return f"{r['label']}: 数据不足，无法检验"
    lo, hi = r["ci95"][0] * scale, r["ci95"][1] * scale
    s = (f"{r['label']}\n"
         f"    配对差均值 = {r['mean_diff']*scale:+.4f} {unit}，配对标准差 = {r['sd_diff']*scale:.4f}\n"
         f"    t({r['n_pairs']-1}) = {r['t']:.3f},  p = {r['p_ttest']:.4f}\n"
         f"    95% 置信区间 = [{lo:+.4f}, {hi:+.4f}] {unit}  "
         f"{'（含 0，差异不可靠）' if lo <= 0 <= hi else '（不含 0，差异可靠）'}\n"
         f"    Cohen dz = {r['cohens_dz']:+.3f}")
    if r["p_wilcoxon"] is not None:
        s += f",  Wilcoxon p = {r['p_wilcoxon']:.4f}"
    return s


def main():
    results = {}
    L = ["配对统计检验（同一批训练种子上的配对比较）", ""]


    pb = C.OUTPUT_DIR / BASE_JSON
    if pb.exists():
        jb = json.load(open(pb, encoding="utf-8"))
        agg = jb["aggregate"]
        seeds = jb["seeds"]
        L += [f"数据来源 A : {BASE_JSON}，种子 {seeds}", ""]
        for met, unit in [("accuracy", "accuracy points"), ("mcc", "MCC points"),
                          ("macro_f1", "macro-F1 points")]:
            if NAME_OURS in agg and NAME_NOATT in agg:
                r = paired_test(agg[NAME_OURS][met]["values"],
                                agg[NAME_NOATT][met]["values"],
                                f"A. 注意力效应，指标 {met}：Ours - CNN+BiLSTM(NoAttn)")
                results[f"attention_{met}"] = r
                L.append(fmt(r, unit))
                L.append("")
        if NAME_RF in agg and NAME_OURS in agg:
            r = paired_test(agg[NAME_RF]["accuracy"]["values"],
                            agg[NAME_OURS]["accuracy"]["values"],
                            "C. 随机森林对照，指标 accuracy：RandomForest - Ours")
            results["rf_vs_ours_accuracy"] = r
            L.append(fmt(r, "accuracy points"))
            L.append("")
    else:
        L += [f"[跳过 A/C] 找不到 {BASE_JSON}，请先跑 multiseed_baselines.py", ""]


    pr = C.OUTPUT_DIR / ROB_JSON
    if pr.exists():
        jr = json.load(open(pr, encoding="utf-8"))
        per = jr["per_seed"]
        seeds = [s for s in jr["seeds"] if str(s) in per]
        L += [f"数据来源 B : {ROB_JSON}，种子 {seeds}", ""]

        def series(variant, getter):
            return [getter(per[str(s)][variant]) for s in seeds]

        checks = [
            ("clean accuracy", lambda d: d["clean_acc"], "accuracy points"),
            ("accuracy under noise 0.1", lambda d: d["noise_acc"]["0.1"], "accuracy points"),
            ("accuracy under noise 0.2", lambda d: d["noise_acc"]["0.2"], "accuracy points"),
            ("accuracy under PGD 0.02", lambda d: d["pgd_acc"]["0.02"], "accuracy points"),
            ("attack recall", lambda d: d["attack_recall"], "recall points"),
        ]
        for label, getter, unit in checks:
            try:
                a = series("noise-aware", getter)
                b = series("standard", getter)
            except KeyError:
                continue
            r = paired_test(a, b, f"B. 加噪训练效应，指标 {label}：noise-aware - standard")
            results[f"noiseaware_{label.replace(' ', '_')}"] = r
            L.append(fmt(r, unit))
            L.append("")
    else:
        L += [f"[跳过 B] 找不到 {ROB_JSON}，请先跑 multiseed_robustness.py", ""]

    L += ["注意：五个种子下 Wilcoxon 符号秩检验能取到的最小 p 值是 0.0625，",
          "所以在正文里应以配对 t 检验为主，Wilcoxon 只作方向一致性的佐证，",
          "并说明五种子检验的统计功效有限，结论是差异小于本实验的分辨率，而非差异恰为零。"]

    json.dump(results, open(C.OUTPUT_DIR / "paired_tests.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    txt = "\n".join(L)
    open(C.OUTPUT_DIR / "paired_tests_summary.txt", "w", encoding="utf-8").write(txt)
    print(txt)
    print("\nsaved: paired_tests.json, paired_tests_summary.txt")


if __name__ == "__main__":
    main()
