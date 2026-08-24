
"""Paired tests on the robustness ablation, computed from its json alone.

Requires: ablation_robustness.json
Produces: paired_tests_robust.json, paired_tests_robust_summary.txt
Paper: the remaining rows of Table 7 and the argument in Section 4.4.1.
"""
import json, os, sys
import numpy as np
from scipy import stats


SRC_CANDIDATES = ["ablation_robustness.json",
                  os.path.join("outputs", "ablation_robustness.json"),
                  "ablation_robustness_ckpt.json",
                  os.path.join("outputs", "ablation_robustness_ckpt.json")]
OUT_DIR = "outputs" if os.path.isdir("outputs") else "."
VARIANTS = ["BiLSTM only", "CNN only", "CNN+BiLSTM (no attn.)", "Full (+attention)"]
METRICS = ["clean_acc", "noise_0.1", "pgd_0.02"]


def load():
    src = next((p for p in SRC_CANDIDATES if os.path.isfile(p)), None)
    if src is None:
        sys.exit("找不到 ablation_robustness.json。请先跑完 ablation_robustness.py，\n"
                 "或者把本脚本放到与该 json 同一目录。已找过：\n  "
                 + "\n  ".join(SRC_CANDIDATES))
    print(">>> 读取 %s" % src)
    with open(src, "r", encoding="utf-8") as f:
        d = json.load(f)
    per_run = d.get("per_run", d)
    seeds = d.get("seeds")
    if seeds is None:
        seeds, seen = [], set()
        for k in per_run:
            s0 = k.split("|")[0]
            if s0 not in seen:
                seen.add(s0)
                seeds.append(int(s0))
    have = [s0 for s0 in seeds
            if all("%s|%s|%s" % (s0, v, t) in per_run
                   for v in VARIANTS for t in ("standard", "noise-aware"))]
    if len(have) < 2:
        sys.exit("只有 %d 个种子的记录是完整的，配对检验至少需要 2 个。\n"
                 "请等 ablation_robustness.py 跑完所有种子再来。" % len(have))
    if len(have) < len(seeds):
        print(">>> 提示：%d 个种子里只有 %d 个记录完整，只用完整的那些做检验。"
              % (len(seeds), len(have)))
    return have, per_run


def series(per_run, seeds, variant, training, metric):
    return np.array([per_run["%s|%s|%s" % (s, variant, training)][metric] for s in seeds], float)


def paired(label, a, b):
    d = a - b
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    t, p = stats.ttest_rel(a, b)
    tcrit = stats.t.ppf(0.975, n - 1)
    lo, hi = mean - tcrit * se, mean + tcrit * se
    try:
        w = float(stats.wilcoxon(a, b).pvalue)
    except Exception:
        w = float("nan")
    return {
        "label": label, "n_pairs": n,
        "values_a": a.tolist(), "values_b": b.tolist(), "diffs": d.tolist(),
        "mean_diff": mean, "sd_diff": sd, "t": float(t), "p_ttest": float(p),
        "ci95": [float(lo), float(hi)],
        "cohens_dz": float(mean / sd) if sd > 0 else float("nan"),
        "p_wilcoxon": w,
    }


def main():
    seeds, per_run = load()
    out = {}


    for m in ["noise_0.1", "pgd_0.02", "clean_acc"]:
        a = series(per_run, seeds, "Full (+attention)", "standard", m)
        b = series(per_run, seeds, "CNN+BiLSTM (no attn.)", "standard", m)
        out["attn_standard_" + m] = paired(
            "B1 attention effect under standard training, metric %s: Full - NoAttn" % m, a, b)


    for m in ["noise_0.1", "pgd_0.02", "clean_acc"]:
        a = series(per_run, seeds, "Full (+attention)", "noise-aware", m)
        b = series(per_run, seeds, "CNN+BiLSTM (no attn.)", "noise-aware", m)
        out["attn_noiseaware_" + m] = paired(
            "B2 attention effect under noise-aware training, metric %s: Full - NoAttn" % m, a, b)


    for v in VARIANTS:
        for m in ["noise_0.1", "pgd_0.02", "clean_acc"]:
            a = series(per_run, seeds, v, "noise-aware", m)
            b = series(per_run, seeds, v, "standard", m)
            out["train_%s_%s" % (v, m)] = paired(
                "B3 noise-aware minus standard, %s, metric %s" % (v, m), a, b)


    lines = []
    for m in ["noise_0.1", "pgd_0.02"]:
        sd_std = [series(per_run, seeds, v, "standard", m).std(ddof=1) for v in VARIANTS]
        sd_na = [series(per_run, seeds, v, "noise-aware", m).std(ddof=1) for v in VARIANTS]
        out["spread_" + m] = {
            "label": "B4 across-seed sample SD, metric %s" % m,
            "variants": VARIANTS,
            "sd_standard": [float(x) for x in sd_std],
            "sd_noise_aware": [float(x) for x in sd_na],
            "mean_sd_standard": float(np.mean(sd_std)),
            "mean_sd_noise_aware": float(np.mean(sd_na)),
            "ratio": float(np.mean(sd_std) / np.mean(sd_na)),
        }

    with open(os.path.join(OUT_DIR, "paired_tests_robust.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    for k, v in out.items():
        if "label" in v and "mean_diff" in v:
            lines.append(v["label"])
            lines.append("    mean paired difference = %+.4f  (%+.2f points)" % (v["mean_diff"], 100 * v["mean_diff"]))
            lines.append("    t(%d) = %.3f,  p = %.4f,  Cohen dz = %+.3f,  Wilcoxon p = %.4f"
                         % (v["n_pairs"] - 1, v["t"], v["p_ttest"], v["cohens_dz"], v["p_wilcoxon"]))
            lines.append("    95%% CI = [%+.4f, %+.4f]  (%+.2f to %+.2f points)  %s"
                         % (v["ci95"][0], v["ci95"][1], 100 * v["ci95"][0], 100 * v["ci95"][1],
                            "contains zero" if v["ci95"][0] < 0 < v["ci95"][1] else "excludes zero"))
            lines.append("")
        else:
            lines.append(v["label"])
            for name, s1, s2 in zip(v["variants"], v["sd_standard"], v["sd_noise_aware"]):
                lines.append("    %-24s SD standard = %.4f   SD noise-aware = %.4f" % (name, s1, s2))
            lines.append("    mean SD standard = %.4f, mean SD noise-aware = %.4f, ratio = %.1f"
                         % (v["mean_sd_standard"], v["mean_sd_noise_aware"], v["ratio"]))
            lines.append("")

    txt = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "paired_tests_robust_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
