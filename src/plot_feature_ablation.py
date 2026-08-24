"""Clean performance and latency against the number of selected features.

Requires: metrics_edge_iiot_binary_k{10,20,30,40,49}.json from train.py
Produces: fig_feature_ablation.png
Paper: Table 8 and Supplementary Figure S4.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt

KS = [10, 20, 30, 40, 49]

def _locate():
    for cand in (Path("."), Path("./outputs"), Path(__file__).resolve().parent):
        if (cand / f"metrics_edge_iiot_binary_k{KS[0]}.json").exists():
            return cand
    return Path("./outputs")

OUT = _locate()

acc, f1, mcc, infer = [], [], [], []
for k in KS:
    m = json.load(open(OUT / f"metrics_edge_iiot_binary_k{k}.json", encoding="utf-8"))
    acc.append(m["accuracy"]); f1.append(m["macro_f1"])
    mcc.append(m["mcc"]); infer.append(m["infer_ms_per_sample"])

plt.rcParams.update({"font.size": 11})
fig, ax1 = plt.subplots(figsize=(8.2, 5.0))


l1, = ax1.plot(KS, acc, marker="o", color="#C44E52", label="Accuracy")
l2, = ax1.plot(KS, f1,  marker="^", color="#4C72B0", label="Macro-F1")
l3, = ax1.plot(KS, mcc, marker="s", color="#55A868", label="MCC")
ax1.set_xlabel("Number of selected features (K)")
ax1.set_ylabel("Score")
ax1.set_ylim(0.78, 0.94)
ax1.set_xticks(KS)
ax1.grid(True, alpha=0.3)


ax2 = ax1.twinx()
l4, = ax2.plot(KS, infer, marker="D", linestyle="--", color="#7F7F7F",
               label="Inference (ms/sample)")
ax2.set_ylabel("Inference time (ms / sample)")
ax2.set_ylim(0, max(infer) * 1.3)


for x, y in zip(KS, acc):
    ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7),
                 ha="center", fontsize=8, color="#C44E52")
for x, y in zip(KS, mcc):
    ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -13),
                 ha="center", fontsize=8, color="#55A868")


lines = [l1, l2, l3, l4]
ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right", framealpha=0.95)

ax1.set_title("Effect of feature-selection size on Edge-IIoTset (binary)",
              fontweight="bold")
fig.tight_layout()
p = OUT / "fig_feature_ablation.png"
fig.savefig(p, dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved:", p)
