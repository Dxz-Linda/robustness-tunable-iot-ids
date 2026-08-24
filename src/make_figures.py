
"""Draw Figure 5 and Figure 7 from the json files, in the palette used by the paper.

Requires: rf_mechanism.json and margin_radius.json
Produces: fig_rf_mechanism.pdf/.png, fig_margin_radius.pdf/.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROSE, BLUE, PURPLE, GREY = "#c1969b", "#4C72B0", "#8c6bb1", "#7F7F7F"
ROSE_L, BLUE_L = "#ded0d2", "#a8bfd8"

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11.5,
    "lines.linewidth": 2.2, "lines.markersize": 7.5,
    "figure.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _place_labels(ax, items, fontsize=9.5):
    """Greedy non-overlapping label placement: try a ring of offsets and keep the
    first that does not collide with an already-placed label."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    offsets = [(7, 6), (7, -12), (-7, 6), (-7, -12), (7, 16), (7, -22),
               (-7, 16), (-7, -22), (0, 18), (0, -24), (14, 0), (-14, 0)]
    placed = []
    for d in sorted(items, key=lambda x: -x["n_splits"]):
        xy = (d["median_threshold"], d["n_splits"])
        best = None
        for dx, dy in offsets:
            t = ax.annotate(d["feature_name"], xy, textcoords="offset points",
                            xytext=(dx, dy), fontsize=fontsize,
                            ha="left" if dx >= 0 else "right")
            bb = t.get_window_extent(renderer=renderer).expanded(1.08, 1.35)
            if not any(bb.overlaps(q) for q in placed):
                placed.append(bb)
                best = t
                break
            t.remove()
        if best is None:
            t = ax.annotate(d["feature_name"], xy, textcoords="offset points",
                            xytext=(7, 6), fontsize=fontsize)
            placed.append(t.get_window_extent(renderer=renderer))


def figure_rf_mechanism(src="rf_mechanism.json", out="fig_rf_mechanism", top_k=15):
    with open(src, "r", encoding="utf-8") as f:
        R = json.load(f)

    names = R["feature_names"]
    is_sparse = np.array(R["is_sparse"], bool)
    gini = np.array(R["importance_impurity"], float)
    perm = np.array(R["importance_permutation"], float)
    budget = R["attack_budget"]
    st = R["split_thresholds_on_sparse"]
    per_feat = {d["feature_index"]: d for d in st["per_feature"]}

    gini_n = gini / gini.sum()
    perm_n = perm / perm.sum() if perm.sum() > 0 else perm
    order = np.argsort(-perm_n)[:top_k][::-1]
    y = np.arange(len(order))
    h = 0.38

    fig, ax = plt.subplots(1, 2, figsize=(15.5, 7.0))


    a = ax[0]
    a.barh(y + h / 2, perm_n[order], height=h, edgecolor="#5a5a5a", linewidth=0.4,
           color=[BLUE if is_sparse[j] else BLUE_L for j in order])
    a.barh(y - h / 2, gini_n[order], height=h, edgecolor="#5a5a5a", linewidth=0.4,
           color=[ROSE if is_sparse[j] else ROSE_L for j in order])

    a.set_yticks(y)
    a.set_yticklabels(["%s  [%s]" % (names[j], "sparse" if is_sparse[j] else "cont.")
                       for j in order])
    a.set_xlabel("Random-forest feature importance, normalised")
    a.set_title("What the forest relies on", fontweight="bold")

    xmax = max(perm_n[order].max(), gini_n[order].max())
    for k, j in enumerate(order):
        if is_sparse[j] and j in per_feat:
            a.text(max(perm_n[j], gini_n[j]) + 0.012 * xmax, y[k],
                   "split at %.4f" % per_feat[j]["median_threshold"],
                   va="center", fontsize=10, color=BLUE)
    a.set_xlim(0, xmax * 1.42)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="#5a5a5a", lw=0.4)
               for c in (BLUE, BLUE_L, ROSE, ROSE_L)]
    a.legend(handles, ["Permutation, sparse", "Permutation, continuous",
                       "Impurity, sparse", "Impurity, continuous"],
             loc="center right", frameon=True)
    a.text(0.98, 0.30,
           "sparse indicators hold %.0f%% of impurity importance\nand %.0f%% of permutation importance"
           % (100 * R["sparse_share_of_importance"]["impurity"],
              100 * R["sparse_share_of_importance"]["permutation"]),
           transform=a.transAxes, ha="right", va="bottom", fontsize=10.5,
           bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6"))
    a.grid(axis="x", alpha=0.3)


    b = ax[1]
    cmap = LinearSegmentedColormap.from_list("house", ["#eaeef3", BLUE_L, BLUE, PURPLE])
    thr = np.array([d["median_threshold"] for d in st["per_feature"]], float)
    nsp = np.array([d["n_splits"] for d in st["per_feature"]], float)
    frac = np.array([d["frac_below_budget"] for d in st["per_feature"]], float)

    b.axvspan(thr.min() * 0.5, budget, color=ROSE, alpha=0.16, zorder=0)
    sc = b.scatter(thr, nsp, s=70 + 320 * frac, c=frac, cmap=cmap, vmin=0, vmax=1,
                   edgecolor="#5a5a5a", linewidth=0.5, zorder=3)
    b.axvline(budget, color=ROSE, ls="--", lw=2.4, zorder=2,
              label="PGD budget %.2f" % budget)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("Median split threshold on a sparse indicator feature")
    b.set_ylabel("Number of splits across the ensemble")
    b.set_title("Where the forest places its thresholds", fontweight="bold")
    cb = plt.colorbar(sc, ax=b)
    cb.set_label("Fraction of that feature's splits below the budget")

    _place_labels(b, [d for d in st["per_feature"]
                      if d["frac_below_budget"] >= 0.30 and d["n_splits"] >= 55])

    b.text(0.02, 0.03,
           "%d of the %d splits on sparse indicators\nlie below the attack budget, %.1f%%"
           % (round(st["fraction_below_budget"] * st["n_splits_total"]),
              st["n_splits_total"], 100 * st["fraction_below_budget"]),
           transform=b.transAxes, ha="left", va="bottom", fontsize=11,
           bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6"))
    b.legend(loc="upper left", frameon=True)
    b.grid(alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(out + ".png")
    fig.savefig(out + ".pdf")
    plt.close(fig)
    print("saved", out + ".pdf")


def figure_margin_radius(src="margin_radius.json", out="fig_margin_radius"):
    with open(src, "r", encoding="utf-8") as f:
        R = json.load(f)
    rows = sorted(R["rows"], key=lambda r: r["sigma"])
    sig = np.array([r["sigma"] for r in rows], float)
    marg = np.array([r["median_margin"] for r in rows], float)
    grad = np.array([r["median_grad_l1"] for r in rows], float)
    pred = np.array([r["median_predicted_radius"] for r in rows], float)
    meas = np.array([r["measured_radius"] for r in rows], float)
    q25 = np.array([r["q25_predicted_radius"] for r in rows], float)
    q75 = np.array([r["q75_predicted_radius"] for r in rows], float)

    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.8))


    a = ax[0]
    l1, = a.plot(sig, grad, "o-", color=ROSE, ms=8,
                 label="Median input-gradient L1 norm, left axis")
    a.set_yscale("log")
    a.set_xlabel("Training-noise level sigma")
    a.set_ylabel("Median input-gradient L1 norm, log scale")
    a.grid(alpha=0.3, which="both")
    a2 = a.twinx()
    l2, = a2.plot(sig, marg, "s--", color=BLUE, ms=8,
                  label="Median decision margin, right axis")
    a2.set_ylabel("Median decision margin")
    a2.set_ylim(0, max(marg) * 1.25)
    a.set_title("The gradient falls by four orders of magnitude,\n"
                "the margin by less than a factor of two", fontweight="bold")
    a.legend(handles=[l1, l2], loc="upper center", frameon=True)
    a.annotate("%.0f" % grad[0], (sig[0], grad[0]), textcoords="offset points",
               xytext=(8, 4), fontsize=10.5, color=ROSE)
    a.annotate("%.2f" % grad[-1], (sig[-1], grad[-1]), textcoords="offset points",
               xytext=(-30, 10), fontsize=10.5, color=ROSE)


    b = ax[1]
    b.fill_between(sig[1:], q25[1:], q75[1:], color=PURPLE, alpha=0.15,
                   label="First-order prediction, interquartile range")
    b.plot(sig, pred, "o-", color=PURPLE, ms=8,
           label="First-order predicted radius, margin over gradient")
    b.plot(sig, meas, "^-", color=BLUE, ms=9,
           label="Measured robustness radius r(sigma)")
    ref = np.linspace(max(sig.min(), 1e-3), sig.max(), 50)
    b.plot(ref, ref, "--", color=GREY, lw=1.8, label="r = sigma reference")
    b.set_yscale("log")
    b.set_xlabel("Training-noise level sigma")
    b.set_ylabel("Robustness radius, log scale")
    b.set_title("Both radii rise by two to three orders of magnitude\n"
                "over the tested range of sigma", fontweight="bold")
    b.grid(alpha=0.3, which="both")
    b.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    fig.savefig(out + ".png")
    fig.savefig(out + ".pdf")
    plt.close(fig)
    print("saved", out + ".pdf")


if __name__ == "__main__":
    figure_rf_mechanism()
    figure_margin_radius()
