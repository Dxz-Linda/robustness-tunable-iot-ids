"""Input-gradient saliency of the standard and noise-aware models.

Produces: fig_feature_saliency.png, feature_saliency.csv
Paper: Figure 6.
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import config as C
from data_preprocessing import preprocess

N_EVAL = 3000
TOPN = 20
ZERO_FRAC_SPARSE = 0.7

def saliency_per_feature(model, X):
    Xt = tf.convert_to_tensor(X, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(Xt)
        prob = model(Xt, training=False)
        pred_prob = tf.reduce_max(prob, axis=1)
    grad = tape.gradient(pred_prob, Xt)
    return tf.reduce_mean(tf.abs(grad), axis=0).numpy()

def main():
    np.random.seed(C.RANDOM_STATE); tf.random.set_seed(C.RANDOM_STATE)
    data = preprocess()
    X_test = data["X_test"]; cont_mask = data["continuous_mask"]
    idx = np.random.RandomState(C.RANDOM_STATE).choice(len(X_test), size=min(N_EVAL,len(X_test)), replace=False)
    Xs = X_test[idx]
    n_feat = Xs.shape[1]

    std = tf.keras.models.load_model(C.OUTPUT_DIR/"model_edge_iiot_binary.keras", compile=False)
    noi = tf.keras.models.load_model(C.OUTPUT_DIR/"model_edge_iiot_binary_noise.keras", compile=False)
    for m,name in [(std,"standard"),(noi,"noise-aware")]:
        if m.input_shape[-1]!=n_feat:
            raise ValueError(f"{name} expects {m.input_shape[-1]} features, got {n_feat}. Set K_FEATURES=1000.")

    sal_std = saliency_per_feature(std, Xs)
    sal_noi = saliency_per_feature(noi, Xs)

    zero_frac = (Xs==0).mean(axis=0)
    is_sparse = cont_mask & (zero_frac>=ZERO_FRAC_SPARSE)


    import csv
    with open(C.OUTPUT_DIR/"feature_saliency.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["feature_index","saliency_standard","saliency_noise_aware","zero_fraction","continuous","sparse_near_binary"])
        for j in range(n_feat):
            w.writerow([j, f"{sal_std[j]:.6f}", f"{sal_noi[j]:.6f}", f"{zero_frac[j]:.3f}", bool(cont_mask[j]), bool(is_sparse[j])])


    def avg(mask, s): return float(s[mask].mean()) if mask.any() else float("nan")
    print("=== mean saliency (standard -> noise-aware) ===")
    print(f"sparse near-binary features: {avg(is_sparse,sal_std):.4f} -> {avg(is_sparse,sal_noi):.4f}")
    print(f"other continuous features  : {avg(cont_mask&~is_sparse,sal_std):.4f} -> {avg(cont_mask&~is_sparse,sal_noi):.4f}")


    order = np.argsort(sal_std)[::-1][:TOPN]
    y = np.arange(len(order))
    fig,ax = plt.subplots(figsize=(8.4,6.2))
    ax.barh(y-0.2, sal_std[order], height=0.4, color="#c1969b", edgecolor="#5a5a5a", lw=0.4, label="Standard")
    ax.barh(y+0.2, sal_noi[order], height=0.4, color="#6f8fb3", edgecolor="#5a5a5a", lw=0.4, label="Noise-aware (σ=0.1)")
    for k,j in enumerate(order):
        tag = "sparse" if is_sparse[j] else ("cont." if cont_mask[j] else "cat.")
        ax.text(max(sal_std[j],sal_noi[j])+0.002, k, f"f{j} ({tag})", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([f"#{k+1}" for k in range(len(order))], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("Mean |gradient| saliency"); ax.legend(loc="lower right")
    ax.set_title("Feature saliency: standard vs noise-aware (Edge-IIoTset, binary)", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(C.OUTPUT_DIR/"fig_feature_saliency.png", dpi=300, bbox_inches="tight")
    print("\nsaved fig_feature_saliency.png and feature_saliency.csv")

if __name__=="__main__":
    main()
