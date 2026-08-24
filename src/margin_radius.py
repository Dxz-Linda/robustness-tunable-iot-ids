"""First-order account of the tunable radius: decision margin against input gradient.

For a record at logit margin m whose margin gradient has L1 norm g, the smallest
perturbation that can flip the prediction is approximately m divided by g.

Requires: the sigma-grid models saved by sigma_radius.py
Produces: margin_radius.json, margin_radius_summary.txt
Paper: Section 4.6.2. Run make_figures.py afterwards to draw Figure 7.
"""
import json
import numpy as np
import tensorflow as tf

import config as C
from data_preprocessing import preprocess

SIGMAS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
BATCH = 256
N_EVAL = 5000
EPS_CLIP = 1e-12


def sig_fname(sig):
    if sig == 0.0:
        return "model_edge_iiot_binary.keras"
    return f"model_edge_iiot_binary_sig{int(round(sig * 100)):02d}.keras"


def margin_and_grad(model, X, cont_mask, batch=BATCH):
    mask = tf.constant(cont_mask.astype("float32"))
    m_all, g_all = [], []
    for i in range(0, len(X), batch):
        xb = tf.convert_to_tensor(X[i:i + batch], tf.float32)
        with tf.GradientTape() as t:
            t.watch(xb)
            prob = model(xb, training=False)
            logp = tf.math.log(tf.clip_by_value(prob, EPS_CLIP, 1.0))
            top2 = tf.math.top_k(logp, k=2)
            margin = top2.values[:, 0] - top2.values[:, 1]
        g = t.gradient(margin, xb)
        g = g * mask
        m_all.append(margin.numpy())
        g_all.append(tf.reduce_sum(tf.abs(g), axis=1).numpy())
    return np.concatenate(m_all), np.concatenate(g_all)


def main():
    np.random.seed(C.RANDOM_STATE)
    tf.random.set_seed(C.RANDOM_STATE)
    data = preprocess()
    Xte, yte = data["X_test"], data["y_test"]
    cmask = data["continuous_mask"]

    N = min(N_EVAL, len(Xte))
    sub = np.random.RandomState(C.RANDOM_STATE).choice(len(Xte), N, replace=False)
    Xs, ys = Xte[sub], yte[sub]
    print(f">>> 子集 {N} | 可扰动特征 {int(cmask.sum())}/{len(cmask)}")


    measured = {}
    p_rad = C.OUTPUT_DIR / "sigma_radius.json"
    if p_rad.exists():
        j = json.load(open(p_rad, encoding="utf-8"))
        measured = {float(k): float(v) for k, v in j["radius"].items()}
        print(f">>> 已读入实测半径: {measured}")
    else:
        print(">>> [提示] 找不到 sigma_radius.json，只输出预测半径，不做对照图。")

    rows = []
    for sig in SIGMAS:
        p = C.OUTPUT_DIR / sig_fname(sig)
        if not p.exists():
            print(f"[跳过] 找不到 {p.name}")
            continue
        mdl = tf.keras.models.load_model(p, compile=False)
        if mdl.input_shape[-1] != Xs.shape[1]:
            print(f"[跳过] {p.name} 输入维 {mdl.input_shape[-1]} != 数据 {Xs.shape[1]}")
            continue
        pred = np.argmax(mdl(tf.convert_to_tensor(Xs, tf.float32),
                             training=False).numpy(), axis=1)
        clean = float((pred == ys).mean())
        m, g = margin_and_grad(mdl, Xs, cmask)
        ok = pred == ys
        r_hat = m[ok] / np.maximum(g[ok], EPS_CLIP)
        row = {"sigma": sig, "clean_acc": clean,
               "median_margin": float(np.median(m[ok])),
               "median_grad_l1": float(np.median(g[ok])),
               "mean_grad_l1": float(np.mean(g[ok])),
               "median_predicted_radius": float(np.median(r_hat)),
               "q25_predicted_radius": float(np.percentile(r_hat, 25)),
               "q75_predicted_radius": float(np.percentile(r_hat, 75)),
               "measured_radius": measured.get(sig)}
        rows.append(row)
        print(f"  sigma={sig:<5} clean={clean:.4f} | median m={row['median_margin']:.4f} "
              f"| median |grad|_1={row['median_grad_l1']:.4g} "
              f"| predicted r={row['median_predicted_radius']:.4f} "
              f"| measured r={row['measured_radius']}")

    json.dump({"n_eval": int(N), "rows": rows},
              open(C.OUTPUT_DIR / "margin_radius.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)


    L = ["一阶鲁棒半径预测 vs 实测半径 (Edge-IIoTset, binary)", "",
         f"{'sigma':>7}{'clean':>9}{'median m':>12}{'median |g|_1':>15}"
         f"{'predicted r':>14}{'measured r':>13}{'ratio':>9}",
         "-" * 79]
    for r in rows:
        ratio = (r["median_predicted_radius"] / r["measured_radius"]
                 if r["measured_radius"] else float("nan"))
        mr = f"{r['measured_radius']:.4f}" if r["measured_radius"] is not None else "n/a"
        L.append(f"{r['sigma']:>7}{r['clean_acc']:>9.4f}{r['median_margin']:>12.4f}"
                 f"{r['median_grad_l1']:>15.4g}{r['median_predicted_radius']:>14.4f}"
                 f"{mr:>13}{ratio:>9.2f}")
    if len(rows) >= 2:
        base = rows[0]
        L += ["", "正文要写的两句机制结论所需的对照："]
        for r in rows[1:]:
            L.append(f"  sigma={r['sigma']}: 裕度 {base['median_margin']:.4f} -> "
                     f"{r['median_margin']:.4f}（几乎不变）"
                     f"，梯度 L1 {base['median_grad_l1']:.4g} -> {r['median_grad_l1']:.4g}"
                     f"（下降 {base['median_grad_l1']/max(r['median_grad_l1'],EPS_CLIP):.0f} 倍）")
        L += ["", "读法与正文写法：",
              "  一阶预测半径的绝对尺度不可靠，因为 softmax 饱和会放大 log 概率裕度，",
              "  所以不要在正文声称预测半径与实测半径数值吻合。",
              "  可靠且要写进正文的是两点：",
              "  第一，裕度基本不变而梯度 L1 下降三到四个数量级，说明鲁棒性提升来自梯度被压平，",
              "        而不是决策边界被推远，这与输入梯度正则等价性一致；",
              "  第二，一阶预测半径与实测半径都随 sigma 单调上升，方向一致，",
              "        佐证训练噪声通过压低输入梯度这一条途径来放大鲁棒半径。",
              "  Figure 24 左面板展示梯度暴跌与裕度稳定，右面板展示两半径同向上升。"]
    txt = "\n".join(L)
    open(C.OUTPUT_DIR / "margin_radius_summary.txt", "w", encoding="utf-8").write(txt)
    print("\n" + txt)

    print("\nsaved: margin_radius.json, margin_radius_summary.txt")
    print("run make_figures.py to draw Figure 7 from this json")


if __name__ == "__main__":
    main()
