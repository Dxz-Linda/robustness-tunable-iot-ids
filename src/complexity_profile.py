"""Parameter count, model size, FLOPs, activation memory and latency.

Latency is reported under three protocols because they differ by more than an
order of magnitude, and the recurrent FLOPs are computed analytically because
the profiler does not resolve the fused recurrent kernels.

Produces: complexity_profile.json, complexity_profile_table.txt
Paper: Table 4.
"""
import json
import os
import platform
import subprocess
import time
import numpy as np
import tensorflow as tf

import config as C
from data_preprocessing import preprocess

N_LATENCY = 500
N_WARMUP = 100
BATCH_FOR_THROUGHPUT = getattr(C, "BATCH_SIZE", 256)

MODEL_FILES = [
    ("Proposed MI-CNN-BiLSTM-Attention, all features", "model_edge_iiot_binary.keras"),
    ("Proposed, K = 20 features", "model_edge_iiot_binary_k20.keras"),
    ("Proposed, K = 10 features", "model_edge_iiot_binary_k10.keras"),
    ("Proposed, noise-aware sigma = 0.1", "model_edge_iiot_binary_noise.keras"),
]


def hardware_string():
    cpu = ""
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                stderr=subprocess.DEVNULL, timeout=20).decode(errors="ignore")
            cpu = out.strip().splitlines()[0].strip() if out.strip() else ""
        except Exception:
            cpu = ""
        if not cpu:
            cpu = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
    if not cpu:
        cpu = platform.processor() or platform.machine()
    gpus = tf.config.list_physical_devices("GPU")
    return "%s | GPUs visible: %d | TF %s | %s" % (cpu, len(gpus), tf.__version__,
                                                   platform.system())


def count_flops_profiler(model):
    try:
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2)
        from tensorflow.python.profiler.model_analyzer import profile
        from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
        spec = tf.TensorSpec([1] + list(model.inputs[0].shape[1:]), tf.float32)
        conc = tf.function(lambda x: model(x)).get_concrete_function(spec)
        frozen = convert_variables_to_constants_v2(conc)
        opts = ProfileOptionBuilder.float_operation()
        opts["output"] = "none"
        info = profile(frozen.graph, options=opts)
        return int(info.total_float_ops)
    except Exception as e:
        print("      [提示] FLOPs 统计失败: %s" % e)
        return None


def _layer_input_shape(layer):
    try:
        s = layer.input_shape
    except Exception:
        try:
            s = layer.input.shape
        except Exception:
            return None
    if isinstance(s, list):
        s = s[0]
    return tuple(s)


def count_flops_recurrent(model):
    total = 0
    detail = []
    for lyr in model.layers:
        n_dir, cell = 0, None
        if isinstance(lyr, tf.keras.layers.Bidirectional):
            n_dir, cell = 2, lyr.forward_layer
        elif isinstance(lyr, tf.keras.layers.LSTM):
            n_dir, cell = 1, lyr
        elif isinstance(lyr, tf.keras.layers.GRU):
            n_dir, cell = 1, lyr
        if cell is None:
            continue
        shp = _layer_input_shape(lyr)
        if shp is None or len(shp) < 3 or shp[1] is None or shp[2] is None:
            print("      [提示] 循环层 %s 的输入形状不完整，无法解析补算" % lyr.name)
            continue
        T, Cin = int(shp[1]), int(shp[2])
        H = int(cell.units)
        gates = 4 if not isinstance(cell, tf.keras.layers.GRU) else 3
        macs = n_dir * T * gates * H * (Cin + H)
        flops = 2 * macs
        total += flops
        detail.append({"layer": lyr.name, "type": type(lyr).__name__,
                       "T": T, "in_channels": Cin, "units": H,
                       "directions": n_dir, "flops": int(flops)})
    return int(total), detail


def activation_footprint(model, dtype_bytes=4):
    peak, total = 0, 0
    for lyr in model.layers:
        s = None
        for getter in (lambda l: l.output_shape,
                       lambda l: tuple(l.output.shape),
                       lambda l: tuple(l.output[0].shape)):
            try:
                s = getter(lyr)
                break
            except Exception:
                continue
        if s is None:
            continue
        if isinstance(s, list):
            s = s[0]
        if not isinstance(s, (list, tuple)):
            continue
        dims = [d for d in s[1:] if d is not None]
        if not dims:
            continue
        n = int(np.prod(dims)) * dtype_bytes
        total += n
        peak = max(peak, n)
    return round(peak / 1024 ** 2, 4), round(total / 1024 ** 2, 4)


def tracemalloc_memory(model, X, n=500):
    import tracemalloc
    tracemalloc.start()
    _ = model.predict(X[:min(n, len(X))], batch_size=1, verbose=0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return round(float(peak / 1024 ** 2), 4)


def latency_three_ways(model, X, n=N_LATENCY, warmup=N_WARMUP,
                       batch=BATCH_FOR_THROUGHPUT):
    res = {}
    n = min(n, len(X))


    for i in range(min(warmup, len(X))):
        model(tf.convert_to_tensor(X[i:i + 1], tf.float32), training=False)
    ts = []
    for i in range(n):
        xb = tf.convert_to_tensor(X[i:i + 1], tf.float32)
        t0 = time.perf_counter()
        model(xb, training=False)
        ts.append(time.perf_counter() - t0)
    ts = np.asarray(ts) * 1e3
    res["eager_ms_per_sample"] = float(np.median(ts))
    res["eager_p95_ms_per_sample"] = float(np.percentile(ts, 95))


    spec = tf.TensorSpec([1] + list(model.inputs[0].shape[1:]), tf.float32)

    @tf.function(input_signature=[spec], reduce_retracing=True)
    def _fwd(x):
        return model(x, training=False)

    for i in range(min(warmup, len(X))):
        _fwd(tf.convert_to_tensor(X[i:i + 1], tf.float32))
    ts = []
    for i in range(n):
        xb = tf.convert_to_tensor(X[i:i + 1], tf.float32)
        t0 = time.perf_counter()
        _fwd(xb)
        ts.append(time.perf_counter() - t0)
    ts = np.asarray(ts) * 1e3
    res["compiled_ms_per_sample"] = float(np.median(ts))
    res["compiled_p95_ms_per_sample"] = float(np.percentile(ts, 95))


    nb = min(len(X), max(batch * 8, 2048))
    model.predict(X[:batch], batch_size=batch, verbose=0)
    t0 = time.perf_counter()
    model.predict(X[:nb], batch_size=batch, verbose=0)
    res["batched_ms_per_sample"] = float((time.perf_counter() - t0) / nb * 1e3)
    res["throughput_batch_size"] = int(batch)
    res["throughput_n_samples"] = int(nb)
    return res


def profile_model(name, mdl, X_real, seed=0):
    n_par = int(mdl.count_params())
    d = int(mdl.input_shape[-1])
    rec = {"name": name, "n_params": n_par,
           "size_MB_float32": round(n_par * 4 / 1024 ** 2, 4),
           "input_dim": d}

    rec["flops_profiled"] = count_flops_profiler(mdl)
    rec_flops, rec_detail = count_flops_recurrent(mdl)
    rec["flops_recurrent_analytic"] = rec_flops
    rec["flops_recurrent_detail"] = rec_detail
    rec["flops_total"] = (rec["flops_profiled"] or 0) + rec_flops

    peak, tot = activation_footprint(mdl)
    rec["activation_peak_MB"] = peak
    rec["activation_sum_MB"] = tot

    if X_real is not None and X_real.shape[1] == d:
        Xuse, synthetic = X_real, False
    else:
        rng = np.random.RandomState(seed)
        Xuse = rng.rand(max(N_LATENCY, BATCH_FOR_THROUGHPUT * 8) + N_WARMUP,
                        d).astype("float32")
        synthetic = True
        print("      [提示] 本模型输入 %d 维，与当前预处理的维度不同，"
              "改用同形状随机张量计时。延迟只取决于形状，结论不受影响。" % d)
    rec["timing_input"] = "synthetic" if synthetic else "real test data"
    rec.update(latency_three_ways(mdl, Xuse))
    rec["tracemalloc_MB_reference_only"] = tracemalloc_memory(mdl, Xuse)
    return rec


def main():
    np.random.seed(C.RANDOM_STATE)
    tf.random.set_seed(C.RANDOM_STATE)
    hw = hardware_string()
    print(">>> 硬件: %s" % hw)

    data = preprocess()
    X_full = data["X_test"]
    n_classes = len(np.unique(data["y_train"]))
    print(">>> 全特征维度 = %d" % X_full.shape[1])

    records = []
    for name, fname in MODEL_FILES:
        p = C.OUTPUT_DIR / fname
        if not p.exists():
            print("[跳过] 找不到 %s" % fname)
            continue
        print("\n--- %s (%s) ---" % (name, fname))
        tf.keras.backend.clear_session()
        mdl = tf.keras.models.load_model(p, compile=False)
        rec = profile_model(name, mdl, X_full)
        rec["source_file"] = fname
        records.append(rec)
        print("      params=%s | size=%.2f MB | flops prof=%s recur=%s total=%s"
              % ("{:,}".format(rec["n_params"]), rec["size_MB_float32"],
                 rec["flops_profiled"], rec["flops_recurrent_analytic"], rec["flops_total"]))
        print("      latency ms/sample: eager=%.4f compiled=%.4f batched=%.4f"
              % (rec["eager_ms_per_sample"], rec["compiled_ms_per_sample"],
                 rec["batched_ms_per_sample"]))

    try:
        import baselines as B
        print("\n--- 结构对照，未训练，只测复杂度 ---")
        for nm, fn in [("CNN + BiLSTM without attention", B.build_cnn_bilstm),
                       ("CNN only", B.build_cnn),
                       ("BiLSTM only", B.build_bilstm)]:
            tf.keras.backend.clear_session()
            m = fn(X_full.shape[1], n_classes)
            rec = profile_model(nm, m, X_full)
            rec["source_file"] = "rebuilt from baselines.py, untrained"
            records.append(rec)
            print("      %s: params=%s | flops total=%s | compiled=%.4f ms/sample"
                  % (nm, "{:,}".format(rec["n_params"]), rec["flops_total"],
                     rec["compiled_ms_per_sample"]))
    except Exception as e:
        print("[提示] 结构对照跳过: %s" % e)

    json.dump({"hardware": hw,
               "notes": {
                   "latency": ("eager 为逐条 eager 调用，主要反映框架开销；"
                               "compiled 为 tf.function 编译后逐条调用，是真实的单样本部署延迟；"
                               "batched 为批量推理折算到每样本，与 Table 5 同口径。"
                               "三者不可混用，正文 Table 11 用 compiled。"),
                   "flops": ("profiler 解析不了融合的循环核，"
                             "循环部分按 2 x n_dir x T x 4 x H x (C + H) 解析补算，"
                             "flops_total 为二者之和。"),
                   "memory": ("activation_peak 与 activation_sum 按层输出张量解析计算，"
                              "batch 为 1，与硬件无关；tracemalloc 只统计 Python 堆，仅供参考。"),
                   "throughput_batch_size": BATCH_FOR_THROUGHPUT},
               "records": records},
              open(C.OUTPUT_DIR / "complexity_profile.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    L = ["Table 11 复杂度与部署足迹（本文测量部分）", "",
         "硬件: %s" % hw,
         "延迟三种口径，单位均为毫秒每样本：",
         "  eager     逐条 eager 调用，主要反映框架开销，仅供诊断，不进正文",
         "  compiled  tf.function 编译后逐条调用，Table 11 的延迟列用这一个",
         "  batched   批量推理折算，batch=%d，与 Table 5 同口径" % BATCH_FOR_THROUGHPUT,
         "注意: 延迟只在同一台机器上可比。Table 11 必须带硬件列，",
         "      并在正文写明跨研究的延迟仅作数量级参照。",
         "      可跨研究比较的是参数量、模型体积、FLOPs 与激活内存。", "",
         "%-44s%10s%8s%14s%14s%14s%10s%10s%10s%9s%9s"
         % ("Model", "Params", "MB", "FLOPs prof", "FLOPs recur", "FLOPs total",
            "eager", "compiled", "batched", "ActPeak", "ActSum"),
         "-" * 165]
    for r in records:
        def f(v, spec, na="n/a"):
            return ("%" + spec) % v if v is not None else na
        L.append("%-44s%10s%8.2f%14s%14s%14s%10.4f%10.4f%10.4f%9.3f%9.3f"
                 % (r["name"][:43],
                    "{:,}".format(r["n_params"]), r["size_MB_float32"],
                    "{:,}".format(r["flops_profiled"]) if r["flops_profiled"] else "n/a",
                    "{:,}".format(r["flops_recurrent_analytic"]),
                    "{:,}".format(r["flops_total"]),
                    r["eager_ms_per_sample"], r["compiled_ms_per_sample"],
                    r["batched_ms_per_sample"],
                    r["activation_peak_MB"], r["activation_sum_MB"]))

    L += ["", "自查：batched 那一列与 Table 5 的 0.044 到 0.083 ms 应当同量级。",
          "      若相差很大，说明 Table 5 当初用的批大小与这里不同，需要统一后重出 Table 5。",
          "", "还需要你手工补的三行：从 [19] 与 [49] 等近期轻量检测器原文里摘参数量与体积，",
          "摘不到的写 not reported，切勿估算，也切勿把别人在别的硬件上的延迟与本文并列解读。"]
    txt = "\n".join(L)
    open(C.OUTPUT_DIR / "complexity_profile_table.txt", "w", encoding="utf-8").write(txt)
    print("\n" + txt)
    print("\nsaved: complexity_profile.json, complexity_profile_table.txt")


if __name__ == "__main__":
    main()
