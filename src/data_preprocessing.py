"""Load, clean, encode, scale, split and select features.

The scaler and the mutual-information selector are fit on the training split
only. Returns a continuous_mask marking the features that robustness
experiments are allowed to perturb.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

import config as C


def resolve_labels(cfg):
    mode = getattr(C, "MODE", "multiclass")
    if mode == "binary":
        return cfg["label_binary"], cfg["label_multiclass"]
    elif mode == "multiclass":
        return cfg["label_multiclass"], cfg["label_binary"]
    else:
        raise ValueError(f"未知 MODE: {mode},只能是 'binary' 或 'multiclass'")


def load_and_clean(cfg):
    df = pd.read_csv(cfg["csv_path"], low_memory=False)
    df = df.replace(["-", " ", "", "NaN", "nan"], np.nan)

    label_col, other_label = resolve_labels(cfg)
    drop_cols = list(cfg["drop_cols"]) + [other_label]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = df.dropna(axis=1, how="all")

    if label_col not in df.columns:
        raise KeyError(f"标签列 {label_col} 不在数据里。当前 MODE={getattr(C,'MODE','?')}。"
                       f" 现有列(前 20):{list(df.columns)[:20]} ...")
    y_raw = df[label_col].astype(str).str.strip()
    X = df.drop(columns=[label_col])
    return X, y_raw


def encode_and_scale(X, y_raw):
    from pandas.api.types import is_numeric_dtype
    categorical_flags = []
    for col in X.columns:
        if is_numeric_dtype(X[col]):
            categorical_flags.append(False)
            continue

        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().mean() >= 0.5:
            X[col] = converted
            categorical_flags.append(False)
        else:
            X[col] = LabelEncoder().fit_transform(X[col].fillna("missing").astype(str))
            categorical_flags.append(True)

    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    return X.astype(float).values, y, label_encoder, np.array(categorical_flags)


def preprocess(cfg=None):
    if cfg is None:
        cfg = C.DATASETS[C.ACTIVE_DATASET]

    X_raw, y_raw = load_and_clean(cfg)


    max_rows = cfg.get("max_rows")
    if max_rows and len(y_raw) > max_rows:
        orig_n = len(y_raw)
        try:
            X_raw, _, y_raw, _ = train_test_split(
                X_raw, y_raw, train_size=max_rows,
                stratify=y_raw, random_state=C.RANDOM_STATE)
        except ValueError:
            idx = np.random.RandomState(C.RANDOM_STATE).choice(
                len(y_raw), size=max_rows, replace=False)
            X_raw, y_raw = X_raw.iloc[idx], y_raw.iloc[idx]
        print(f"[数据] 原始 {orig_n} 行 -> 分层抽样到 {max_rows} 行")

    X_all, y_all, label_encoder, cat_flags_full = encode_and_scale(X_raw, y_raw)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_all, y_all, test_size=C.TEST_SIZE,
        random_state=C.RANDOM_STATE, stratify=y_all)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=C.VAL_SIZE,
        random_state=C.RANDOM_STATE, stratify=y_temp)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    k = min(C.K_FEATURES, X_train.shape[1])
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train = selector.fit_transform(X_train, y_train)
    X_val = selector.transform(X_val)
    X_test = selector.transform(X_test)
    selected_idx = selector.get_support(indices=True)


    continuous_mask = ~cat_flags_full[selected_idx]

    print(f"[数据] 模式={getattr(C,'MODE','?')} | 类别数={len(label_encoder.classes_)} | "
          f"特征数={X_train.shape[1]}(其中连续={int(continuous_mask.sum())}) | "
          f"train={X_train.shape[0]} val={X_val.shape[0]} test={X_test.shape[0]}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "label_encoder": label_encoder,
        "selected_idx": selected_idx,
        "continuous_mask": continuous_mask,
        "scaler": scaler, "selector": selector,
    }


if __name__ == "__main__":
    data = preprocess()
    print("类别名:", list(data["label_encoder"].classes_))
    print("连续特征数 / 总特征数:",
          int(data["continuous_mask"].sum()), "/", len(data["continuous_mask"]))
