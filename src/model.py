"""The proposed MI-CNN-BiLSTM-Attention detector.

A Gaussian noise layer is inserted after the input when TRAIN_NOISE_STD > 0.
It is active during training only and is removed at inference, so the deployed
model is deterministic and carries no inference-time cost.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

import config as C

NUM_HEADS = getattr(C, "NUM_HEADS", 4)
KEY_DIM = getattr(C, "KEY_DIM", 32)
CLIPNORM = getattr(C, "CLIPNORM", 1.0)
TRAIN_NOISE_STD = getattr(C, "TRAIN_NOISE_STD", 0.0)


def categorical_focal_loss(gamma=C.FOCAL_GAMMA, alpha=C.FOCAL_ALPHA):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return loss_fn


def get_loss():
    loss_type = getattr(C, "LOSS_TYPE", "focal")
    if loss_type == "ce":
        return "categorical_crossentropy"
    elif loss_type == "focal":
        return categorical_focal_loss()
    else:
        raise ValueError(f"未知的 LOSS_TYPE: {loss_type},只能是 'ce' 或 'focal'")


def build_model(n_features, n_classes):
    inp = layers.Input(shape=(n_features,), name="input_features")


    # Active during training only; Keras disables it at inference, so the
    # deployed model is deterministic and identical in cost to the standard one.
    if TRAIN_NOISE_STD and TRAIN_NOISE_STD > 0:
        x = layers.GaussianNoise(float(TRAIN_NOISE_STD), name="input_gaussian_noise")(inp)
    else:
        x = inp

    x = layers.Reshape((n_features, 1))(x)

    x = layers.Conv1D(C.CONV1_FILTERS, C.KERNEL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv1D(C.CONV2_FILTERS, C.KERNEL_SIZE, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(C.DROPOUT)(x)

    x = layers.Bidirectional(layers.LSTM(C.LSTM_UNITS, return_sequences=True))(x)

    attn = layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM,
                                     name="multi_head_self_attention")(x, x)
    x = layers.Add(name="residual")([x, attn])
    x = layers.LayerNormalization(name="attn_norm")(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(C.DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(C.DROPOUT)(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = models.Model(inp, out, name="MI_CNN_BiLSTM_MHSA")


    if CLIPNORM:
        opt = optimizers.Adam(learning_rate=C.LEARNING_RATE, clipnorm=float(CLIPNORM))
    else:
        opt = optimizers.Adam(learning_rate=C.LEARNING_RATE)

    model.compile(optimizer=opt, loss=get_loss(), metrics=["accuracy"])
    return model


if __name__ == "__main__":
    m = build_model(n_features=C.K_FEATURES, n_classes=10)
    m.summary()
    print("总参数量:", m.count_params(),
          "| 损失:", getattr(C, "LOSS_TYPE", "focal"),
          "| clipnorm:", CLIPNORM,
          "| 训练加噪 std:", TRAIN_NOISE_STD)
