"""Central configuration. Edit this file first, every script reads it.

The values below are the main configuration used for all headline results:
Edge-IIoTset, binary task, all features, cross-entropy, no training noise.
Set TRAIN_NOISE_STD = 0.1 and EXP_NOTE = "noise" to build the noise-aware model.
"""
from pathlib import Path


DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


ACTIVE_DATASET = "edge_iiot"
MODE = "binary"


EXP_NOTE = ""


DATASETS = {
    "ton_iot": {
        "csv_path": DATA_DIR / "Train_Test_Network.csv",
        "label_multiclass": "type",
        "label_binary": "label",
        "max_rows": None,
        "drop_cols": [
            "ts", "src_ip", "src_port", "dst_ip",
            "dns_query", "ssl_subject", "ssl_issuer",
            "http_uri", "http_referrer", "http_user_agent",
            "weird_name", "weird_addl", "weird_notice",
        ],
    },
    "edge_iiot": {
        "csv_path": DATA_DIR / "DNN-EdgeIIoT-dataset.csv",
        "label_multiclass": "Attack_type",
        "label_binary": "Attack_label",
        "max_rows": 200000,


        "drop_cols": [

            "frame.time", "ip.src_host", "ip.dst_host",
            "arp.src.proto_ipv4", "arp.dst.proto_ipv4",

            "tcp.checksum", "tcp.ack_raw",

            "tcp.srcport", "tcp.dstport",
            "tcp.options", "tcp.payload", "mqtt.msg",
        ],
    },
}


_note = EXP_NOTE if (EXP_NOTE == "" or EXP_NOTE.startswith("_")) else f"_{EXP_NOTE}"
RUN_TAG = f"{ACTIVE_DATASET}_{MODE}{_note}"


TEST_SIZE = 0.20
VAL_SIZE = 0.20
RANDOM_STATE = 42
K_FEATURES = 1000      # 1000 means "keep every feature"; lower it for the selection ablation


CONV1_FILTERS = 64
CONV2_FILTERS = 128
KERNEL_SIZE = 3
LSTM_UNITS = 64
DENSE_UNITS = 64
DROPOUT = 0.3
BATCH_SIZE = 256
EPOCHS = 60
LEARNING_RATE = 1e-3
CLIPNORM = None
EARLY_STOP_PATIENCE = 15
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 4


TRAIN_NOISE_STD = 0.0  # set to 0.05 or 0.1, with EXP_NOTE set accordingly, for the noise-aware models


USE_CLASS_WEIGHT = False
LOSS_TYPE = "ce"
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25


NUM_HEADS = 4
KEY_DIM = 32


RF_PARAMS = dict(n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt")


PGD_STEPS = 7
NOISE_REPEATS = 3
ROBUST_SAMPLE = 5000
