import os
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
CACHE_DIR = "./cache_fan_fast"   # slider / pump 로 바꾸면 됨
IDS = ["id_00", "id_02", "id_04", "id_06"]
SNR = "snr6"

BASIC_DIM = 8
FEATURE_NAMES = [
    "RMS", "Kurtosis", "Skewness", "Peak-to-Peak",
    "Crest Factor", "Mean", "Std", "Entropy"
]

# =========================================================
# LOAD BASIC AUDIO (RAW)
# =========================================================
def load_basic(kind):
    X_list = []
    for id_name in IDS:
        path = os.path.join(CACHE_DIR, f"{SNR}_{id_name}_{kind}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        X = np.load(path)          # (N, 136)
        X_list.append(X[:, -BASIC_DIM:])  # basic only

    return np.vstack(X_list)       # (N_total, 8)

X_normal = load_basic("normal")
X_abnormal = load_basic("abnormal")

# =========================================================
# GLOBAL SCALE (feature-wise 동일 스케일)
# =========================================================
ymin = np.minimum(X_normal.min(axis=0), X_abnormal.min(axis=0))
ymax = np.maximum(X_normal.max(axis=0), X_abnormal.max(axis=0))

# =========================================================
# VISUALIZATION
# =========================================================
plt.figure(figsize=(24, 6))

for d in range(BASIC_DIM):
    # -------- Normal --------
    plt.subplot(2, BASIC_DIM, d + 1)
    plt.plot(X_normal[:, d], linewidth=0.8)
    plt.title(f"Normal | {FEATURE_NAMES[d]}")
    plt.ylim(ymin[d], ymax[d])
    plt.xlabel("Sample Index")
    if d == 0:
        plt.ylabel("Value")
    plt.grid(True)

    # -------- Abnormal --------
    plt.subplot(2, BASIC_DIM, BASIC_DIM + d + 1)
    plt.plot(X_abnormal[:, d], linewidth=0.8, color="tab:red")
    plt.title(f"Abnormal | {FEATURE_NAMES[d]}")
    plt.ylim(ymin[d], ymax[d])
    plt.xlabel("Sample Index")
    if d == 0:
        plt.ylabel("Value")
    plt.grid(True)

plt.tight_layout()
plt.show()
