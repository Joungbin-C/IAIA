import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
CACHE_DIR = "./cache_fan_fast"     # slider / pump 로 변경 가능
SNR = "snr6"
IDS = ["id_00", "id_02", "id_04", "id_06"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASIC_DIM = 8
FEATURE_NAMES = [
    "RMS", "Kurtosis", "Skewness", "Peak-to-Peak",
    "Crest Factor", "Mean", "Std", "Entropy"
]

# DDAE model path (이미 학습된 것)
DDAE_BASIC_PATH = "./cache_fan_fast/ddae_basic_nf05.pt"

# =========================================================
# LOAD BASIC FEATURES (이미 extraction된 것 사용)
# =========================================================
def load_basic(kind):
    feats = []
    for id_name in IDS:
        path = os.path.join(CACHE_DIR, f"{SNR}_{id_name}_{kind}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        X = np.load(path)          # (N, 136)
        feats.append(X[:, -BASIC_DIM:])  # basic only
    return np.vstack(feats)        # (N_total, 8)

Xn_raw = load_basic("normal")
Xa_raw = load_basic("abnormal")

# =========================================================
# LOAD MIN–MAX (학습 시 사용한 것)
# =========================================================
# =========================================================
# COMPUTE MIN–MAX FROM snr6 NORMAL BASIC (ON THE FLY)
# =========================================================
def compute_basic_minmax():
    feats = []
    for id_name in IDS:
        path = os.path.join(CACHE_DIR, f"{SNR}_{id_name}_normal.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        X = np.load(path)          # (N, 136)
        feats.append(X[:, -BASIC_DIM:])

    Xall = np.vstack(feats)        # (N_total, 8)
    return Xall.min(axis=0), Xall.max(axis=0)

xmin, xmax = compute_basic_minmax()

def normalize(X):
    return np.clip((X - xmin) / (xmax - xmin + 1e-8), 0, 1)

Xn_norm = normalize(Xn_raw)
Xa_norm = normalize(Xa_raw)

# =========================================================
# DDAE MODEL
# =========================================================
class DDAE(nn.Module):
    def __init__(self, dims):
        super().__init__()
        enc, dec = [], []
        for a, b in zip(dims[:-1], dims[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        enc.pop()
        for a, b in zip(dims[::-1][:-1], dims[::-1][1:]):
            dec += [nn.Linear(a, b), nn.ReLU()]
        dec.pop()
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)
        self.out = nn.Sigmoid()

    def forward(self, x):
        return self.out(self.decoder(self.encoder(x)))

model = DDAE([8, 16, 32, 64, 128]).to(DEVICE)
model.load_state_dict(torch.load(DDAE_BASIC_PATH, map_location=DEVICE))
model.eval()

@torch.no_grad()
def denoise(X):
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    return model(Xt).cpu().numpy()

Xn_dn = denoise(Xn_norm)
Xa_dn = denoise(Xa_norm)

# =========================================================
# GLOBAL Y-SCALE (feature-wise)
# =========================================================
X_all = np.vstack([Xn_norm, Xn_dn, Xa_norm, Xa_dn])
ymin = X_all.min(axis=0)
ymax = X_all.max(axis=0)

# =========================================================
# VISUALIZATION
# =========================================================
plt.figure(figsize=(26, 12))

for d in range(BASIC_DIM):
    # ---------- Normal Raw ----------
    plt.subplot(4, BASIC_DIM, d + 1)
    plt.plot(Xn_norm[:, d], linewidth=0.7)
    plt.title(f"Raw | {FEATURE_NAMES[d]}")
    plt.ylim(ymin[d], ymax[d])
    plt.grid(True)

    # ---------- Normal DDAE ----------
    plt.subplot(4, BASIC_DIM, BASIC_DIM + d + 1)
    plt.plot(Xn_dn[:, d], linewidth=0.7)
    plt.title(f"DDAE | {FEATURE_NAMES[d]}")
    plt.ylim(ymin[d], ymax[d])
    plt.grid(True)

    # ---------- Abnormal Raw ----------
    plt.subplot(4, BASIC_DIM, 2*BASIC_DIM + d + 1)
    plt.plot(Xa_norm[:, d], linewidth=0.7, color="tab:red")
    plt.title(f"Raw | {FEATURE_NAMES[d]}")
    plt.ylim(ymin[d], ymax[d])
    plt.grid(True)

    # ---------- Abnormal DDAE ----------
    plt.subplot(4, BASIC_DIM, 3*BASIC_DIM + d + 1)
    plt.plot(Xa_dn[:, d], linewidth=0.7, color="tab:red")
    plt.title(f"DDAE | {FEATURE_NAMES[d]}")
    plt.ylim(ymin[d], ymax[d])
    plt.grid(True)

plt.tight_layout()
plt.show()
