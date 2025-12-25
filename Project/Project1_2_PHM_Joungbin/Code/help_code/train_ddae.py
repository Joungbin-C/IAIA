import os
import numpy as np
import torch
import torch.nn as nn

# =========================================================
# CONFIG
# =========================================================
CACHE_DIR = "./cache_pump_fast"   # pump면 cache_pump_fast
IDS = ["id_00", "id_02", "id_04", "id_06"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 50
BATCH = 256
LR = 1e-3
NF = 0.1

SAVE_PATH = os.path.join(CACHE_DIR, "ddae_mfcc_nf01.pt")

# =========================================================
# LOAD FEATURE (snr6 normal only)
# =========================================================
X_list = []

for id_name in IDS:
    path = os.path.join(CACHE_DIR, f"snr6_{id_name}_normal.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    X = np.load(path)        # (N, 136)
    X_list.append(X[:, :128])  # MFCC mean only

X = np.vstack(X_list)        # (N_total, 128)

# =========================================================
# MIN-MAX NORMALIZATION (이미 계산된 것 사용)
# =========================================================
xmin, xmax = np.load(os.path.join(CACHE_DIR, "minmax_mfcc.npy"))
X = (X - xmin) / (xmax - xmin + 1e-8)
X = np.clip(X, 0, 1)

X = torch.tensor(X, dtype=torch.float32).to(DEVICE)

# =========================================================
# DDAE MODEL
# =========================================================
class DDAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128)
        )
        self.out = nn.Sigmoid()

    def forward(self, x):
        return self.out(self.decoder(self.encoder(x)))

model = DDAE().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
sigma = X.std(0, keepdim=True) + 1e-12

# =========================================================
# TRAIN
# =========================================================
for ep in range(EPOCHS):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), BATCH):
        xb = X[idx[i:i+BATCH]]
        xn = torch.clamp(
            xb + torch.randn_like(xb) * (NF * sigma),
            0, 1
        )
        loss = ((model(xn) - xb) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    if (ep + 1) % 50 == 0:
        print(f"[Epoch {ep+1}/{EPOCHS}] loss = {loss.item():.6f}")

# =========================================================
# SAVE
# =========================================================
torch.save(model.state_dict(), SAVE_PATH)
print("Saved:", SAVE_PATH)
