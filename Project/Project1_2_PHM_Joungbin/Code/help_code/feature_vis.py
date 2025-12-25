import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# =========================================================
# CONFIG
# =========================================================
CACHE_DIR = "./cache_slider_fast"
NORMAL_PATH = f"{CACHE_DIR}/snr6_id_06_normal.npy"
ABNORMAL_PATH = f"{CACHE_DIR}/snr6_id_06_abnormal.npy"
MINMAX_PATH = f"{CACHE_DIR}/minmax_mfcc.npy"
DDAE_PATH = f"{CACHE_DIR}/ddae_mfcc_nf05.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# DDAE
# =========================================================
class DDAE(nn.Module):
    def __init__(self, dims):
        super().__init__()
        enc = []
        for a, b in zip(dims[:-1], dims[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        enc.pop()
        self.encoder = nn.Sequential(*enc)

        dec = []
        for a, b in zip(dims[::-1][:-1], dims[::-1][1:]):
            dec += [nn.Linear(a, b), nn.ReLU()]
        dec.pop()
        self.decoder = nn.Sequential(*dec)

        self.out = nn.Sigmoid()

    def forward(self, x):
        return self.out(self.decoder(self.encoder(x)))

@torch.no_grad()
def denoise_batch(model, X):
    Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    return model(Xt).cpu().numpy()

# =========================================================
# MAIN
# =========================================================
def main():
    # ---------- load data ----------
    Xn = np.load(NORMAL_PATH)[:, :128]     # (N,128)
    Xa = np.load(ABNORMAL_PATH)[:, :128]

    xmin, xmax = np.load(MINMAX_PATH)

    Xn = np.clip((Xn - xmin) / (xmax - xmin + 1e-8), 0, 1)
    Xa = np.clip((Xa - xmin) / (xmax - xmin + 1e-8), 0, 1)

    # ---------- load DDAE ----------
    model = DDAE([128, 64, 32, 16, 8]).to(DEVICE)
    model.load_state_dict(torch.load(DDAE_PATH, map_location=DEVICE))
    model.eval()

    Xn_dn = denoise_batch(model, Xn)
    Xa_dn = denoise_batch(model, Xa)

    # ---------- global color scale ----------
    vmin = min(Xn.min(), Xa.min(), Xn_dn.min(), Xa_dn.min())
    vmax = max(Xn.max(), Xa.max(), Xn_dn.max(), Xa_dn.max())

    # =====================================================
    # VISUALIZATION
    # =====================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)

    im0 = axes[0,0].imshow(Xn.T, aspect="auto", origin="lower",
                           vmin=vmin, vmax=vmax, cmap="magma")
    axes[0,0].set_title("Normal – Raw MFCC")

    axes[0,1].imshow(Xa.T, aspect="auto", origin="lower",
                     vmin=vmin, vmax=vmax, cmap="magma")
    axes[0,1].set_title("Abnormal – Raw MFCC")

    axes[1,0].imshow(Xn_dn.T, aspect="auto", origin="lower",
                     vmin=vmin, vmax=vmax, cmap="magma")
    axes[1,0].set_title("Normal – After DDAE")

    axes[1,1].imshow(Xa_dn.T, aspect="auto", origin="lower",
                     vmin=vmin, vmax=vmax, cmap="magma")
    axes[1,1].set_title("Abnormal – After DDAE")

    for ax in axes.flat:
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("MFCC Channel")

    fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.9)
    fig.suptitle("MFCC Channel-wise Trends Across Samples", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()
