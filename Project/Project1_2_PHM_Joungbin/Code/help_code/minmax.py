import os
import numpy as np

CACHE_DIR = "./cache_pump_fast"   # 또는 cache_slider_fast
IDS = ["id_00", "id_02", "id_04", "id_06"]

all_feats = []

for id_name in IDS:
    path = os.path.join(CACHE_DIR, f"snr6_{id_name}_normal.npy")
    X = np.load(path)             # (N,136)
    all_feats.append(X[:, :128])  # MFCC only

X = np.vstack(all_feats)          # (N_total, 128)

xmin = X.min(axis=0)
xmax = X.max(axis=0)

np.save(os.path.join(CACHE_DIR, "minmax_mfcc.npy"),
        np.stack([xmin, xmax]))

print("Saved minmax_mfcc.npy")
