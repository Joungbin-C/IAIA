import os
import glob
import numpy as np
from tqdm import tqdm

import librosa
from scipy.stats import kurtosis, skew
from scipy.fftpack import dct

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, confusion_matrix

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns

# Config
ROOT = r"F:\MIMII Dataset"
CACHE_DIR = "./cache_fan_fast"
FIG_DIR = "./cm_fig"

MACHINE = "fan"
IDS = ["id_00", "id_02", "id_04", "id_06"]

SNR_DIRS = {
    "snr6": "6_dB",
    "snr0": "0_dB",
    "snr_6": "-6_dB",
}

SR = 16000
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NFS = [0.7]
EPOCHS = 100
BATCH = 64
LR = 1e-4

TABLE2 = {
    "id_00": {"ddae": 285, "ad_train": 794, "ad_test": 339},
    "id_02": {"ddae": 276, "ad_train": 769, "ad_test": 330},
    "id_04": {"ddae": 278, "ad_train": 773, "ad_test": 330},
    "id_06": {"ddae": 277, "ad_train": 770, "ad_test": 329},
}

# Feature Extraction
def extract_features(wav_path):
    y, _ = librosa.load(wav_path, sr=SR, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=512,
        win_length=int(0.025 * SR),
        hop_length=int(0.010 * SR),
        n_mels=128, power=2.0
    )
    logmel = np.log(mel + 1e-8)
    mfcc = dct(logmel, axis=0, norm="ortho").mean(axis=1)

    rms = np.sqrt(np.mean(y ** 2) + 1e-12)
    ptp = np.ptp(y)
    peak = np.max(np.abs(y)) + 1e-12
    crest = peak / (rms + 1e-12)

    hist, _ = np.histogram(y, bins=256, density=True)
    hist = hist[hist > 0]
    ent = -np.sum(hist * np.log2(hist + 1e-12))

    basic = np.array(
        [rms, kurtosis(y), skew(y), ptp, crest,
         np.mean(y), np.std(y), ent],
        dtype=np.float32,
    )

    return np.concatenate([mfcc.astype(np.float32), basic])

# Cache
def cache_path(snr, id_name, kind):
    return os.path.join(CACHE_DIR, f"{snr}_{id_name}_{kind}.npy")

def list_wavs(snr, id_name, kind):
    base = os.path.join(ROOT, f"{SNR_DIRS[snr]}_{MACHINE}", MACHINE, id_name, kind)
    return sorted(glob.glob(os.path.join(base, "*.wav")))

def load_or_extract(snr, id_name, kind):
    path = cache_path(snr, id_name, kind)
    if os.path.exists(path):
        return np.load(path)
    wavs = list_wavs(snr, id_name, kind)
    X = np.stack([extract_features(w) for w in wavs])
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(path, X)
    return X

# DDAE
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

def train_ddae(X, dims, nf):
    model = DDAE(dims).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    sigma = Xt.std(0, keepdim=True) + 1e-12

    for _ in range(EPOCHS):
        idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            xb = Xt[idx[i:i+BATCH]]
            xn = torch.clamp(xb + torch.randn_like(xb) * (nf * sigma), 0, 1)
            loss = ((model(xn) - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

@torch.no_grad()
def denoise(model, X):
    return model(torch.tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy()


# MAP Test
def map_test(X):
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    R = np.corrcoef(X, rowvar=False)
    eig, vec = np.linalg.eigh(R)
    eig, vec = eig[::-1], vec[:, ::-1]

    scores = []
    for m in range(1, min(20, X.shape[1])):
        Rr = R - vec[:, :m] @ np.diag(eig[:m]) @ vec[:, :m].T
        d = np.clip(np.diag(Rr), 1e-12, None)
        Rr = (Rr / np.sqrt(d)).T / np.sqrt(d)
        scores.append(np.mean((Rr - np.diag(np.diag(Rr))) ** 2))
    return np.argmin(scores) + 1


# Confusion Matrix
def cm_from_scores(score_tr, score_te, y_true, perc=95):
    th = np.percentile(score_tr, perc)
    y_pred = (score_te > th).astype(int)
    return confusion_matrix(y_true, y_pred)

def plot_cm(cm, title, save_path):
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Abnormal"],
        yticklabels=["Normal", "Abnormal"],
        cbar=False
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    # plt.show()
    plt.close()


# Main
def main():

    ref = [load_or_extract("snr6", i, "normal") for i in IDS]
    Xref = np.vstack(ref)
    xmin, xmax = Xref.min(0), Xref.max(0)

    data = {}
    for s in SNR_DIRS:
        data[s] = {}
        for i in IDS:
            Xn = (load_or_extract(s, i, "normal") - xmin) / (xmax - xmin + 1e-8)
            Xa = (load_or_extract(s, i, "abnormal") - xmin) / (xmax - xmin + 1e-8)
            data[s][i] = (Xn, Xa)

    Xddae = np.vstack([data["snr6"][i][0][:TABLE2[i]["ddae"]] for i in IDS])
    for nf in NFS:

        mfcc_ddae = train_ddae(Xddae[:, :128], [128, 64, 32, 16, 8], nf)
        basic_ddae = train_ddae(Xddae[:, 128:], [8, 16, 32, 64, 128], nf)

        for s in SNR_DIRS:

            aucs = {"IF": [], "OCSVM": [], "LOF": []}
            cms = {
                "IF": np.zeros((2, 2), dtype=int),
                "OCSVM": np.zeros((2, 2), dtype=int),
                "LOF": np.zeros((2, 2), dtype=int)
            }

            for i in IDS:
                cfg = TABLE2[i]
                Xn, Xa = data[s][i]

                Xn = np.hstack([
                    denoise(mfcc_ddae, Xn[:, :128]),
                    denoise(basic_ddae, Xn[:, 128:])
                ])
                Xa = np.hstack([
                    denoise(mfcc_ddae, Xa[:, :128]),
                    denoise(basic_ddae, Xa[:, 128:])
                ])

                ab = cfg["ad_test"] // 2
                no = cfg["ad_test"] - ab

                Xtr = Xn[:cfg["ad_train"]]
                Xte = np.vstack([
                    Xn[cfg["ad_train"]:cfg["ad_train"] + no],
                    Xa[:ab]
                ])
                Yte = np.array([0] * no + [1] * ab)

                nm = min(map_test(Xtr[:, :128]), Xtr.shape[0] - 1)
                nb = min(map_test(Xtr[:, 128:]), Xtr.shape[0] - 1)

                pm = PCA(n_components=nm).fit(Xtr[:, :128])
                pb = PCA(n_components=nb).fit(Xtr[:, 128:])

                print(
                    f"[PCA DIM] {MACHINE} | {s} | {i} | "
                    f"MFCC: {pm.n_components_}, "
                    f"Basic: {pb.n_components_}, "
                    f"Total: {pm.n_components_ + pb.n_components_}"
                )

                Ztr = np.hstack([
                    pm.transform(Xtr[:, :128]),
                    pb.transform(Xtr[:, 128:])
                ])
                Zte = np.hstack([
                    pm.transform(Xte[:, :128]),
                    pb.transform(Xte[:, 128:])
                ])

                ifm = IsolationForest(random_state=SEED).fit(Ztr)
                s_tr = -ifm.decision_function(Ztr)
                s_te = -ifm.decision_function(Zte)
                aucs["IF"].append(roc_auc_score(Yte, s_te))
                cms["IF"] += cm_from_scores(s_tr, s_te, Yte)

                svm = OneClassSVM(gamma="scale", nu=0.1).fit(Ztr)
                s_tr = -svm.decision_function(Ztr)
                s_te = -svm.decision_function(Zte)
                aucs["OCSVM"].append(roc_auc_score(Yte, s_te))
                cms["OCSVM"] += cm_from_scores(s_tr, s_te, Yte)

                lof = LocalOutlierFactor(novelty=True).fit(Ztr)
                s_tr = -lof.score_samples(Ztr)
                s_te = -lof.score_samples(Zte)
                aucs["LOF"].append(roc_auc_score(Yte, s_te))
                cms["LOF"] += cm_from_scores(s_tr, s_te, Yte)

            print(f"\n[RESULT AUC]  NF = {nf}")
            print("-" * 40)
            print("SNR     Model    AUC")

            for k in aucs:
                print(f"{s:<7} {k:<7} {np.mean(aucs[k]):.3f}")

            for k in cms:
                save_path = f"{FIG_DIR}/{MACHINE}_NF{nf}_{s}_{k}.png"
                plot_cm(cms[k],
                        title=f"{MACHINE} | NF={nf} | {s} | {k}",
                        save_path=save_path)


if __name__ == "__main__":
    main()
