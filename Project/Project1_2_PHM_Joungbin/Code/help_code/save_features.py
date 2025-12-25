import os
import glob
import numpy as np
from tqdm import tqdm

import librosa
from scipy.stats import kurtosis, skew
from scipy.fftpack import dct

# =============================
# Config
# =============================
ROOT = r"F:\MIMII Dataset"
OUT_ROOT = "./features_all"

MACHINES = ["fan", "pump", "slider"]
IDS = ["id_00", "id_02", "id_04", "id_06"]

SNR_DIRS = {
    "snr6": "6_dB",
    "snr0": "0_dB",
    "snr_6": "-6_dB",
}

SR = 16000
SEED = 0
np.random.seed(SEED)

# =============================
# Feature extraction (136D)
# =============================
def extract_features(wav_path, sr=SR):
    y, _ = librosa.load(wav_path, sr=sr, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=512,
        win_length=int(0.025 * sr),
        hop_length=int(0.010 * sr),
        n_mels=128,
        power=2.0
    )
    logmel = np.log(mel + 1e-8)
    mfcc = dct(logmel, axis=0, norm="ortho").mean(axis=1)

    rms = np.sqrt(np.mean(y**2) + 1e-12)
    ptp = np.ptp(y)
    peak = np.max(np.abs(y)) + 1e-12
    crest = peak / (rms + 1e-12)

    hist, _ = np.histogram(y, bins=256, density=True)
    hist = hist[hist > 0]
    ent = -np.sum(hist * np.log2(hist + 1e-12))

    basic = np.array([
        rms, kurtosis(y), skew(y),
        ptp, crest, np.mean(y), np.std(y), ent
    ], dtype=np.float32)

    feat = np.concatenate([mfcc.astype(np.float32), basic])
    assert feat.shape == (136,)
    return feat

# =============================
# Helpers
# =============================
def list_wavs(machine, snr, id_name, kind):
    base = os.path.join(
        ROOT,
        f"{SNR_DIRS[snr]}_{machine}",
        machine,
        id_name,
        kind
    )
    return sorted(glob.glob(os.path.join(base, "*.wav")))

def save_npy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

# =============================
# Main
# =============================
def main():
    raw_dir  = f"{OUT_ROOT}/raw_features"
    norm_dir = f"{OUT_ROOT}/norm_features"
    prm_dir  = f"{OUT_ROOT}/norm_params"

    # -------------------------------------------------
    # 1) Feature extraction (ID별 저장)
    # -------------------------------------------------
    print("=== Feature Extraction (ID-wise) ===")
    for m in MACHINES:
        for s in SNR_DIRS:
            for i in IDS:
                feats_n, feats_a = [], []

                for w in list_wavs(m, s, i, "normal"):
                    feats_n.append(extract_features(w))
                for w in list_wavs(m, s, i, "abnormal"):
                    feats_a.append(extract_features(w))

                if not feats_n or not feats_a:
                    continue

                Xn = np.stack(feats_n)
                Xa = np.stack(feats_a)

                save_npy(f"{raw_dir}/{m}/{s}/{i}/normal.npy", Xn)
                save_npy(f"{raw_dir}/{m}/{s}/{i}/abnormal.npy", Xa)

                print(f"{m}-{s}-{i}: normal={len(Xn)}, abnormal={len(Xa)}")

    # -------------------------------------------------
    # 2) Global min–max (snr6 normal, ALL machines & IDs)
    # -------------------------------------------------
    print("\n=== Compute global min–max (snr6 normal) ===")
    all_norm = []

    for m in MACHINES:
        for i in IDS:
            path = f"{raw_dir}/{m}/snr6/{i}/normal.npy"
            if os.path.exists(path):
                all_norm.append(np.load(path))

    Xall = np.vstack(all_norm)
    xmin = Xall.min(axis=0)
    xmax = Xall.max(axis=0)

    save_npy(f"{prm_dir}/xmin.npy", xmin)
    save_npy(f"{prm_dir}/xmax.npy", xmax)

    # -------------------------------------------------
    # 3) Normalization & ID별 저장
    # -------------------------------------------------
    print("\n=== Normalize features (ID-wise) ===")
    for m in MACHINES:
        for s in SNR_DIRS:
            for i in IDS:
                for kind in ["normal", "abnormal"]:
                    src = f"{raw_dir}/{m}/{s}/{i}/{kind}.npy"
                    if not os.path.exists(src):
                        continue

                    X = np.load(src)
                    Xn = (X - xmin) / (xmax - xmin + 1e-8)
                    save_npy(f"{norm_dir}/{m}/{s}/{i}/{kind}.npy", Xn)

    print("\n=== ALL DONE (ID-wise, No DDAE) ===")

if __name__ == "__main__":
    main()
