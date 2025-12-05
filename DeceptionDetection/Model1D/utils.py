import os
import numpy as np
import librosa
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch


def split_into_segments(y, sr, duration, step):
    window = int(sr * duration)
    hop = int(sr * step)
    segments = []

    for start in range(0, len(y) - window + 1, hop):
        segments.append(y[start:start + window])

    if len(segments) == 0:
        segments.append(np.pad(y, (0, window - len(y))))

    return segments


def process_dataset(folders, SR, duration, step, N_MFCC, augment_settings=None):
    X, y = [], []

    augment = None
    if augment_settings:
        augment = Compose([
            AddGaussianNoise(*augment_settings["noise"], p=0.5),
            PitchShift(*augment_settings["pitch"], p=0.5),
            TimeStretch(*augment_settings["stretch"], p=0.5)
        ])

    for label, folder in folders.items():
        label_val = 0 if "truth" in label.lower() else 1

        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    try:
                        y_audio, _ = librosa.load(file_path, sr=SR, mono=True)
                        segments = split_into_segments(y_audio, sr=SR, duration=duration, step=step)

                        for seg in segments:
                            if augment:
                                seg = augment(samples=seg, sample_rate=SR)

                            mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=N_MFCC)
                            target_frames = int(SR * duration / 512)
                            mfcc = librosa.util.fix_length(mfcc, size=target_frames, axis=1)

                            X.append(mfcc)
                            y.append(label_val)

                    except:
                        continue

    X = np.array(X)
    y = np.array(y)

    X = np.transpose(X, (0, 2, 1))
    return X, y


def extract_features_from_audio(audio, SR, DURATION, STEP, N_MFCC, mean, std):
    segments = split_into_segments(audio, SR, DURATION, STEP)
    X_segments = []

    for seg in segments:
        mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=N_MFCC)
        target_frames = int(SR * DURATION / 512)
        mfcc = librosa.util.fix_length(mfcc, size=target_frames, axis=1)
        X_segments.append(mfcc)

    X_segments = np.array(X_segments)

    X_segments = (X_segments - mean) / (std + 1e-10)

    X_segments = np.transpose(X_segments, (0, 2, 1))
    return X_segments