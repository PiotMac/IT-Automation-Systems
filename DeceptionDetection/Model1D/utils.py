import os
import numpy as np
import librosa
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch


def get_all_file_paths(folders):
    """
    Zwraca listę krotek (ścieżka_do_pliku, etykieta).
    Nie wczytuje audio, tylko zbiera ścieżki.
    """
    file_list = []
    for label_name, folder_path in folders.items():
        # Ustalanie etykiety (0 = prawda, 1 = kłamstwo)
        label_val = 0 if "truth" in label_name.lower() else 1

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".wav"):
                    full_path = os.path.join(root, file)
                    file_list.append((full_path, label_val))
    return file_list


def create_dataset_from_file_list(file_list, SR, duration, step, N_MFCC, augment_settings=None):
    """
    Wczytuje pliki z podanej listy, tnie na segmenty i tworzy X, y.
    """
    X, y = [], []

    augment = None
    if augment_settings:
        augment = Compose([
            AddGaussianNoise(*augment_settings["noise"], p=0.5),
            PitchShift(*augment_settings["pitch"], p=0.5),
            TimeStretch(*augment_settings["stretch"], p=0.5)
        ])

    for file_path, label_val in file_list:
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
        except Exception as e:
            print(f"Błąd przy przetwarzaniu {file_path}: {e}")
            continue

    X = np.array(X)
    y = np.array(y)

    # Transpozycja dla Conv1D (Batch, Time, Features)
    if X.size > 0:
        X = np.transpose(X, (0, 2, 1))

    return X, y



def split_into_segments(y, sr, duration, step):
    window = int(sr * duration)
    hop = int(sr * step)
    segments = []

    for start in range(0, len(y) - window + 1, hop):
        segments.append(y[start:start + window])

    if len(segments) == 0:
        segments.append(np.pad(y, (0, window - len(y))))

    return segments



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