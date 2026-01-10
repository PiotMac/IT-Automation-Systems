import os
import librosa
import numpy as np
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


def create_dataset_from_file_list(file_list, SR, duration, step, n_mels, n_fft, hop_length, augment_settings=None):
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

                mel_spec = librosa.feature.melspectrogram(
                    y=seg, sr=SR, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
                )
                log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                target_frames = int(SR * duration / hop_length)
                log_mel_spec = librosa.util.fix_length(log_mel_spec, size=target_frames, axis=1)

                X.append(log_mel_spec)
                y.append(label_val)

        except Exception as e:
            print(f"Błąd przy przetwarzaniu {file_path}: {e}")
            continue

    X = np.array(X)
    y = np.array(y)

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


def extract_features_from_audio(SR, y_audio, n_mels, n_fft, hop_length, duration, step, mean, std):
    segments = split_into_segments(y_audio, SR, duration, step)
    X_segments = []

    for seg in segments:
        mel_spec = librosa.feature.melspectrogram(
            y=seg, sr=SR, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        target_frames = int(SR * duration / hop_length)
        log_mel_spec = librosa.util.fix_length(log_mel_spec, size=target_frames, axis=1)
        X_segments.append(log_mel_spec)

    X_segments = np.array(X_segments)

    # Normalizacja przy użyciu statystyk treningowych
    X_segments = (X_segments - mean) / (std + 1e-10)

    # Dodanie wymiaru kanału (samples, freq, time, 1)
    X_segments = np.expand_dims(X_segments, axis=-1)
    return X_segments