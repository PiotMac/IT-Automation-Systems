import os
import librosa
import numpy as np

def split_into_segments(y, sr, duration, step):
    window = int(sr * duration)
    hop = int(sr * step)
    segments = []

    for start in range(0, len(y) - window + 1, hop):
        segments.append(y[start:start + window])

    if len(segments) == 0:
        segments.append(np.pad(y, (0, window - len(y))))

    return segments

def process_dataset(folders, SR, duration, step, n_mels, n_fft, hop_length, augment_settings=None):
    X, y = [], []
    for label, folder in folders.items():
        label_val = 0 if "truth" in label.lower() else 1
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    try:
                        y_audio, _ = librosa.load(file_path, sr=SR, mono=True)
                        segments = split_into_segments(y_audio, SR, duration, step)

                        for seg in segments:
                            mel_spec = librosa.feature.melspectrogram(
                                y=seg, sr=SR, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
                            )
                            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                            target_frames = int(SR * duration / hop_length)
                            log_mel_spec = librosa.util.fix_length(log_mel_spec, size=target_frames, axis=1)

                            X.append(log_mel_spec)
                            y.append(label_val)

                    except Exception as e:
                        continue

    X = np.array(X)
    y = np.array(y)

    return X, y


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