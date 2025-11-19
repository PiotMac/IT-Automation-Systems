import os
import sys
import numpy as np
import librosa
from tensorflow.keras.models import load_model

MODEL_PATH = "final_cnn_1D_model.h5"

SR = 22050
N_MFCC = 20
DURATION = 2.0
STEP = 1.0

def split_into_segments(y, sr, duration, step):
    window = int(sr * duration)
    hop = int(sr * step)
    segments = []

    for start in range(0, len(y) - window + 1, hop):
        segments.append(y[start:start + window])

    if len(segments) == 0:
        segments.append(np.pad(y, (0, window - len(y))))

    return segments

def preprocess_audio(path):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    segments = split_into_segments(audio, SR, DURATION, STEP)
    X = []

    for seg in segments:
        mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=N_MFCC)
        target_frames = int(SR * DURATION / 512)
        mfcc = librosa.util.fix_length(mfcc, size=target_frames, axis=1)
        X.append(mfcc)

    X = np.array(X)

    mean = np.mean(X, axis=(0, 2), keepdims=True)
    std = np.std(X, axis=(0, 2), keepdims=True)
    X = (X - mean) / (std + 1e-10)
    X = np.transpose(X, (0, 2, 1))

    return X


def predict_file(path):
    if not os.path.exists(path):
        print("❌ Plik nie istnieje:", path)
        return


    model = load_model(MODEL_PATH)
    X = preprocess_audio(path)
    preds = model.predict(X)
    avg_pred = preds.mean()

    label = "Truth" if avg_pred <= 0.5 else "Lie"
    print(f"Predykcja dla pliku '{path}': {label} (średnia predykcja segmentów: {avg_pred:.3f})")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python3 predict_single.py <sciezka_do_pliku.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]
    predict_file(audio_path)
