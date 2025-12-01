import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

SR = 22050
N_MFCC = 20
DURATION = 2.0
STEP = 1.0
MODEL_PATH = "final_cnn_1D_model.h5"
NEW_DATA_FOLDER = "New clips"

folders = {
    "truth": "New clips/Truthful",
    "lie": "New clips/Deceptive"
}


def split_into_segments(y, sr, duration, step):
    window = int(sr * duration)
    hop = int(sr * step)
    segments = []
    for start in range(0, len(y) - window + 1, hop):
        segments.append(y[start:start + window])

    if len(segments) == 0:
        segments.append(np.pad(y, (0, window - len(y))))
    return segments

def extract_mfcc_segments(path):
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

def predict_file(model, path):
    X = extract_mfcc_segments(path)
    y_pred_prob = model.predict(X, verbose=0)
    y_pred = int(np.mean(y_pred_prob) > 0.5)
    return y_pred

if __name__ == "__main__":
    print("Wczytywanie modelu...")
    model = load_model(MODEL_PATH)

    y_true, y_pred = [], []

    for label, folder in folders.items():
        label_val = 0 if label == "truth" else 1
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                path = os.path.join(folder, file)
                pred = predict_file(model, path)
                y_true.append(label_val)
                y_pred.append(pred)
                print(f"{file}: przewidziano {pred} (prawda=0, fałsz=1)")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=================== WYNIKI DLA NOWYCH NAGRAŃ ===================")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion matrix:")
    print(cm)
