import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BEST_PARAMS = {
    "conv_layers": 3,
    "filters": 64,
    "kernel_size": 5,
    "dropout_rate": 0.1,
    "dense_units": 128,
    "pool_size": 2
}

SR = 22050
N_MFCC = 20
DURATION = 2.0
STEP = 1.0
MODEL_OUT = "final_cnn_1D_model.h5"

folders = {
    "edited_truthful": "../Edited clips/Truthful",
    "edited_lies": "../Edited clips/Deceptive"
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

def extract_features_from_audio(audio):
    segments = split_into_segments(audio, SR, DURATION, STEP)
    X_segments = []

    for seg in segments:
        mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=N_MFCC)
        target_frames = int(SR * DURATION / 512)
        mfcc = librosa.util.fix_length(mfcc, size=target_frames, axis=1)
        X_segments.append(mfcc)

    X_segments = np.array(X_segments)
    mean = np.mean(X_segments, axis=(0, 2), keepdims=True)
    std = np.std(X_segments, axis=(0, 2), keepdims=True)
    X_segments = (X_segments - mean) / (std + 1e-10)
    X_segments = np.transpose(X_segments, (0, 2, 1))
    return X_segments

def load_dataset():
    X, y = [], []

    for label, folder in folders.items():
        label_val = 0 if "truth" in label.lower() else 1

        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".wav"):
                    path = os.path.join(root, file)
                    try:
                        audio, _ = librosa.load(path, sr=SR, mono=True)
                        X_segments = extract_features_from_audio(audio)
                        for seg in X_segments:
                            X.append(seg)
                            y.append(label_val)
                    except:
                        continue

    return np.array(X), np.array(y)


def build_final_cnn(input_shape, params):
    model = Sequential()
    model.add(Input(shape=input_shape))

    for i in range(params["conv_layers"]):
        model.add(Conv1D(
            params["filters"],
            kernel_size=params["kernel_size"],
            activation="relu",
            padding="same"
        ))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(params["pool_size"]))
        model.add(Dropout(params["dropout_rate"]))

    model.add(Flatten())
    model.add(Dense(params["dense_units"], activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def predict_file(model, file_path):
    audio, _ = librosa.load(file_path, sr=SR, mono=True)
    X_segments = extract_features_from_audio(audio)
    probs = model.predict(X_segments, verbose=0)
    mean_prob = probs.mean()
    predicted_class = 1 if mean_prob > 0.5 else 0
    return predicted_class, mean_prob


if __name__ == "__main__":
    print("Wczytywanie danych...")
    X, y = load_dataset()

    print("Podział na zbiór treningowy i walidacyjny...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Budowa modelu...")
    model = build_final_cnn(X_train.shape[1:], BEST_PARAMS)

    print("🚀 Trenowanie modelu...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=8,
        verbose=1
    )

    print("Ewaluacja modelu na zbiorze walidacyjnym (segmenty)...")
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)

    print("\n=================== WYNIKI DLA SEGMENTÓW ===================")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion matrix:")
    print(cm)

    print("\nEwaluacja modelu na poziomie całych plików...")
    file_paths = []
    file_labels = []
    for label, folder in folders.items():
        label_val = 0 if "truth" in label.lower() else 1
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".wav"):
                    file_paths.append(os.path.join(root, file))
                    file_labels.append(label_val)

    y_file_pred = []
    y_file_prob = []

    for path in file_paths:
        pred_class, prob = predict_file(model, path)
        y_file_pred.append(pred_class)
        y_file_prob.append(prob)

    acc_f = accuracy_score(file_labels, y_file_pred)
    prec_f = precision_score(file_labels, y_file_pred)
    rec_f = recall_score(file_labels, y_file_pred)
    f1_f = f1_score(file_labels, y_file_pred)
    cm_f = confusion_matrix(file_labels, y_file_pred)

    print("\n=================== WYNIKI DLA PLIKÓW CAŁKOWITYCH ===================")
    print(f"Accuracy : {acc_f:.4f}")
    print(f"Precision: {prec_f:.4f}")
    print(f"Recall   : {rec_f:.4f}")
    print(f"F1-score : {f1_f:.4f}")
    print("\nConfusion matrix:")
    print(cm_f)

    print("Zapis modelu do:", MODEL_OUT)
    model.save(MODEL_OUT)

    print("Gotowe! Model zapisano jako final_cnn_1D_model.h5")
