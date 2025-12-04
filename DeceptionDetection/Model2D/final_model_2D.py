import os
import csv
import numpy as np
import librosa
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch

BEST_PARAMS = {
    "conv_layers": 3,
    "filters": 128,
    "kernel_size": 7,
    "dropout_rate": 0.1,
    "dense_units": 64,
    "pool_size": 2
}

SR = 22050
EPOCHS = 20
CONST_MELS = 128
CONST_FFT = 2048
CONST_HOP_LENGTH = 512
CONST_BATCH_SIZE = 8
CONST_AUGMENT = None
DURATION = 2.0
STEP = 1.0

MODEL_OUT = "final_cnn_2D_model.h5"

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

def extract_features_from_audio(y_audio, n_mels, n_fft, hop_length, duration, step):
    segments = split_into_segments(y_audio, SR, duration, step)
    X_segments = []

    for seg in segments:
        # Ekstrakcja Mel-Spektrogramu (2D)
        mel_spec = librosa.feature.melspectrogram(
            y=seg, sr=SR, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Zapewnienie stałej długości ramki czasowej
        target_frames = int(SR * duration / hop_length)
        log_mel_spec = librosa.util.fix_length(log_mel_spec, size=target_frames, axis=1)
        X_segments.append(log_mel_spec)

    X_segments = np.array(X_segments)

    mean = np.mean(X_segments, axis=(0, 1, 2), keepdims=True)
    std = np.std(X_segments, axis=(0, 1, 2), keepdims=True)
    X_segments = (X_segments - mean) / (std + 1e-10)

    # Dodanie wymiaru kanału (samples, freq, time, 1)
    X_segments = np.expand_dims(X_segments, axis=-1)
    return X_segments

def process_dataset(duration, step, n_mels, n_fft, hop_length, augment_settings=None):
    X, y = [], []

    # augment = None
    # if augment_settings:
    #     augment = Compose([
    #         AddGaussianNoise(*augment_settings["noise"], p=0.5),
    #         PitchShift(*augment_settings["pitch"], p=0.5),
    #         TimeStretch(*augment_settings["stretch"], p=0.5)
    #     ])

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
                            # if augment:
                            #     seg = augment(samples=seg, sample_rate=SR)

                            #  Ekstrakcja Mel-Spektrogramu (2D)
                            mel_spec = librosa.feature.melspectrogram(
                                y=seg, sr=SR, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
                            )
                            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

                            # Zapewnienie stałej długości ramki czasowej
                            target_frames = int(SR * duration / hop_length)
                            log_mel_spec = librosa.util.fix_length(log_mel_spec, size=target_frames, axis=1)

                            X.append(log_mel_spec)
                            y.append(label_val)

                    except Exception as e:
                        # print(f"Error processing {file_path}: {e}")
                        continue

    X = np.array(X)
    y = np.array(y)

    # Normalizacja
    mean = np.mean(X, axis=(0, 1, 2), keepdims=True)
    std = np.std(X, axis=(0, 1, 2), keepdims=True)
    X = (X - mean) / (std + 1e-10)

    # Dodanie wymiaru kanału (samples, freq, time, 1)
    X = np.expand_dims(X, axis=-1)
    return X, y

def build_final_cnn_2d(input_shape, arch_params):
    model = Sequential()
    model.add(Input(shape=input_shape))

    for i in range(arch_params["conv_layers"]):
        model.add(Conv2D(arch_params["filters"], kernel_size=(arch_params["kernel_size"], arch_params["kernel_size"]), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(arch_params["pool_size"], arch_params["pool_size"])))
        model.add(Dropout(arch_params["dropout_rate"]))

    model.add(Flatten())
    model.add(Dense(arch_params["dense_units"], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def predict_file(model, file_path):
    audio, _ = librosa.load(file_path, sr=SR, mono=True)
    X_segments = extract_features_from_audio(audio, CONST_MELS, CONST_FFT, CONST_HOP_LENGTH, DURATION, STEP)
    probs = model.predict(X_segments, verbose=0)
    mean_prob = probs.mean()
    predicted_class = 1 if mean_prob > 0.5 else 0
    return predicted_class, mean_prob


if __name__ == "__main__":
    print("Wczytywanie danych...")
    X, y = process_dataset(DURATION, STEP, CONST_MELS, CONST_FFT, CONST_HOP_LENGTH, CONST_AUGMENT)

    print("Podział na zbiór treningowy i walidacyjny...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("Budowa modelu...")
    model = build_final_cnn_2d(X_train.shape[1:], BEST_PARAMS)

    print("🚀 Trenowanie modelu...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=CONST_BATCH_SIZE,
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
