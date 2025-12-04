import os
import csv
import numpy as np
import librosa
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch

# STAŁE PARAMETRY EKSTRAKCJI CECH
SR = 22050
EPOCHS = 20

# Stałe parametry dla ekstrakcji Mel-Spektrogramu (do użycia przed tuningiem cech)
CONST_MELS = 128
CONST_FFT = 2048
CONST_HOP_LENGTH = 512
CONST_BATCH_SIZE = 8
CONST_AUGMENT = None  # Brak augmentacji na etapie tuningu architektury (dla uproszczenia)
DURATION = 2.0
STEP = 1.0

folders = {
    "edited_truthful": "../Edited clips/Truthful",
    "edited_lies": "../Edited clips/Deceptive"
}

# PARAMETRY ARCHITEKTURY DO TUNINGU 2D
conv_layers_options = [2, 3]
filters_options = [64, 128]
kernel_sizes = [3, 5, 7]
dropout_rates = [0.1, 0.2, 0.3]
dense_units_options = [32, 64, 128]
pool_sizes = [2, 3]


def split_into_segments(y, sr, duration, step):
    window = int(sr * duration)
    hop = int(sr * step)
    segments = []
    for start in range(0, len(y) - window + 1, hop):
        segments.append(y[start:start + window])
    if len(segments) == 0:
        segments.append(np.pad(y, (0, window - len(y))))
    return segments


def process_dataset(n_mels, n_fft, hop_length, augment_settings=None):
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
                        segments = split_into_segments(y_audio, SR, DURATION, STEP)
                        for seg in segments:
                            if augment:
                                seg = augment(samples=seg, sample_rate=SR)
                            # --- Mel-Spektrogram (2D) ---
                            mel_spec = librosa.feature.melspectrogram(
                                y=seg, sr=SR, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
                            )
                            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                            target_frames = int(SR * DURATION / hop_length)
                            log_mel_spec = librosa.util.fix_length(log_mel_spec, size=target_frames, axis=1)
                            X.append(log_mel_spec)
                            y.append(label_val)
                    except:
                        continue
    X = np.array(X)
    y = np.array(y)
    mean = np.mean(X, axis=(0, 1, 2), keepdims=True)
    std = np.std(X, axis=(0, 1, 2), keepdims=True)
    X = (X - mean) / (std + 1e-10)
    X = np.expand_dims(X, axis=-1)
    return X, y


# Architektura CNN 2D dla tuningu
def build_custom_cnn_2d(input_shape, conv_layers, filters, kernel_size,
                        dropout_rate, dense_units, pool_size):
    model = Sequential()
    model.add(Input(shape=input_shape))

    for i in range(conv_layers):
        model.add(Conv2D(filters, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_evaluate_custom_cnn_2d(X, y, conv_layers, filters, kernel_size,
                                 dropout_rate, dense_units, batch_size, pool_size, epochs=20):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Sprawdzenie, czy jest wystarczająco danych
    if X_train.size == 0 or X_val.size == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    model = build_custom_cnn_2d(
        input_shape=X.shape[1:],
        conv_layers=conv_layers,
        filters=filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
        pool_size=pool_size
    )

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred)
    }


# PRZETWARZANIE DANYCH Z UŻYCIEM STAŁYCH PARAMETRÓW CECH
print("Wczytywanie i przetwarzanie danych dla tuningu architektury (stałe parametry cech)...")
X, y = process_dataset(
    n_mels=CONST_MELS,
    n_fft=CONST_FFT,
    hop_length=CONST_HOP_LENGTH,
    augment_settings=CONST_AUGMENT
)

if X.size == 0:
    print("Nie wczytano żadnych danych. Sprawdź ścieżki do plików.")
    exit()

# --- TUNING ARCHITEKTURY ---
print(f"Rozpoczęcie tuningu architektury 2D CNN na {X.shape[0]} segmentach...")
results_arch = []

for conv_layers, filters, kernel_size, dropout_rate, dense_units, pool_size in product(
        conv_layers_options, filters_options, kernel_sizes, dropout_rates, dense_units_options, pool_sizes
):
    metrics = train_evaluate_custom_cnn_2d(
        X, y,
        conv_layers=conv_layers,
        filters=filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
        batch_size=CONST_BATCH_SIZE,
        pool_size=pool_size,
        epochs=EPOCHS
    )

    results_arch.append({
        "conv_layers": conv_layers,
        "filters": filters,
        "kernel_size": kernel_size,
        "dropout_rate": dropout_rate,
        "dense_units": dense_units,
        "pool_size": pool_size,
        **metrics
    })

    print(f"conv={conv_layers}, filters={filters}, kernel={kernel_size}, dropout={dropout_rate}, "
          f"dense={dense_units}, pool={pool_size} --> acc={metrics['accuracy']:.3f}, prec={metrics['precision']:.3f}, "
          f"rec={metrics['recall']:.3f}, f1={metrics['f1']:.3f}")

keys = results_arch[0].keys()
with open('csv/cnn_2d_architecture_tuning.csv', 'w', newline='') as f:
    dict_writer = csv.DictWriter(f, keys)
    dict_writer.writeheader()
    dict_writer.writerows(results_arch)

print("Wyniki tuningu architektury 2D zapisane w cnn_2d_architecture_tuning.csv")