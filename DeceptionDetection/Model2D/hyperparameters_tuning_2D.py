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

# STAŁE
SR = 22050
# DURATION = 2.0
# STEP = 1.0
EPOCHS = 10
NO_CONV_LAYERS = 3
FILTER_SIZE = 128
KERNEL_SIZE = 7
DROPOUT_RATE = 0.1
DENSE_SIZE = 64
POOL_SIZE = 2

folders = {
    "edited_truthful": "../Edited clips/Truthful",
    "edited_lies": "../Edited clips/Deceptive"
}

# PARAMETRY DO TUNINGU 2D
n_mels_values = [64, 128, 256]
n_fft_values = [2048] # [1024, 2048]
hop_length_values = [512] #, 1024]
batch_sizes = [8, 16, 32]
durations = [2.0, 3.0, 4.0]
steps = [1.0, 1.5, 2.0]

augment_params = [
    {"noise": (0.001, 0.01), "pitch": (-1, 1), "stretch": (0.95, 1.05)},
    {"noise": (0.001, 0.015), "pitch": (-2, 2), "stretch": (0.9, 1.1)},
    {"noise": (0.005, 0.02), "pitch": (-3, 3), "stretch": (0.85, 1.15)},
    None
]


def split_into_segments(y, sr, duration, step):
    window = int(sr * duration)
    hop = int(sr * step)
    segments = []
    for start in range(0, len(y) - window + 1, hop):
        segments.append(y[start:start + window])
    if len(segments) == 0:
        segments.append(np.pad(y, (0, window - len(y))))
    return segments


def process_dataset(duration, step, n_mels, n_fft, hop_length, augment_settings=None):
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
                        segments = split_into_segments(y_audio, SR, duration, step)

                        for seg in segments:
                            if augment:
                                seg = augment(samples=seg, sample_rate=SR)

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


# Podstawowa Architektura CNN 2D
def build_base_cnn_2d(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))

    for i in range(NO_CONV_LAYERS):
        model.add(Conv2D(FILTER_SIZE, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(POOL_SIZE, POOL_SIZE)))
        model.add(Dropout(DROPOUT_RATE))

        # Conv2D(128, kernel_size=(7, 7), activation='relu', padding='same'),
        # BatchNormalization(),
        # MaxPooling2D(pool_size=(2, 2)),
        # Dropout(0.1),
        #
        # Conv2D(128, kernel_size=(7, 7), activation='relu', padding='same'),
        # BatchNormalization(),
        # MaxPooling2D(pool_size=(2, 2)),
        # Dropout(0.1),
        #
        # Conv2D(128, kernel_size=(7, 7), activation='relu', padding='same'),
        # BatchNormalization(),
        # MaxPooling2D(pool_size=(2, 2)),
        # Dropout(0.1),

    model.add(Flatten())
    model.add(Dense(DENSE_SIZE, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
        # Flatten(),
        # Dense(DENSE_SIZE, activation='relu'),
        # Dense(1, activation='sigmoid')

    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_evaluate_model_2d(X, y, batch_size, epochs):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_base_cnn_2d(input_shape=X.shape[1:])

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    y_pred = (model.predict(X_val) > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred)
    }


# Eksperymenty: tuning hiperparametrów 2D
results = []

for duration, step, n_mels, n_fft, hop_length, aug in product(durations, steps, n_mels_values, n_fft_values, hop_length_values, augment_params):

    X, y = process_dataset(
        duration=duration,
        step=step,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        augment_settings=aug
    )

    if X.size == 0:
        continue

    for batch_size in batch_sizes:
        metrics = train_evaluate_model_2d(
            X, y, batch_size=batch_size, epochs=EPOCHS
        )

        results.append({
            "duration": duration,
            "step": step,
            "N_MELS": n_mels,
            "N_FFT": n_fft,
            "HOP_LENGTH": hop_length,
            "augment": str(aug),
            "batch_size": batch_size,
            **metrics
        })

        print(f"duration={duration}, step={step},  MELS={n_mels}, FFT={n_fft}, HOP={hop_length}, aug={aug}, batch={batch_size} "
              f"--> acc={metrics['accuracy']:.3f}, prec={metrics['precision']:.3f}, rec={metrics['recall']:.3f}, f1={metrics['f1']:.3f}")

# Zapis do CSV
keys = results[0].keys()
with open('csv/cnn_2d_hyperparameter_tuning.csv', 'w', newline='') as f:
    dict_writer = csv.DictWriter(f, keys)
    dict_writer.writeheader()
    dict_writer.writerows(results)

print("Wyniki zapisane w cnn_2d_hyperparameter_tuning.csv")