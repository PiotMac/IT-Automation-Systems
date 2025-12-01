import os
import csv
import numpy as np
import librosa
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from audiomentations import Compose, AddGaussianNoise, PitchShift, TimeStretch

folders = {
    "edited_truthful": "Edited clips/Truthful",
    "edited_lies": "Edited clips/Deceptive"
}


best_hyperparams = {
    "N_MFCC": 20,
    "duration": 2.0,
    "step": 1.0,
    "augment": None,
    "batch_size": 8
}

epochs = 20

def split_into_segments(y, sr, duration, step):
    window = int(sr * duration)
    hop = int(sr * step)
    segments = []

    for start in range(0, len(y) - window + 1, hop):
        segments.append(y[start:start + window])

    if len(segments) == 0:
        segments.append(np.pad(y, (0, window - len(y))))

    return segments


def process_dataset(SR, duration, step, N_MFCC, augment_settings=None):
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

    mean = np.mean(X, axis=(0, 2), keepdims=True)
    std = np.std(X, axis=(0, 2), keepdims=True)
    X = (X - mean) / (std + 1e-10)

    X = np.transpose(X, (0, 2, 1))  # (samples, time_steps, features)
    return X, y


def build_custom_cnn(input_shape, conv_layers=2, filters=32, kernel_size=5,
                     dropout_rate=0.1, dense_units=64, pool_size=2):
    model = Sequential()
    model.add(Input(shape=input_shape))

    for i in range(conv_layers):
        model.add(Conv1D(filters, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=pool_size))
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


def train_evaluate_custom_cnn(X, y, conv_layers, filters, kernel_size,
                              dropout_rate, dense_units, batch_size, pool_size=2, epochs=20):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_custom_cnn(
        input_shape=(X.shape[1], X.shape[2]),
        conv_layers=conv_layers,
        filters=filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
        pool_size=pool_size
    )

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, verbose=0)

    y_pred = (model.predict(X_val) > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred)
    }


X, y = process_dataset(
    SR=22050,
    duration=best_hyperparams["duration"],
    step=best_hyperparams["step"],
    N_MFCC=best_hyperparams["N_MFCC"],
    augment_settings=best_hyperparams["augment"]
)


conv_layers_options = [1, 2, 3]
filters_options = [32, 64, 128]
kernel_sizes = [3, 5, 7]
dropout_rates = [0.1, 0.2, 0.3]
dense_units_options = [32, 64, 128]
pool_sizes = [2, 3]

results_arch = []

for conv_layers, filters, kernel_size, dropout_rate, dense_units, pool_size in product(
        conv_layers_options, filters_options, kernel_sizes, dropout_rates, dense_units_options, pool_sizes
):
    metrics = train_evaluate_custom_cnn(
        X, y,
        conv_layers=conv_layers,
        filters=filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
        batch_size=best_hyperparams["batch_size"],
        pool_size=pool_size,
        epochs=epochs
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
          f"dense={dense_units}, pool={pool_size} --> acc={metrics['accuracy']:.3f}, f1={metrics['f1']:.3f}")


keys = results_arch[0].keys()
with open('cnn_architecture_tuning.csv', 'w', newline='') as f:
    dict_writer = csv.DictWriter(f, keys)
    dict_writer.writeheader()
    dict_writer.writerows(results_arch)

print("Wyniki tuningu architektury zapisane w cnn_architecture_tuning.csv")
