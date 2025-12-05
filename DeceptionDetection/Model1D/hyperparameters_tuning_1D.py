import csv
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from utils import process_dataset

folders = {
    "edited_truthful": "../Edited clips/Truthful",
    "edited_lies": "../Edited clips/Deceptive"
}

mfcc_values = [13, 20, 40]
durations = [2.0, 3.0, 4.0]
steps = [1.0, 1.5, 2.0]
batch_sizes = [8, 16, 32]

augment_params = [
    {"noise": (0.001, 0.01), "pitch": (-1, 1), "stretch": (0.95, 1.05)},
    {"noise": (0.001, 0.015), "pitch": (-2, 2), "stretch": (0.9, 1.1)},
    {"noise": (0.005, 0.02), "pitch": (-3, 3), "stretch": (0.85, 1.15)},
    None
]

SR = 22050
epochs = 50

MEAN_FILE = "X_hyper_MEAN.npy"
STD_FILE = "X_hyper_STD.npy"

NO_CONV_LAYERS = 3
FILTER_SIZE = 128
KERNEL_SIZE = 3
DROPOUT_RATE = 0.3
DENSE_SIZE = 64
POOL_SIZE = 3

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


def train_evaluate_model(X_train, X_val, y_train, y_val, batch_size, epochs):
    if X_train.size == 0 or X_val.size == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    model = build_custom_cnn(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        conv_layers=NO_CONV_LAYERS,
        filters=FILTER_SIZE,
        kernel_size=KERNEL_SIZE,
        dropout_rate=DROPOUT_RATE,
        dense_units=DENSE_SIZE,
        pool_size=POOL_SIZE
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=0,
        mode='min',
        restore_best_weights=True
    )

    callbacks_list = [early_stopping]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks_list
    )

    y_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred)
    }


results = []

for N_MFCC, duration, step, aug in product(mfcc_values, durations, steps, augment_params):

    X_raw, y = process_dataset(
        folders=folders,
        SR=SR,
        duration=duration,
        step=step,
        N_MFCC=N_MFCC,
        augment_settings=aug
    )

    if X_raw.size == 0:
        # print("Nie wczytano żadnych danych. Sprawdź ścieżki do plików.")
        exit()

    # print("Podział na zbiór treningowy i walidacyjny...")
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=42
    )

    # print("Obliczanie i zapis statystyk normalizacyjnych...")
    X_hyper_MEAN = np.mean(X_train_raw, axis=(0, 1, 2), keepdims=True)
    X_hyper_STD = np.std(X_train_raw, axis=(0, 1, 2), keepdims=True)

    np.save(MEAN_FILE, X_hyper_MEAN)
    np.save(STD_FILE, X_hyper_STD)
    # print(f"Zapisano statystyki do {MEAN_FILE} i {STD_FILE}")

    X_train = (X_train_raw - X_hyper_MEAN) / (X_hyper_STD + 1e-10)
    X_val = (X_val_raw - X_hyper_MEAN) / (X_hyper_STD + 1e-10)

    for batch_size in batch_sizes:

        metrics = train_evaluate_model(
            X_train, X_val, y_train, y_val, batch_size=batch_size, epochs=epochs
        )

        results.append({
            "N_MFCC": N_MFCC,
            "duration": duration,
            "step": step,
            "augment": str(aug),
            "batch_size": batch_size,
            **metrics
        })

        print(f"MFCC={N_MFCC}, dur={duration}, step={step}, aug={aug}, batch={batch_size} "
              f"--> acc={metrics['accuracy']:.3f}, prec={metrics['precision']:.3f}, "
              f"rec={metrics['recall']:.3f}, f1={metrics['f1']:.3f}")


# === Zapis do CSV ===
keys = results[0].keys()
with open('csv/cnn_hyperparameter_tuning.csv', 'w', newline='') as f:
    dict_writer = csv.DictWriter(f, keys)
    dict_writer.writeheader()
    dict_writer.writerows(results)

print("Wyniki zapisane w csv/cnn_hyperparameter_tuning.csv")
