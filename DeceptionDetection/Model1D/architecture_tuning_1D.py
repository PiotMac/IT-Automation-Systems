import numpy as np
import csv
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


best_hyperparams = {
    "N_MFCC": 20,
    "duration": 2.0,
    "step": 1.0,
    "augment": None,
    "batch_size": 8
}

conv_layers_options = [3]
filters_options = [32, 64, 128]
kernel_sizes = [3, 5, 7]
dropout_rates = [0.1, 0.2, 0.3]
dense_units_options = [32, 64, 128]
pool_sizes = [2, 3]


epochs = 50

# MODEL_OUT = "arch_cnn_1D_model.h5"
MEAN_FILE = "X_arch_MEAN.npy"
STD_FILE = "X_arch_STD.npy"


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


def train_evaluate_custom_cnn(X_train, X_val, y_train, y_val, conv_layers, filters, kernel_size,
                              dropout_rate, dense_units, batch_size, pool_size=2, epochs=20):
    if X_train.size == 0 or X_val.size == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    model = build_custom_cnn(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        conv_layers=conv_layers,
        filters=filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
        pool_size=pool_size
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=0,
        mode='min',
        restore_best_weights=True
    )

    # model_checkpoint = ModelCheckpoint(
    #     MODEL_OUT,
    #     monitor='val_loss',
    #     save_best_only=True,
    #     mode='min',
    #     verbose=0
    # )

    callbacks_list = [early_stopping]

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks_list)

    y_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred)
    }

X_raw, y = process_dataset(
    folders,
    SR=22050,
    duration=best_hyperparams["duration"],
    step=best_hyperparams["step"],
    N_MFCC=best_hyperparams["N_MFCC"],
    augment_settings=best_hyperparams["augment"]
)

if X_raw.size == 0:
    print("Nie wczytano żadnych danych. Sprawdź ścieżki do plików.")
    exit()

print("Podział na zbiór treningowy i walidacyjny...")
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
X_raw, y, test_size=0.2, stratify=y, random_state=42
)

print("Obliczanie i zapis statystyk normalizacyjnych...")
X_arch_MEAN = np.mean(X_train_raw, axis=(0, 1, 2), keepdims=True)
X_arch_STD = np.std(X_train_raw, axis=(0, 1, 2), keepdims=True)

np.save(MEAN_FILE, X_arch_MEAN)
np.save(STD_FILE, X_arch_STD)
print(f"Zapisano statystyki do {MEAN_FILE} i {STD_FILE}")

X_train = (X_train_raw - X_arch_MEAN) / (X_arch_STD + 1e-10)
X_val = (X_val_raw - X_arch_MEAN) / (X_arch_STD + 1e-10)

results_arch = []

for conv_layers, filters, kernel_size, dropout_rate, dense_units, pool_size in product(
        conv_layers_options, filters_options, kernel_sizes, dropout_rates, dense_units_options, pool_sizes
):
    metrics = train_evaluate_custom_cnn(
        X_train, X_val, y_train, y_val,
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
          f"dense={dense_units}, pool={pool_size} --> acc={metrics['accuracy']:.3f}, prec={metrics['precision']:.3f}, "
          f"rec={metrics['recall']:.3f}, f1={metrics['f1']:.3f}")


keys = results_arch[0].keys()
with open('csv/cnn_architecture_tuning.csv', 'w', newline='') as f:
    dict_writer = csv.DictWriter(f, keys)
    dict_writer.writeheader()
    dict_writer.writerows(results_arch)

print("Wyniki tuningu architektury zapisane w csv/cnn_architecture_tuning.csv")
