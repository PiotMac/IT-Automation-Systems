import csv
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from utils import get_all_file_paths, create_dataset_from_file_list

# STAŁE PARAMETRY EKSTRAKCJI CECH
SR = 22050
EPOCHS = 50

# Stałe parametry dla ekstrakcji Mel-Spektrogramu (do użycia przed tuningiem cech)
CONST_MELS = 128
CONST_FFT = 2048
CONST_HOP_LENGTH = 512
CONST_BATCH_SIZE = 8
CONST_AUGMENT = None
DURATION = 2.0
STEP = 1.0

folders = {
    "edited_truthful": "../Edited clips/Truthful",
    "edited_lies": "../Edited clips/Deceptive"
}

MEAN_FILE = "X_arch_MEAN.npy"
STD_FILE = "X_arch_STD.npy"

# PARAMETRY ARCHITEKTURY DO TUNINGU 2D
conv_layers_options = [1, 2, 3]
filters_options = [32, 64, 128]
kernel_sizes = [3, 5, 7]
dropout_rates = [0.1, 0.2, 0.3]
dense_units_options = [32, 64, 128]
pool_sizes = [2, 3]


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


def train_evaluate_custom_cnn_2d(X_train, X_val, y_train, y_val, conv_layers, filters, kernel_size,
                                 dropout_rate, dense_units, batch_size, pool_size, epochs=20):

    if X_train.size == 0 or X_val.size == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    model = build_custom_cnn_2d(
        input_shape=X_train.shape[1:],
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



print("Indeksowanie plików...")
all_files = get_all_file_paths(folders)
labels = [item[1] for item in all_files]

print("Podział plików na zbiory (Train/Val)...")
train_files, val_files = train_test_split(
    all_files, test_size=0.2, stratify=labels, random_state=42
)

print(f"Liczba plików treningowych: {len(train_files)}")
print(f"Liczba plików walidacyjnych: {len(val_files)}")

print("Generowanie segmentów treningowych...")
X_train_raw, y_train = create_dataset_from_file_list(
    train_files, SR, DURATION, STEP, CONST_MELS, CONST_FFT, CONST_HOP_LENGTH, CONST_AUGMENT
)

print("Generowanie segmentów walidacyjnych...")
X_val_raw, y_val = create_dataset_from_file_list(
    val_files, SR, DURATION, STEP, CONST_MELS, CONST_FFT, CONST_HOP_LENGTH, augment_settings=None
)

print("Obliczanie statystyk normalizacyjnych (tylko na Train)...")
X_MEAN = np.mean(X_train_raw, axis=(0, 1, 2), keepdims=True)
X_STD = np.std(X_train_raw, axis=(0, 1, 2), keepdims=True)

np.save(MEAN_FILE, X_MEAN)
np.save(STD_FILE, X_STD)

X_train = (X_train_raw - X_MEAN) / (X_STD + 1e-10)
X_val = (X_val_raw - X_MEAN) / (X_STD + 1e-10)

X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)

print(f"Kształt X_train: {X_train.shape}")
print(f"Kształt X_val: {X_val.shape}")

print(f"Rozpoczęcie tuningu architektury 2D CNN na {X_train.shape[0]} segmentach...")
results_arch = []

for conv_layers, filters, kernel_size, dropout_rate, dense_units, pool_size in product(
        conv_layers_options, filters_options, kernel_sizes, dropout_rates, dense_units_options, pool_sizes
):
    metrics = train_evaluate_custom_cnn_2d(
        X_train, X_val, y_train, y_val,
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
with open('csv_test/cnn_2d_architecture_tuning.csv', 'w', newline='') as f:
    dict_writer = csv.DictWriter(f, keys)
    dict_writer.writeheader()
    dict_writer.writerows(results_arch)

print("Wyniki tuningu architektury 2D zapisane w cnn_2d_architecture_tuning.csv")