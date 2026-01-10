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

# STAŁE
SR = 22050
EPOCHS = 50
NO_CONV_LAYERS = 3
FILTER_SIZE = 32
KERNEL_SIZE = 5
DROPOUT_RATE = 0.3
DENSE_SIZE = 32
POOL_SIZE = 2

folders = {
    "edited_truthful": "../Edited clips/Truthful",
    "edited_lies": "../Edited clips/Deceptive"
}

MEAN_FILE = "X_hyper_MEAN.npy"
STD_FILE = "X_hyper_STD.npy"

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


# Podstawowa Architektura CNN 2D
def build_base_cnn_2d(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))

    for i in range(NO_CONV_LAYERS):
        model.add(Conv2D(FILTER_SIZE, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(POOL_SIZE, POOL_SIZE)))
        model.add(Dropout(DROPOUT_RATE))


    model.add(Flatten())
    model.add(Dense(DENSE_SIZE, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_evaluate_model_2d(X_train, X_val, y_train, y_val, batch_size, epochs):
    if X_train.size == 0 or X_val.size == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    model = build_base_cnn_2d(input_shape=X_train.shape[1:])

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


all_files = get_all_file_paths(folders)
labels = [item[1] for item in all_files]

train_files, val_files = train_test_split(
    all_files, test_size=0.2, stratify=labels, random_state=42
)

# Eksperymenty: tuning hiperparametrów 2D
results = []

for duration, step, n_mels, n_fft, hop_length, aug in product(durations, steps, n_mels_values, n_fft_values, hop_length_values, augment_params):

    X_train_raw, y_train = create_dataset_from_file_list(
        train_files, SR, duration, step, n_mels, n_fft, hop_length, aug
    )

    X_val_raw, y_val = create_dataset_from_file_list(
        val_files, SR, duration, step, n_mels, n_fft, hop_length, augment_settings=None
    )

    X_MEAN = np.mean(X_train_raw, axis=(0, 1, 2), keepdims=True)
    X_STD = np.std(X_train_raw, axis=(0, 1, 2), keepdims=True)

    np.save(MEAN_FILE, X_MEAN)
    np.save(STD_FILE, X_STD)

    X_train = (X_train_raw - X_MEAN) / (X_STD + 1e-10)
    X_val = (X_val_raw - X_MEAN) / (X_STD + 1e-10)

    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)

    for batch_size in batch_sizes:
        metrics = train_evaluate_model_2d(
            X_train, X_val, y_train, y_val, batch_size=batch_size, epochs=EPOCHS
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

print("Wyniki zapisane w csv/cnn_2d_hyperparameter_tuning.csv")