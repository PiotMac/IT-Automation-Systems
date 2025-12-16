import csv
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Dense, BatchNormalization, Input,
    LSTM, Reshape, Permute, Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from utils import process_dataset


# STAŁE PARAMETRY DANYCH
SR = 22050
EPOCHS = 50
CONST_MELS = 128
CONST_FFT = 2048
CONST_HOP_LENGTH = 512
CONST_BATCH_SIZE = 8
CONST_AUGMENT = None
DURATION = 2.0
STEP = 1.0

MODEL_OUT = "arch_eern_lstm_model.h5"
MEAN_FILE = "X_arch_MEAN.npy"
STD_FILE = "X_arch_STD.npy"

folders = {
    "edited_truthful": "../Edited clips/Truthful",
    "edited_lies": "../Edited clips/Deceptive"
}

# === PARAMETRY DO TUNINGU ARCHITEKTURY (EERN) ===
# Parametry CNN (Ekstraktor cech)
conv_layers_options = [3]
filters_options = [128]
kernel_sizes = [3]
pool_sizes = [2]

# Parametry LSTM (Analiza sekwencji)
lstm_layers_options = [1, 2]
lstm_units_options = [32, 64, 128]
bidirectional_options = [True, False]
dropout_rates = [0.1, 0.2, 0.3]
dense_units_options = [32, 64, 128]



def build_eern_model(input_shape, conv_layers, filters, kernel_size,
                     dropout_rate, dense_units, pool_size,
                     lstm_units, lstm_layers, is_bidirectional):
    """
    Buduje architekturę CRNN (Convolutional Recurrent Neural Network).
    Conv2D ekstrahuje cechy -> Przekształcenie wymiarów -> LSTM analizuje czas.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))  # Oczekiwany kształt: (FREQ, TIME, 1)

    # 1. Część Konwolucyjna (Feature Extraction)
    for i in range(conv_layers):
        model.add(Conv2D(filters, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
        model.add(Dropout(dropout_rate))

    # W tym momencie mamy kształt: (Batch, New_Freq, New_Time, Filters)

    # 2. Most między CNN a LSTM (Reshape)
    # Musimy zamienić (Freq, Time, Filters) na (Time, Freq * Filters)
    # Permute przenosi wymiar czasu (index 2) na początek: (Time, Freq, Filters)
    # Keras Permute używa indeksowania od 1 (pomijając batch) -> (2, 1, 3)
    model.add(Permute((2, 1, 3)))

    # Teraz Reshape spłaszcza wymiar częstotliwości i filtrów w jeden wektor cech dla każdego kroku czasowego
    # shape[1] to teraz Time, shape[2] to Freq, shape[3] to Filters
    # Używamy -1, aby Keras sam obliczył rozmiar ostatniego wymiaru (Freq * Filters)
    model.add(Reshape((-1, filters * (input_shape[0] // (pool_size ** conv_layers)))))
    # Uwaga: Powyższe obliczenie w Reshape może być trudne dynamicznie,
    # bezpieczniejsza metoda w Keras to zostawienie inferencji kształtu warstwie:
    # Ale najprościej użyć warstwy Lambda lub po prostu TimeDistributed(Flatten)
    # Wróćmy do prostszego zapisu Reshape, który zadziała dynamicznie:

    # Prawidłowe dynamiczne podejście w Keras Functional API, ale w Sequential robimy trick:
    # Po Permute mamy (Time, Freq, Filters). Chcemy (Time, Freq*Filters).
    # Ponieważ rozmiar po poolingu jest stały dla danego inputu, możemy pozwolić Kerasowi zgadnąć:
    model.add(Reshape((-1, model.output_shape[-1] * model.output_shape[-2])))

    # 3. Część Rekurencyjna (LSTM)
    for i in range(lstm_layers):
        # Czy to ostatnia warstwa LSTM?
        is_last_lstm = (i == lstm_layers - 1)

        # Jeśli nie ostatnia, musi zwracać sekwencję (return_sequences=True) dla następnej warstwy LSTM
        # Jeśli ostatnia, zwraca wektor (return_sequences=False) do warstwy Dense
        ret_seq = not is_last_lstm

        layer = LSTM(lstm_units, return_sequences=ret_seq, dropout=dropout_rate)

        if is_bidirectional:
            model.add(Bidirectional(layer))
        else:
            model.add(layer)

    # model.add(LSTM(lstm_units, return_sequences=False))
    # model.add(Dropout(dropout_rate))

    # 4. Klasyfikator
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_evaluate_eern(X_train, X_val, y_train, y_val, conv_layers, filters, kernel_size,
                        dropout_rate, dense_units, pool_size,
                        lstm_units, lstm_layers, is_bidirectional,
                        batch_size, epochs=20):
    if X_train.size == 0 or X_val.size == 0:
        return {"accuracy": 0.0, "f1": 0.0}

    model = build_eern_model(
        input_shape=X_train.shape[1:],
        conv_layers=conv_layers, filters=filters, kernel_size=kernel_size,
        dropout_rate=dropout_rate, dense_units=dense_units, pool_size=pool_size,
        lstm_units=lstm_units, lstm_layers=lstm_layers, is_bidirectional=is_bidirectional
    )

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, verbose=0, mode='min', restore_best_weights=True
    )

    # ModelCheckpoint opcjonalnie wyłączony dla szybkości tuningu, lub nadpisuje ten sam plik
    # model_checkpoint = ModelCheckpoint(...)

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


# === GŁÓWNA PĘTLA PROGRAMU ===

print("Wczytywanie danych (stałe parametry cech)...")
X_raw, y = process_dataset(folders, SR, DURATION, STEP, CONST_MELS, CONST_FFT, CONST_HOP_LENGTH, CONST_AUGMENT)

if X_raw.size == 0:
    print("Brak danych.")
    exit()

print("Podział i normalizacja...")
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_raw, y, test_size=0.2, stratify=y, random_state=42
)

X_arch_MEAN = np.mean(X_train_raw, axis=(0, 1, 2), keepdims=True)
X_arch_STD = np.std(X_train_raw, axis=(0, 1, 2), keepdims=True)

np.save(MEAN_FILE, X_arch_MEAN)
np.save(STD_FILE, X_arch_STD)

X_train = (X_train_raw - X_arch_MEAN) / (X_arch_STD + 1e-10)
X_val = (X_val_raw - X_arch_MEAN) / (X_arch_STD + 1e-10)

# Dodanie wymiaru kanału (Batch, Freq, Time, 1)
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)

print(f"Rozpoczęcie tuningu EERN (CNN-LSTM) na {X_train.shape[0]} segmentach...")
results_arch = []

params_grid = list(product(conv_layers_options, filters_options, kernel_sizes, dropout_rates,
                           dense_units_options, pool_sizes, lstm_units_options,
                           lstm_layers_options, bidirectional_options))

for i, (conv_layers, filters, kernel_size, dropout_rate, dense_units,
        pool_size, lstm_units, lstm_layers, bidir) in enumerate(params_grid):

    metrics = train_evaluate_eern(
        X_train, X_val, y_train, y_val,
        conv_layers=conv_layers,
        filters=filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
        batch_size=CONST_BATCH_SIZE,
        pool_size=pool_size,
        lstm_units=lstm_units,
        lstm_layers=lstm_layers,
        is_bidirectional=bidir,
        epochs=EPOCHS
    )

    results_arch.append({
        "conv_layers": conv_layers,
        "filters": filters,
        "kernel_size": kernel_size,
        "dropout_rate": dropout_rate,
        "dense_units": dense_units,
        "pool_size": pool_size,
        "lstm_layers": lstm_layers,
        "lstm_units": lstm_units,
        "bidirectional": bidir,
        **metrics
    })

    print(f"conv={conv_layers}, filters={filters}, kernel={kernel_size}, dropout={dropout_rate}, "
          f"dense={dense_units}, pool={pool_size}, lstm_layers={lstm_layers}, lstm_units={lstm_units}, "
          f"bidirectional={bidir} --> acc={metrics['accuracy']:.3f}, prec={metrics['precision']:.3f}, "
          f"rec={metrics['recall']:.3f}, f1={metrics['f1']:.3f}")

# Zapis do CSV
keys = results_arch[0].keys()
with open('csv/eern_architecture_tuning.csv', 'w', newline='') as f:
    dict_writer = csv.DictWriter(f, keys)
    dict_writer.writeheader()
    dict_writer.writerows(results_arch)

print("\n\nWyniki tuningu EERN zapisane w csv/eern_architecture_tuning.csv")