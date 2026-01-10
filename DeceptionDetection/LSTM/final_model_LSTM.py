import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Dense, BatchNormalization, Input,
    LSTM, Reshape, Permute, Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import extract_features_from_audio, get_all_file_paths, create_dataset_from_file_list

BEST_PARAMS = {
    "conv_layers": 3,
    "filters": 64,
    "kernel_size": 7,
    "dropout_rate": 0.1,
    "dense_units": 64,
    "pool_size": 2,
    "lstm_units": 128,
    "lstm_layers": 2,
    "is_bidirectional": True
}

SR = 22050
EPOCHS = 50
CONST_MELS = 128
CONST_FFT = 2048
CONST_HOP_LENGTH = 512
CONST_BATCH_SIZE = 8
CONST_AUGMENT = None
DURATION = 2.0
STEP = 1.0

MODEL_OUT = "best_cnn_lstm_model.h5"
MEAN_FILE = "X_MEAN.npy"
STD_FILE = "X_STD.npy"

folders = {
    "edited_truthful": "../Edited clips/Truthful",
    "edited_lies": "../Edited clips/Deceptive"
}

def build_final_cnn_lstm(input_shape, arch_params):
    model = Sequential()
    model.add(Input(shape=input_shape))

    # 1. Część Konwolucyjna (Feature Extraction)
    for i in range(arch_params["conv_layers"]):
        model.add(Conv2D(arch_params["filters"], kernel_size=(arch_params["kernel_size"], arch_params["kernel_size"]), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(arch_params["pool_size"], arch_params["pool_size"])))
        model.add(Dropout(arch_params["dropout_rate"]))


    # 2. Most między CNN a LSTM (Reshape)
    model.add(Permute((2, 1, 3)))
    reshaped_dim = arch_params["filters"] * (input_shape[0] // (arch_params["pool_size"] ** arch_params["conv_layers"]))
    model.add(Reshape((-1, reshaped_dim)))

    # 3. Część Rekurencyjna (LSTM)
    for i in range(arch_params["lstm_layers"]):
        is_last_lstm = (i == arch_params["lstm_layers"] - 1)

        ret_seq = not is_last_lstm

        layer = LSTM(arch_params["lstm_units"], return_sequences=ret_seq, dropout=arch_params["dropout_rate"])

        if arch_params["is_bidirectional"]:
            model.add(Bidirectional(layer))
        else:
            model.add(layer)

    # 4. Klasyfikator
    model.add(Dense(arch_params["dense_units"], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def predict_file(model, file_path, mean, std):
    audio, _ = librosa.load(file_path, sr=SR, mono=True)
    X_segments = extract_features_from_audio(SR, audio, CONST_MELS, CONST_FFT, CONST_HOP_LENGTH, DURATION, STEP, mean, std)

    if X_segments.shape[0] == 0:
        return 0, 0.5

    probs = model.predict(X_segments, verbose=0)
    mean_prob = probs.mean()
    predicted_class = 1 if mean_prob > 0.5 else 0
    return predicted_class, mean_prob


if __name__ == "__main__":
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

    print("Budowa modelu...")
    model = build_final_cnn_lstm(X_train.shape[1:], BEST_PARAMS)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        MODEL_OUT,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    callbacks_list = [early_stopping, model_checkpoint]

    print("🚀 Trenowanie modelu...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=CONST_BATCH_SIZE,
        verbose=1,
        callbacks=callbacks_list
    )

    try:
        model = load_model(MODEL_OUT)
        print(f"\nUżyto najlepszego modelu zapisanego w {MODEL_OUT} do ewaluacji.")
    except Exception as e:
        print(f"\nBłąd ładowania najlepszego modelu ({MODEL_OUT}). Użyto stanu z ostatniej epoki.")

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
    y_file_true = []
    y_file_pred = []
    for file_path, label in val_files:
        pred_class, _ = predict_file(model, file_path, X_MEAN, X_STD)
        y_file_true.append(label)
        y_file_pred.append(pred_class)

    acc_f = accuracy_score(y_file_true, y_file_pred)
    prec_f = precision_score(y_file_true, y_file_pred)
    rec_f = recall_score(y_file_true, y_file_pred)
    f1_f = f1_score(y_file_true, y_file_pred)
    cm_f = confusion_matrix(y_file_true, y_file_pred)

    print("\n=================== WYNIKI DLA PLIKÓW CAŁKOWITYCH ===================")
    print(f"Accuracy : {acc_f:.4f}")
    print(f"Precision: {prec_f:.4f}")
    print(f"Recall   : {rec_f:.4f}")
    print(f"F1-score : {f1_f:.4f}")
    print("\nConfusion matrix:")
    print(cm_f)

    print(f"Najlepszy model został już zapisany przez ModelCheckpoint do: {MODEL_OUT}")