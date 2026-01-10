import visualkeras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import (Conv1D, Conv2D, MaxPooling1D, MaxPooling2D,
                                     Dense, Flatten, Dropout, BatchNormalization,
                                     LSTM, Reshape, Permute, Bidirectional)
from PIL import ImageFont
from collections import defaultdict

MODEL_OUT = 'visualization_CNN_LSTM.png'

# 1. Wczytaj model
print("Wczytywanie modelu...")
real_model = load_model('../LSTM/best_cnn_lstm_model.h5')

# ==================================================================
# CACHE KLAS - KLUCZ DO SUKCESU
# ==================================================================
# Tu będziemy trzymać stworzone już klasy naprawcze, żeby ich nie dublować
class_cache = {}


def create_patched_layer(original_layer):
    original_class = original_layer.__class__

    # 1. Sprawdzamy, czy już stworzyliśmy naprawioną klasę dla tego typu (np. Conv1D)
    if original_class not in class_cache:
        # Jeśli nie, to tworzymy ją TERAZ (tylko raz!)
        class PatchedLayer(original_class):
            def __init__(self, shape, name):
                self._output_shape = shape
                self._name = name

            @property
            def output_shape(self):
                return self._output_shape

            @property
            def name(self):
                return self._name

        # Podmieniamy nazwę, żeby w legendzie było ładnie
        PatchedLayer.__name__ = original_class.__name__

        # Zapisujemy do cache
        class_cache[original_class] = PatchedLayer

    # 2. Pobieramy klasę z cache
    PatchedClass = class_cache[original_class]

    # 3. Obliczamy kształt (logika Keras 3)
    try:
        if isinstance(original_layer.output, list):
            shape = original_layer.output[0].shape
        else:
            shape = original_layer.output.shape
    except AttributeError:
        shape = (None, 1, 1)

    # 4. Zwracamy instancję (wszystkie Conv1D będą teraz instancjami tej samej klasy!)
    return PatchedClass(shape, original_layer.name)


# Przetwarzanie warstw
print("Naprawianie warstw...")
fixed_layers = []
for layer in real_model.layers:
    try:
        patched = create_patched_layer(layer)
        fixed_layers.append(patched)
    except Exception as e:
        print(f"Pominięto: {e}")


class DummyModel:
    def __init__(self, layers):
        self.layers = layers


dummy_model = DummyModel(fixed_layers)

# ==================================================================
# 2. KOLORY - TERAZ MUSZĄ PASOWAĆ DO KLAS Z CACHE
# ==================================================================

# Twoje definicje kolorów dla ORYGINALNYCH klas
raw_colors = {
    Conv1D: '#f1c40f',  # Żółty
    Conv2D: '#f1c40f',
    LSTM: '#ff8000', # Pomarańczowy
    MaxPooling1D: '#e74c3c',  # Czerwony
    MaxPooling2D: '#e74c3c',
    Dense: '#3498db',  # Niebieski
    Flatten: '#9b59b6',  # Fioletowy
    Dropout: '#95a5a6',  # Szary
    BatchNormalization: '#2ecc71',  # Zielony
    Reshape: '#ff66b2', # Różowy
    Permute: '#000000', # Czarny
    Bidirectional: '#ffe5cc' # Beżowy
}

# Tłumaczymy to na mapę zrozumiałą dla visualkeras (używając naszych PatchedClass)
final_color_map = defaultdict(dict)

for original_cls, color in raw_colors.items():
    # Jeśli dana klasa była użyta w modelu (jest w cache), przypisz jej kolor
    if original_cls in class_cache:
        patched_cls = class_cache[original_cls]
        final_color_map[patched_cls]['fill'] = color

# ==================================================================

# 3. GENEROWANIE OBRAZKA
print("Generowanie wizualizacji...")

try:
    font = ImageFont.truetype("arial.ttf", 22)
except IOError:
    font = None

visualkeras.layered_view(
    dummy_model,
    to_file=MODEL_OUT,
    legend=True,
    draw_volume=True,
    color_map=final_color_map,  # Używamy przetłumaczonej mapy
    font=font,
    scale_xy=0.8,
    scale_z=1.0,
    max_xy=500,
    max_z=200,
    spacing=40
)

print(f"Gotowe! Stworzono plik {MODEL_OUT}")