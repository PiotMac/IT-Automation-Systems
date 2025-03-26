import pandas as pd
import numpy as np
import random


class KNNClassifier:
    def __init__(self, k=3, metric=2.0):
        self.k = k
        self.metric = metric
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = {}
        for test_index, test_sample in x_test.iterrows():
            distances = []
            for train_index, train_sample in self.x_train.iterrows():
                distance = minkowski_distance(test_sample.values, train_sample.values, self.metric)
                distances.append([train_index, distance])

            distances.sort(key=lambda x: x[1])
            # Get indices of k nearest neighbors
            k_nearest_neighbors = [index for index, _ in distances[:self.k]]
            neighbor_classes = self.y_train.loc[k_nearest_neighbors].values
            unique_classes, counts = np.unique(neighbor_classes, return_counts=True)
            max_count = np.max(counts)
            most_common_classes = unique_classes[counts == max_count]
            predicted_class = random.choice(most_common_classes)

            predictions[test_index] = predicted_class

        return pd.Series(predictions)


def accuracy(y_true, y_pred):
    """Oblicza dokładność klasyfikacji."""
    assert y_true.index.equals(y_pred.index), "Indices of true and predicted labels do not match."

    correct_predictions = (y_true == y_pred).sum()

    accuracy_score = correct_predictions / len(y_pred)

    return accuracy_score


def precision(y_true, y_pred, average='macro'):
    """Oblicza precyzję. Obsługuje micro/macro averaging."""
    assert y_true.index.equals(y_pred.index), "Indices of true and predicted labels do not match."

    classes = np.unique(y_true)

    tp = {c: 0 for c in classes}  # True Positives per class
    fp = {c: 0 for c in classes}  # False Positives per class

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp[true] += 1  # Correct prediction (True Positive)
        else:
            fp[pred] += 1  # Wrong prediction (False Positive)

    if average == "micro":
        total_tp = sum(tp.values())
        total_fp = sum(fp.values())
        return total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0

    elif average == "macro":
        precisions = []
        for c in classes:
            if tp[c] + fp[c] > 0:
                precisions.append(tp[c] / (tp[c] + fp[c]))
            else:
                precisions.append(0.0)
        return np.mean(precisions)

    else:
        raise ValueError("Parameter 'average' has to be set to 'micro' or 'macro'.")


def recall(y_true, y_pred, average='macro'):
    """Oblicza recall. Obsługuje micro/macro averaging."""
    assert y_true.index.equals(y_pred.index), "Indices of true and predicted labels do not match."

    classes = np.unique(y_true)

    tp = {c: 0 for c in classes}  # True Positives per class
    fn = {c: 0 for c in classes}  # False Negatives per class

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp[true] += 1  # Correct prediction (True Positive)
        else:
            fn[true] += 1  # Wrong prediction (False Negative)

    if average == "micro":
        total_tp = sum(tp.values())
        total_fn = sum(fn.values())
        return total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    elif average == "macro":
        precisions = []
        for c in classes:
            if tp[c] + fn[c] > 0:
                precisions.append(tp[c] / (tp[c] + fn[c]))
            else:
                precisions.append(0.0)
        return np.mean(precisions)

    else:
        raise ValueError("Parameter 'average' has to be set to 'micro' or 'macro'.")


def f1(y_true, y_pred, average='macro'):
    """Oblicza F1-score. Obsługuje micro/macro averaging."""
    prec = precision(y_true, y_pred, average)
    rec = recall(y_true, y_pred, average)

    return 2.0 * (prec * rec) / (prec + rec) if (prec + rec) > 0.0 else 0.0


def confusion_matrix(y_true, y_pred):
    """Oblicza macierz pomyłek pomiędzy prawdziwymi i przewidywanymi etykietami."""
    y_true_aligned = y_true.loc[y_pred.index]
    classes = np.unique(y_true)

    label_to_index = {label: i for i, label in enumerate(classes)}

    cm = np.zeros((len(classes), len(classes)), dtype=int)

    for true_label, pred_label in zip(y_true_aligned, y_pred):
        i = label_to_index[true_label]  # Row index (actual)
        j = label_to_index[pred_label]  # Column index (predicted)
        cm[i, j] += 1

    return cm


def minkowski_distance(X, Y, p=2.0):
    """Oblicza odległość Minkowskiego pomiędzy dwoma punktami."""
    return np.linalg.norm(X - Y, ord=p)
