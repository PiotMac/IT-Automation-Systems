import numpy as np
import pandas as pd


class OneVsRestClassifier:
    def __init__(self, base_estimator_class, **kwargs):
        self.base_estimator_class = base_estimator_class
        self.estimators = {}
        self.kwargs = kwargs

    def fit(self, X, y):
        classes = np.unique(y)
        for c in classes:
            # Tworzymy etykiety binarne: 1 dla klasy c, 0 dla innych
            y_binary = (y == c).astype(int)
            estimator = self.base_estimator_class(**self.kwargs)
            estimator.fit(X, y_binary)
            self.estimators[c] = estimator

    def predict(self, X):
        margins = {}

        for c, est in self.estimators.items():
            proba = est.predict_proba(X)

            # Jeśli predict_proba zwraca 2D (np. SVC), bierzemy tylko kolumnę dla klasy "1"
            if proba.ndim == 2:
                margins[c] = proba[:, 1]
            else:
                margins[c] = proba  # dla Perceptronu

        margins_df = pd.DataFrame(margins, index=X.index if isinstance(X, pd.DataFrame) else None)
        return margins_df.idxmax(axis=1)

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=50):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for epoch in range(self.epochs):
            total_error = 0.0
            predictions = []
            for i in range(n_samples):
                linear_output = np.dot(X_train.iloc[i], self.weights) + self.bias
                prediction = int(heaviside(linear_output))
                predictions.append(prediction)

                if prediction != y_train.iloc[i]:
                    error = y_train.iloc[i] - prediction
                    total_error += abs(error)

                    self.weights += self.learning_rate * error * X_train.iloc[i]
                    self.bias += self.learning_rate * error

    def predict(self, X_test):
        linear_output = self.predict_proba(X_test)
        predictions = heaviside(linear_output)

        if isinstance(X_test, pd.DataFrame):
            return pd.Series(predictions, index=X_test.index)
        else:
            return pd.Series(predictions)

    def predict_proba(self, X_test):
        return np.dot(X_test, self.weights) + self.bias


def heaviside(x):
    return np.where(x >= 0.0, 1, 0)


def accuracy(y_true, y_pred):
    """Oblicza dokładność klasyfikacji."""
    y_true_aligned = y_true.loc[y_pred.index]

    correct_predictions = (y_true_aligned == y_pred).sum()

    accuracy_score = correct_predictions / len(y_pred)

    return accuracy_score


def precision(y_true, y_pred, average='macro'):
    """Oblicza precyzję. Obsługuje micro/macro averaging."""
    y_true_aligned = y_true.loc[y_pred.index]

    classes = np.unique(y_true_aligned)

    tp = {c: 0 for c in classes}  # True Positives per class
    fp = {c: 0 for c in classes}  # False Positives per class

    for true, pred in zip(y_true_aligned, y_pred):
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
    y_true_aligned = y_true.loc[y_pred.index]

    classes = np.unique(y_true_aligned)

    tp = {c: 0 for c in classes}  # True Positives per class
    fn = {c: 0 for c in classes}  # False Negatives per class

    for true, pred in zip(y_true_aligned, y_pred):
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
