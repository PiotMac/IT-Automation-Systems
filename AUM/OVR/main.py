from sklearn.datasets import load_wine
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import plots
from perceptron import (OneVsRestClassifier, Perceptron, accuracy, precision, recall, f1,
                        confusion_matrix)

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

test_size = 0.2
num_samples = X.shape[0]
num_test = int(num_samples * test_size)

indices = np.arange(num_samples)
np.random.shuffle(indices)

train_indices = indices[num_test:]
test_indices = indices[:num_test]

X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

# Standardization
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

# Min-max normalization
min_vals = X_train.min(axis=0)
max_vals = X_train.max(axis=0)
X_train_norm = (X_train - min_vals) / (max_vals - min_vals)
X_test_norm = (X_test - min_vals) / (max_vals - min_vals)

ovr = OneVsRestClassifier(Perceptron, learning_rate=0.01, epochs=50)
ovr.fit(X_train_scaled, y_train)

y_pred = ovr.predict(X_test_scaled)

svc_linear = lambda: SVC(kernel="linear", probability=True)

ovr_svc = OneVsRestClassifier(svc_linear)
ovr_svc.fit(X_train_scaled, y_train)

y_pred_svc = ovr_svc.predict(X_test_scaled)

predicted = [
    y_pred,
    y_pred_svc
]

names = [
    "################# PERCEPTRON ##################",
"##################### SVC #####################"
]

for i in range(0, len(names)):
    print(names[i])
    print("Accuracy:", accuracy(y_test, predicted[i]))

    print("Precision (macro):", precision(y_test, predicted[i], average='macro'))
    print("Recall (macro):", recall(y_test, predicted[i], average='macro'))
    print("F1-score (macro):", f1(y_test, predicted[i], average='macro'))

    print("Precision (micro):", precision(y_test, predicted[i], average='micro'))
    print("Recall (micro):", recall(y_test, predicted[i], average='micro'))
    print("F1-score (micro):", f1(y_test, predicted[i], average='micro'))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predicted[i]))

    if i != len(names) - 1:
        print("-----------------------------------------------")


plots.plot_tsne(X, y)
plots.plot_boundaries_scaling_and_non_scaling(X, y)