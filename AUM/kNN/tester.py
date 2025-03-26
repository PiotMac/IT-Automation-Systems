import my_knn
from sklearn.datasets import load_wine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap


def k_fold_cross_validation(X_train, y_train, k_folds=5, k_neighbours=3, metric=2.0):
    num_samples = X_train.shape[0]

    fold_size = num_samples // k_folds
    accuracy_mean = 0.0

    for i in range(k_folds):
        start, end = fold_size * i, fold_size * (i + 1)
        X_validation, y_validation = X_train.iloc[start:end], y_train.iloc[start:end]

        X_train_rest = pd.concat([X_train.iloc[:start], X_train.iloc[end:]])
        y_train_rest = pd.concat([y_train.iloc[:start], y_train.iloc[end:]])

        kNN = my_knn.KNNClassifier(k=k_neighbours, metric=metric)
        kNN.fit(X_train_rest, y_train_rest)
        y_pred = kNN.predict(X_validation)
        accuracy_score = my_knn.accuracy(y_validation, y_pred)
        accuracy_mean += accuracy_score

    accuracy_mean = accuracy_mean / k_folds

    return accuracy_mean


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

no_folds = 5
acc_values = []
the_best_k = -1
the_best_accuracy = -1
p_values = [1.0, 1.5, 2.0, 3.0, 4.0]
the_best_p = -1.0
k_range = range(1, 21)
for k in k_range:
    for p in p_values:
        acc_for_k = k_fold_cross_validation(X_train, y_train, k_folds=no_folds, k_neighbours=k, metric=p)
        acc_values.append([k, p, acc_for_k])
        if acc_for_k > the_best_accuracy:
            the_best_k = k
            the_best_p = p
            the_best_accuracy = acc_for_k

final_knn = my_knn.KNNClassifier(k=the_best_k, metric=the_best_p)
final_knn.fit(X_train, y_train)
y_pred = final_knn.predict(X_test)
print(f"Final k-NN on testing set (k={the_best_k}, p={the_best_p}):")
print("Accuracy         :", my_knn.accuracy(y_test, y_pred))
print("Precision (macro):", my_knn.precision(y_test, y_pred, average='macro'))
print("Precision (micro):", my_knn.precision(y_test, y_pred, average='micro'))
print("Recall (macro)   :", my_knn.recall(y_test, y_pred, average='macro'))
print("Recall (micro)   :", my_knn.recall(y_test, y_pred, average='micro'))
print("F1 Score (macro) :", my_knn.f1(y_test, y_pred, average='macro'))
print("F1 Score (micro) :", my_knn.f1(y_test, y_pred, average='micro'))
print("Confusion matrix :")
print(my_knn.confusion_matrix(y_test, y_pred))

plt.figure(figsize=(10, 6))

for p in p_values:
    k_values = [k for k, p_val, _ in acc_values if p_val == p]
    errors = [1.0 - acc for _, p_val, acc in acc_values if p_val == p]

    plt.plot(k_values, errors, marker='o', linestyle='-', label=f"p = {p}")

plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Classification Error (1 - Accuracy)")
plt.title("KNN Classification Error for different k and p values")
plt.xticks(k_range)
plt.grid(True)
plt.legend()
plt.savefig("classification_error_for_p.png")

X_embedded = TSNE(n_components=2).fit_transform(X)

plt.figure(figsize=(8, 6))

for class_label in np.unique(y):
    plt.scatter(
        X_embedded[y == class_label, 0],
        X_embedded[y == class_label, 1],
        label=f"Class {class_label}",
    )

plt.xlabel("TSNE Component 1")
plt.ylabel("TSNE Component 2")
plt.title("t-SNE Visualization of Wine Dataset")
plt.legend()
plt.grid(True)
plt.savefig("TSNE.png")


def plot_decision_boundaries(X, y, k, metric):
    """
    Wizualizacja klasyfikacji na podstawie danych zredukowanych do 2D
    oraz wpływu parametrów k i metryki odległości.
    """
    knn = my_knn.KNNClassifier(k=k, metric=metric)
    knn.fit(X, y)

    h = 0.5
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = knn.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()]))
    Z = np.array(Z).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    cmap_background = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_points = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background)

    # Dodajemy punkty danych
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=cmap_points, edgecolor="k")
    plt.xlabel("TSNE Component 1")
    plt.ylabel("TSNE Component 2")
    plt.title(f"KNN Decision Boundaries (k={k}, p={metric})")
    plt.legend(handles=scatter.legend_elements()[0], labels=["Class " + str(label) for label in np.unique(y)])
    plt.grid(True)
    plt.savefig(f"decision_boundary_k{k}_p{metric}.png")


X_embedded = pd.DataFrame(TSNE(n_components=2).fit_transform(X))

for k in [1, 4, 11]:
    for p in p_values:
        print(f"Printing: k = {k}, p = {p}")
        plot_decision_boundaries(X_embedded, y, k, p)
