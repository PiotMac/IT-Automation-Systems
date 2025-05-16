from sklearn.svm import SVC
from perceptron import Perceptron, OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


def plot_tsne(X, y):
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)

    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(12, 8))
    for class_value in [0, 1, 2]:
        idx = y == class_value
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1],
                    color=colors[class_value], label=f"Class {class_value}", s=50)

    plt.title("TSNE Visualization")
    plt.xlabel("TSNE component 1")
    plt.ylabel("TSNE component 2")
    plt.legend()
    plt.grid(True)
    plt.savefig("tSNE/t-SNE_visualization.png")

    mean = X.mean(axis=0)
    std = X.std(axis=0)

    X_1 = (X - mean) / std
    X_embedded = tsne.fit_transform(X_1)

    plt.figure(figsize=(12, 8))
    for class_value in [0, 1, 2]:
        idx = y == class_value
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1],
                    color=colors[class_value], label=f"Class {class_value}", s=50)

    plt.title("TSNE Visualization After Standardization")
    plt.xlabel("TSNE component 1")
    plt.ylabel("TSNE component 2")
    plt.legend()
    plt.grid(True)
    plt.savefig("tSNE/t-SNE_visualization_std.png")

    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    X_2 = (X - min_vals) / (max_vals - min_vals)
    X_embedded = tsne.fit_transform(X_2)

    plt.figure(figsize=(12, 8))
    for class_value in [0, 1, 2]:
        idx = y == class_value
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1],
                    color=colors[class_value], label=f"Class {class_value}", s=50)

    plt.title("TSNE Visualization After Normalization")
    plt.xlabel("TSNE component 1")
    plt.ylabel("TSNE component 2")
    plt.legend()
    plt.grid(True)
    plt.savefig("tSNE/t-SNE_visualization_norm.png")



def plot_boundaries_scaling_and_non_scaling(X, y):
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    X_std = (X - mean) / std

    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    X_norm = (X - min_vals) / (max_vals - min_vals)

    X_tsne = pd.DataFrame(TSNE(n_components=2, random_state=42).fit_transform(X))
    X_tsne_std = pd.DataFrame(TSNE(n_components=2, random_state=42).fit_transform(X_std))
    X_tsne_norm = pd.DataFrame(TSNE(n_components=2, random_state=42).fit_transform(X_norm))

    plot_perceptron_decision_boundary(X_tsne, y, path='original')
    plot_perceptron_decision_boundary(X_tsne_std, y, path='std')
    plot_perceptron_decision_boundary(X_tsne_norm, y, path='norm')


def plot_perceptron_decision_boundary(X, y, learning_rate=0.01, epochs=150, path='original'):
    """
    Rysuje granice decyzyjne Perceptronu na zredukowanych do 2D danych.
    """
    svc_linear = lambda: SVC(kernel="linear", probability=True)
    svc_model = OneVsRestClassifier(svc_linear)
    perceptron_model = OneVsRestClassifier(Perceptron, learning_rate=learning_rate, epochs=epochs)

    directories_names = ["svc", "perceptron"]

    models = [
        svc_model,
        perceptron_model
    ]

    counter = 0
    for model in models:
        model.fit(X, y)

        h = 0.05
        x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
        y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Siatka punktów do predykcji
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(mesh_points)
        Z = np.array(Z).reshape(xx.shape)

        # Wykres
        plt.figure(figsize=(12, 8))
        cmap_background = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
        cmap_points = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background)

        scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=cmap_points, edgecolor="k")
        plt.xlabel("TSNE Component 1")
        plt.ylabel("TSNE Component 2")
        if path == 'original':
            plt.title(f"Decision Boundary Without Scaling")
        elif path == 'std':
            plt.title(f"Decision Boundary After Standardization")
        else:
            plt.title(f"Decision Boundary After Normalization")
        plt.legend(handles=scatter.legend_elements()[0],
                   labels=[f"Class {label}" for label in np.unique(y)])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"decision_boundaries/{directories_names[counter]}/perceptron_boundary_{path}.png")
        plt.close()

        counter += 1

