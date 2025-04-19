import perceptron
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


def plot_tsne(X, y):
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X)

    colors = ['red', 'blue']
    plt.figure(figsize=(12, 8))
    for class_value in [0, 1]:
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

    colors = ['red', 'blue']
    plt.figure(figsize=(12, 8))
    for class_value in [0, 1]:
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

    colors = ['red', 'blue']
    plt.figure(figsize=(12, 8))
    for class_value in [0, 1]:
        idx = y == class_value
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1],
                    color=colors[class_value], label=f"Class {class_value}", s=50)

    plt.title("TSNE Visualization After Normalization")
    plt.xlabel("TSNE component 1")
    plt.ylabel("TSNE component 2")
    plt.legend()
    plt.grid(True)
    plt.savefig("tSNE/t-SNE_visualization_norm.png")


def plot_metrics_for_different_lr(learning_rates, accuracies, precisions, recalls, f1_scores):
    plt.figure(figsize=(12, 8))

    plt.plot(learning_rates, accuracies, label='Accuracy', marker='o')
    plt.plot(learning_rates, precisions, label='Precision', marker='p')
    plt.plot(learning_rates, recalls, label='Recall', marker='x')
    plt.plot(learning_rates, f1_scores, label='F1 Score', marker='*')

    plt.xlabel('Learning Rate')
    plt.ylabel('Metric Value')
    plt.title('Performance Metrics vs Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig("metrics_for_different_lr.png")
    plt.close()


def plot_metrics_for_epochs(learning_rates, EPOCHS, acc_per_epoch, precision_per_epoch, recall_per_epoch, f1_per_epoch,
                            acc_per_epoch_std, precision_per_epoch_std, recall_per_epoch_std, f1_per_epoch_std,
                            acc_per_epoch_norm, precision_per_epoch_norm, recall_per_epoch_norm, f1_per_epoch_norm):
    counter = 0
    for i in learning_rates:
        plt.figure(figsize=(12, 8))
        plt.plot(np.arange(1, EPOCHS + 1, 1), acc_per_epoch[counter], label='Non-scaled', marker='o', color='blue')
        plt.plot(np.arange(1, EPOCHS + 1, 1), acc_per_epoch_std[counter], label='Standardized', marker='o', color='red')
        plt.plot(np.arange(1, EPOCHS + 1, 1), acc_per_epoch_norm[counter], label='Normalized', marker='o',
                 color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy per Epoch for Learning Rate = {i:.3f}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"epochs/accuracy/accuracy_per_epoch_{i:.3f}.png")
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(np.arange(1, EPOCHS + 1, 1), precision_per_epoch[counter], label='Non-scaled', marker='o',
                 color='blue')
        plt.plot(np.arange(1, EPOCHS + 1, 1), precision_per_epoch_std[counter], label='Standardized', marker='o',
                 color='red')
        plt.plot(np.arange(1, EPOCHS + 1, 1), precision_per_epoch_norm[counter], label='Normalized', marker='o',
                 color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.title(f'Precision per Epoch for Learning Rate = {i:.3f}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"epochs/precision/precision_per_epoch_{i:.3f}.png")
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(np.arange(1, EPOCHS + 1, 1), recall_per_epoch[counter], label='Non-scaled', marker='o', color='blue')
        plt.plot(np.arange(1, EPOCHS + 1, 1), recall_per_epoch_std[counter], label='Standardized', marker='o',
                 color='red')
        plt.plot(np.arange(1, EPOCHS + 1, 1), recall_per_epoch_norm[counter], label='Normalized', marker='o',
                 color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.title(f'Recall per Epoch for Learning Rate = {i:.3f}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"epochs/recall/recall_per_epoch_{i:.3f}.png")
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(np.arange(1, EPOCHS + 1, 1), f1_per_epoch[counter], label='Non-scaled', marker='o', color='blue')
        plt.plot(np.arange(1, EPOCHS + 1, 1), f1_per_epoch_std[counter], label='Standardized', marker='o', color='red')
        plt.plot(np.arange(1, EPOCHS + 1, 1), f1_per_epoch_norm[counter], label='Normalized', marker='o', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.title(f'F1 Score per Epoch for Learning Rate = {i:.3f}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"epochs/f1_score/f1_per_epoch_{i:.3f}.png")
        plt.close()

        counter += 1


def plot_scaled_vs_non_scaled(learning_rates, accuracies, accuracies_scaled, accuracies_norm,
                              precisions, precisions_scaled, precisions_norm, recalls, recalls_scaled, recalls_norm,
                              f1_scores, f1_scores_scaled, f1_scores_norm):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(learning_rates, accuracies, label='Non-scaled', marker='o', color='blue')
    plt.plot(learning_rates, accuracies_scaled, label='Standardized', marker='o', color='red')
    plt.plot(learning_rates, accuracies_norm, label='Normalized', marker='o', color='green')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Scaled and Non-Scaled Data')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(learning_rates, precisions, label='Non-scaled', marker='p', color='blue')
    plt.plot(learning_rates, precisions_scaled, label='Standardized', marker='p', color='red')
    plt.plot(learning_rates, precisions_norm, label='Normalized', marker='p', color='green')
    plt.xlabel('Learning Rate')
    plt.ylabel('Precision')
    plt.title('Precision vs Scaled and Non-Scaled Data')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(learning_rates, recalls, label='Non-scaled', marker='x', color='blue')
    plt.plot(learning_rates, recalls_scaled, label='Standardized', marker='x', color='red')
    plt.plot(learning_rates, recalls_norm, label='Normalized', marker='x', color='green')
    plt.xlabel('Learning Rate')
    plt.ylabel('Recall')
    plt.title('Recall vs Scaled and Non-Scaled Data')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(learning_rates, f1_scores, label='Non-scaled', marker='*', color='blue')
    plt.plot(learning_rates, f1_scores_scaled, label='Standardized', marker='*', color='red')
    plt.plot(learning_rates, f1_scores_norm, label='Normalized', marker='*', color='green')
    plt.xlabel('Learning Rate')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Scaled and Non-Scaled Data')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("scaled_vs_non_scaled.png")
    plt.close()


def plot_metrics_vs_lr_separately(learning_rates, accuracies, precisions, recalls, f1_scores):
    # Plotting metrics vs learning rates
    plt.figure(figsize=(12, 8))

    # Accuracy plot
    plt.subplot(2, 2, 1)
    plt.plot(learning_rates, accuracies, label='Accuracy', marker='o', color='blue')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Learning Rate')
    plt.grid(True)

    # Precision plot
    plt.subplot(2, 2, 2)
    plt.plot(learning_rates, precisions, label='Precision', marker='p', color='orange')
    plt.xlabel('Learning Rate')
    plt.ylabel('Precision')
    plt.title('Precision vs Learning Rate')
    plt.grid(True)

    # Recall plot
    plt.subplot(2, 2, 3)
    plt.plot(learning_rates, recalls, label='Recall', marker='x', color='green')
    plt.xlabel('Learning Rate')
    plt.ylabel('Recall')
    plt.title('Recall vs Learning Rate')
    plt.grid(True)

    # F1 Score plot
    plt.subplot(2, 2, 4)
    plt.plot(learning_rates, f1_scores, label='F1 Score', marker='*', color='red')
    plt.xlabel('Learning Rate')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Learning Rate')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("metrics_separately.png")
    plt.close()


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

    for lr in [0.001, 0.01, 0.1, 0.5, 0.9]:
        for epochs in [10, 50, 100]:
            plot_perceptron_decision_boundary(X_tsne, y, learning_rate=lr, epochs=epochs, path='original')
            plot_perceptron_decision_boundary(X_tsne_std, y, learning_rate=lr, epochs=epochs, path='std')
            plot_perceptron_decision_boundary(X_tsne_norm, y, learning_rate=lr, epochs=epochs, path='norm')


def plot_perceptron_decision_boundary(X, y, learning_rate=0.1, epochs=10, path='original'):
    """
    Rysuje granice decyzyjne Perceptronu na zredukowanych do 2D danych.
    """
    model = perceptron.Perceptron(learning_rate, epochs)
    model.fit(X, y)

    h = 0.5
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
        plt.title(f"Perceptron Decision Boundary (lr={learning_rate}, epochs={epochs})")
    elif path == 'std':
        plt.title(f"Perceptron Decision Boundary After Standardization (lr={learning_rate}, epochs={epochs})")
    else:
        plt.title(f"Perceptron Decision Boundary After Normalization(lr={learning_rate}, epochs={epochs})")
    plt.legend(handles=scatter.legend_elements()[0],
               labels=[f"Class {label}" for label in np.unique(y)])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"decision_boundaries/{path}/perceptron_boundary_lr{learning_rate}_ep{epochs}.png")
    plt.close()

