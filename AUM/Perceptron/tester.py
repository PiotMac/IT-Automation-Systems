import perceptron
import my_plots
from ucimlrepo import fetch_ucirepo
import numpy as np

# fetch dataset
banknote_authentication = fetch_ucirepo(id=267)

# data (as pandas dataframes)
X = banknote_authentication.data.features
y = banknote_authentication.data.targets["class"]

# Plot t-SNE visualizations
my_plots.plot_tsne(X, y)

# Shuffle the dataset
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

X = X.iloc[indices]
y = y.iloc[indices]

# # 80% train, 20% test
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Standardization
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std
#
# Min-max normalization
min_vals = X_train.min(axis=0)
max_vals = X_train.max(axis=0)
X_train_norm = (X_train - min_vals) / (max_vals - min_vals)
X_test_norm = (X_test - min_vals) / (max_vals - min_vals)

# learning_rates = np.arange(0.1, 1.1, 0.1)
learning_rates = [0.001, 0.01, 0.1, 0.5, 0.9]
epochs_array = [5, 10, 30]
EPOCHS = 30

accuracies = []
precisions = []
recalls = []
f1_scores = []

accuracies_scaled = []
precisions_scaled = []
recalls_scaled = []
f1_scores_scaled = []

accuracies_norm = []
precisions_norm = []
recalls_norm = []
f1_scores_norm = []

acc_per_epoch = []
acc_per_epoch_std = []
acc_per_epoch_norm = []

precision_per_epoch = []
precision_per_epoch_std = []
precision_per_epoch_norm = []

recall_per_epoch = []
recall_per_epoch_std = []
recall_per_epoch_norm = []

f1_per_epoch = []
f1_per_epoch_std = []
f1_per_epoch_norm = []


# Testing different learning rates
for i in learning_rates:
    model = perceptron.Perceptron(learning_rate=i, epochs=EPOCHS)
    metrics = model.fit(X_train, y_train)

    acc_per_epoch.append(metrics[0])
    precision_per_epoch.append(metrics[1])
    recall_per_epoch.append(metrics[2])
    f1_per_epoch.append(metrics[3])

    y_pred = model.predict(X_test)

    acc = perceptron.accuracy(y_test, y_pred)
    prec = perceptron.precision(y_test, y_pred)
    rec = perceptron.recall(y_test, y_pred)
    f1 = perceptron.f1(y_test, y_pred)

    # Append metrics to lists
    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

    metrics = model.fit(X_train_scaled, y_train)

    acc_per_epoch_std.append(metrics[0])
    precision_per_epoch_std.append(metrics[1])
    recall_per_epoch_std.append(metrics[2])
    f1_per_epoch_std.append(metrics[3])

    y_pred = model.predict(X_test_scaled)

    acc = perceptron.accuracy(y_test, y_pred)
    prec = perceptron.precision(y_test, y_pred)
    rec = perceptron.recall(y_test, y_pred)
    f1 = perceptron.f1(y_test, y_pred)

    # Append metrics to lists
    accuracies_scaled.append(acc)
    precisions_scaled.append(prec)
    recalls_scaled.append(rec)
    f1_scores_scaled.append(f1)

    metrics = model.fit(X_train_norm, y_train)

    acc_per_epoch_norm.append(metrics[0])
    precision_per_epoch_norm.append(metrics[1])
    recall_per_epoch_norm.append(metrics[2])
    f1_per_epoch_norm.append(metrics[3])

    y_pred = model.predict(X_test_norm)

    acc = perceptron.accuracy(y_test, y_pred)
    prec = perceptron.precision(y_test, y_pred)
    rec = perceptron.recall(y_test, y_pred)
    f1 = perceptron.f1(y_test, y_pred)

    accuracies_norm.append(acc)
    precisions_norm.append(prec)
    recalls_norm.append(rec)
    f1_scores_norm.append(f1)

my_plots.plot_metrics_for_different_lr(learning_rates, accuracies, precisions, recalls, f1_scores)
my_plots.plot_metrics_vs_lr_separately(learning_rates, accuracies, precisions, recalls, f1_scores)

my_plots.plot_metrics_for_epochs(learning_rates, EPOCHS, acc_per_epoch, precision_per_epoch, recall_per_epoch,
                                 f1_per_epoch, acc_per_epoch_std, precision_per_epoch_std, recall_per_epoch_std,
                                 f1_per_epoch_std, acc_per_epoch_norm, precision_per_epoch_norm, recall_per_epoch_norm,
                                 f1_per_epoch_norm)

my_plots.plot_scaled_vs_non_scaled(learning_rates, accuracies, precisions, recalls, f1_scores,
                                   accuracies_scaled, precisions_scaled, recalls_scaled, f1_scores_scaled,
                                   accuracies_norm, precisions_norm, recalls_norm, f1_scores_norm)

my_plots.plot_boundaries_scaling_and_non_scaling(X, y)

model = perceptron.Perceptron(learning_rate=0.01, epochs=50)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

acc = perceptron.accuracy(y_test, y_pred)
prec = perceptron.precision(y_test, y_pred)
rec = perceptron.recall(y_test, y_pred)
f1_score = perceptron.f1(y_test, y_pred)

print("Accuracy:  ", acc)
print("Precision: ", prec)
print("Recall:    ", rec)
print("F1-score:  ", f1_score)
