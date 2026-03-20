from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import seaborn as sns


def train_and_evaluate(X, y, k=3):
    """
    Train and evaluate a Decision Tree and a k-Nearest Neighbors classifier.

    Scaling is required for kNN because it relies on distance calculations, so
    features with larger numeric ranges can dominate the result if they are not
    standardized. Decision Trees split on feature thresholds and are therefore
    not sensitive to feature scale in the same way.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    decision_tree = DecisionTreeClassifier(random_state=42)
    knn = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=k)),
        ]
    )

    models = {
        "knn": knn,
        "decision_tree": decision_tree,
    }

    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results[model_name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, predictions),
            "y_true": y_test,
            "y_pred": predictions,
        }

    return results


def plot_accuracy_comparison(results):
    """Plot a bar chart comparing KNN and Decision Tree accuracy."""
    model_labels = ["KNN", "Decision Tree"]
    accuracies = [
        results["knn"]["accuracy"],
        results["decision_tree"]["accuracy"],
    ]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_labels, accuracies, color=["#4C72B0", "#55A868"])
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison: KNN vs Decision Tree")
    plt.ylim(0, 1)

    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 0.01,
            f"{accuracy:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()



def plot_confusion_matrix(y_true, y_pred, title):
    """Plot a confusion matrix heatmap for a set of predictions."""
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()



def plot_all_comparisons(results):
    """Plot accuracy and confusion matrix comparisons for both models."""
    plot_accuracy_comparison(results)
    plot_confusion_matrix(
        results["knn"]["y_true"],
        results["knn"]["y_pred"],
        "KNN Confusion Matrix",
    )
    plot_confusion_matrix(
        results["decision_tree"]["y_true"],
        results["decision_tree"]["y_pred"],
        "Decision Tree Confusion Matrix",
    )
