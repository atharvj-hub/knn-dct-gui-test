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
        "Decision Tree": decision_tree,
        "KNN": knn,
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
        }

    return results