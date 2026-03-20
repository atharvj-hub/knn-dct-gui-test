import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_csv(filepath):
    """Load a CSV dataset, split features and encoded target, and return numpy arrays."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CSV file not found: {filepath}") from exc
    except Exception as exc:
        raise ValueError(f"Unable to load CSV file '{filepath}': {exc}") from exc

    if df.empty:
        raise ValueError("The CSV file is empty.")

    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    valid_target_names = {"species", "target", "class"}
    target_col = next(
        (column for column in df.columns if column.strip().lower() in valid_target_names),
        None,
    )

    if target_col is None:
        raise ValueError(
            "The CSV file must contain a target column named one of: "
            "Species, species, target, or class."
        )

    X = df.drop(columns=[target_col]).to_numpy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(df[target_col])

    return X, y
