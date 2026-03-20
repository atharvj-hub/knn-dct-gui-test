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

    if "Species" not in df.columns:
        raise ValueError("The CSV file must contain a 'Species' target column.")

    X = df.drop(columns=["Species"]).to_numpy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Species"])

    return X, y