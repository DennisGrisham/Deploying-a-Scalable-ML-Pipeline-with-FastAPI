import pickle
from typing import Tuple, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


# ----------------------------- Training -----------------------------
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Trains a machine learning model and returns it.
    Uses a compact LogisticRegression baseline to keep model artifact small.
    """
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# ----------------------------- Metrics -----------------------------
def compute_model_metrics(y: np.ndarray, preds: np.ndarray) -> Tuple[float, float, float]:
    """
    Validates the trained model using precision, recall, and F1 (fbeta with beta=1).
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


# ----------------------------- Inference -----------------------------
def inference(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Run model inferences and return the predictions.
    """
    return model.predict(X)


# ----------------------------- Persistence -----------------------------
def save_model(model_or_encoder: Any, path: str) -> None:
    """
    Serializes a model or encoder to a file with pickle.
    """
    with open(path, "wb") as f:
        pickle.dump(model_or_encoder, f)


def load_model(path: str) -> Any:
    """
    Loads pickle file from `path` and returns it.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


# ----------------------------- Slice metrics -----------------------------
def performance_on_categorical_slice(
    data,
    column_name: str,
    slice_value,
    categorical_features,
    label: str,
    encoder,
    lb,
    model,
) -> Tuple[float, float, float]:
    """
    Computes precision/recall/f1 on a slice of the data defined by (column_name == slice_value).
    Uses the provided encoder/lb with training=False to ensure consistency with full-test processing.
    """
    sliced_df = data[data[column_name] == slice_value]
    if sliced_df.shape[0] == 0:
        return 0.0, 0.0, 0.0

    X_slice, y_slice, _, _ = process_data(
        sliced_df,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta

