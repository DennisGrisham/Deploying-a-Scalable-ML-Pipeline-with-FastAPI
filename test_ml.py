import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def small_data():
    data_path = Path(__file__).resolve().parent / "data" / "census.csv"
    df = pd.read_csv(data_path, nrows=500)  # small, fast
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["salary"]
    )

    X_train, y_train, encoder, lb = process_data(
        train_df, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    X_test, y_test, _, _ = process_data(
        test_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return X_train, y_train, X_test, y_test


def test_process_data_shapes(small_data):
    X_train, y_train, X_test, y_test = small_data
    assert X_train.ndim == 2 and X_test.ndim == 2
    assert y_train.ndim == 1 and y_test.ndim == 1
    assert X_train.shape[0] == y_train.shape[0] and X_train.shape[0] > 0
    assert X_test.shape[0] == y_test.shape[0] and X_test.shape[0] > 0


def test_train_and_inference(small_data):
    X_train, y_train, X_test, y_test = small_data
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (y_test.shape[0],)
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_known_values():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    p, r, f1 = compute_model_metrics(y_true, y_pred)
    assert np.isclose(p, 2 / 3)
    assert np.isclose(r, 2 / 3)
    assert np.isclose(f1, 2 / 3)

