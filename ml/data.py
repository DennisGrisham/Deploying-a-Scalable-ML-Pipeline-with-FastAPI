import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X,
    categorical_features=None,
    label=None,
    training=True,
    encoder=None,
    lb=None,
):
    """
    Process the data used in the ML pipeline.

    One-hot encodes categorical features and binarizes labels.
    Works for both training and inference.

    Note: depending on the model, you may also want to scale continuous
    features.

    Inputs
    ------
    X : pd.DataFrame
        Data with features and (optionally) the label column.
    categorical_features : list[str]
        Names of categorical feature columns.
    label : str
        Name of the label column in X. If None, returns empty y array.
    training : bool
        If True, fit encoders on X; otherwise use provided encoders.
    encoder : sklearn.preprocessing.OneHotEncoder
        Trained OneHotEncoder (used when training=False).
    lb : sklearn.preprocessing.LabelBinarizer
        Trained LabelBinarizer (used when training=False).

    Returns
    -------
    X : np.ndarray
        Processed feature matrix.
    y : np.ndarray
        Processed labels if label is not None, else empty array.
    encoder : OneHotEncoder
        Fitted encoder if training=True; otherwise the input encoder.
    lb : LabelBinarizer
        Fitted binarizer if training=True; otherwise the input binarizer.
    """
    if categorical_features is None:
        categorical_features = []

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training:
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
        )
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        except AttributeError:
            # y is None at inference time
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb


def apply_label(inference):
    """Convert a single binary prediction into the string label."""
    if inference[0] == 1:
        return ">50K"
    elif inference[0] == 0:
        return "<=50K"
