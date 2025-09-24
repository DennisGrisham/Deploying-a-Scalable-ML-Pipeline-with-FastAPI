from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Resolve repo root dynamically (folder where this file lives)
PROJECT_PATH = Path(__file__).resolve().parent
DATA_PATH = PROJECT_PATH / "data" / "census.csv"
MODEL_DIR = PROJECT_PATH / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading data from: {DATA_PATH}")

# Load the census.csv data
data = pd.read_csv(DATA_PATH)

# Split into train / test (stratify on label to preserve class balance)
train, test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["salary"]
)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Fit encoder/lb on TRAIN
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

# Transform TEST with same encoder/lb
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train model
model = train_model(X_train, y_train)

# Save model & encoder
model_path = PROJECT_PATH / "model" / "model.pkl"
encoder_path = PROJECT_PATH / "model" / "encoder.pkl"
save_model(model, str(model_path))
save_model(encoder, str(encoder_path))
print(f"Model saved to {model_path}")
print(f"Encoder saved to {encoder_path}")

# Reload model (sanity check)
model = load_model(str(model_path))

# Inference on test
preds = inference(model, X_test)

# Overall metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Slice metrics → slice_output.txt (fresh each run)
slice_output_path = PROJECT_PATH / "slice_output.txt"
with open(slice_output_path, "w") as f:
    f.write("Model performance on categorical slices\n")
    f.write("======================================\n")

for col in cat_features:
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        sp, sr, sfb = performance_on_categorical_slice(
            data=test,
            column_name=col,
            slice_value=slicevalue,
            categorical_features=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            model=model,
        )
        with open(slice_output_path, "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {sp:.4f} | Recall: {sr:.4f} | F1: {sfb:.4f}", file=f)

print(f"Slice metrics written to: {slice_output_path}")

