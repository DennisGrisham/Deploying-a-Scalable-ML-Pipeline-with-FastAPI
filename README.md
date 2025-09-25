# Deploying a Scalable ML Pipeline with FastAPI

GitHub Repository: [DennisGrisham/Deploying-a-Scalable-ML-Pipeline-with-FastAPI](https://github.com/DennisGrisham/Deploying-a-Scalable-ML-Pipeline-with-FastAPI)

This project demonstrates an end-to-end ML pipeline built with FastAPI. It trains a model on the Census Income dataset, evaluates its performance on feature slices, and serves predictions through a REST API with unit tests and CI/CD.


---

## Environment Setup

This project was developed and tested with **Python 3.10** in a Conda environment named **`fastapi`**.
To reproduce the environment:

```bash
conda create -n fastapi python=3.10 -y
conda activate fastapi
pip install -r requirements.txt
```

Once installed, you can confirm the Python and Pandas versions with:

```bash
which python
python --version
python -c "import pandas as pd; print(pd.__version__)"
```

Expected output (or very close):

```bash
/home/<user>/miniconda3/envs/fastapi/bin/python
Python 3.10.x
2.2.x
```


---

## Data

The dataset used is `data/census.csv` which is already included in the repository. 
This file contains demographic information from the U.S. Census Bureau, along with the target label: whether an individual’s income is `<=50K` or `>50K`.

You do not need to clean or modify the file manually.
The script **`ml/data.py`** provides the `process_data` function, which:
- Encodes categorical features with a `OneHotEncoder`.
- Binarizes the target label with a `LabelBinarizer`.
- Splits features into numerical and categorical parts automatically.

The training pipeline (`train_model.py`) calls this function directly. Running:

```bash
python train_model.py
```

will process the data and train the model.


---

## Training the Model

To train the model and generate all required artifacts, run:

```bash
python train_model.py
```

This script will:
1. Load the dataset from data/census.csv.
2. Split the data into training and test sets.
3. Process categorical and numerical features using ml/data.py.
4. Train a Logistic Regression model on the training data.
5. Save the following artifacts into the model/ folder:
    - `model.pkl` — trained Logistic Regression model
    - `encoder.pkl` — fitted OneHotEncoder
6. Evaluate the model on the test set and print overall metrics (precision, recall, F1).
7. Compute metrics on slices of the categorical features and save them to `slice_output.txt`.

Example console output:
```
Precision: 0.7280 | Recall: 0.5702 | F1: 0.6395
```


---

## API Usage

The model is served using **FastAPI**.

Start the FastAPI server with:

```bash
uvicorn main:app --reload
```

When the server starts, it will run at:

```
http://127.0.0.1:8000
```

Available Endpoints:
- **GET /** = Returns a simple welcome message confirming the API is running.
- **POST /data/** = Accepts a JSON payload with the Census features, runs the model, and returns a prediction: {"result": "<=50K"} or {"result": ">50K"}

Example POST request

```json
{
  "age": 37,
  "workclass": "Private",
  "fnlgt": 178356,
  "education": "HS-grad",
  "education-num": 10,
  "marital-status": "Married-civ-spouse",
  "occupation": "Prof-specialty",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital-gain": 0,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States"
}
```

Example POST response

```json
POST /data/ result: {"result": "<=50K"}
```


---

## Local API Test

A helper script `local_api.py` is provided to automatically test both endpoints once the server is running.

Open a second terminal and run:

```bash
python local_api.py
```

This script will:
- Send a GET request to http://127.0.0.1:8000/ and print the welcome message.
- Send a POST request to http://127.0.0.1:8000/ with sample data and print the model prediction.

Example output:

```
GET / status: 200
GET / result: {'message': 'Hello from the API!'}
POST /data/ status: 200
POST /data/ result: {'result': '<=50K'}
```

A screenshot of this test run is included in the repository at:
`screenshots/local_api.png`

---

## Unit Tests

Unit tests are provided in `test_ml.py` to verify that the pipeline works as intended.

These tests check that:
1. `process_data` correctly transforms the dataset and returns expected shapes.
2. The training + inference pipeline produces predictions of the correct type.
3. The `compute_model_metrics` function returns the correct precision/recall/F1 when given known values.

Run the tests with:

```bash
pytest -v
```

Example output:
```
collected 3 items
test_ml.py::test_process_data_shapes PASSED
test_ml.py::test_train_and_inference PASSED
test_ml.py::test_compute_model_metrics_known_values PASSED
```

A screenshot of a successful test run is included at:
`screenshots/unit_test.png`


---

## Continuous Integration

Continuous integration is set up with GitHub Actions.
Workflow file: `.github/workflows/ci.yml`

On every push to GitHub, the workflow automatically:

1. Installs the project dependencies.
2. Runs **flake8** to check Python style and linting.
3. Runs **pytest** to ensure all unit tests pass.

The project only passes CI if both linting and tests succeed.

A screenshot of a successful run is included at: 
`screenshots/continuous_integration.png`


---

## Screenshots

The following screenshots are included in the repository to demonstrate successful runs of each required step:

- **Local API test** → `screenshots/local_api.png`
  Shows both the GET welcome message and the POST prediction result.
- **Unit tests** → `screenshots/unit_test.png`
  Shows all three unit tests passing with `pytest -v`.
- **CI passing** → `screenshots/continuous_integration.png`
  Shows GitHub Actions successfully completing both linting (`flake8`) and testing (`pytest`).


---

## Model Card

A completed model card is included in `model_card.md`.

It follows the full template and documents:
- **Model details** (author, algorithm, artifacts, code files)
- **Intended use** and audience
- **Training data** and preprocessing steps
- **Evaluation data** (test split and slice evaluations)
- **Metrics** (overall precision, recall, F1, plus slice metrics in `slice_output.txt`)
- **Ethical considerations** around sensitive attributes
- **Caveats and recommendations** for future improvements
- **References** to dataset and framework sources
