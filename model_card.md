# Model Card


---

## Model Details

- **Author**: Dennis Grisham (Udacity student project)
- **Algorithm**: Logistic Regression (scikit-learn)
- **Version**: v1.0 (trained September 2025)
- **Artifacts**:
  - `model/model.pkl` = trained Logistic Regression model
  - `model/encoder.pkl` = fitted OneHotEncoder
- **Code files**: `train_model.py`, `ml/model.py`, `ml/data.py`


---

## Intended Use

- **Primary purpose**: Educational demonstration of building and deploying a ML pipeline.
- **Prediction task**: Classify whether an individual earns more than `$50K` annually (`>50K`) or not (`<=50K`) using demographic data.
- **Intended audience**: Udacity reviewers, instructors, and other learners.
- **Not intended for**: Real-world employment, credit, or financial decision-making. The model is simplified and not validated for production.


---

## Training Data

- **Source**: `data/census.csv` (U.S. Census Bureau Adult dataset).
- **Features**: Demographics such as age, education, occupation, marital status, race, sex, hours per week, native country, etc.
- **Target**: Binary label indicating income `<=50K` or `>50K`.
- **Preprocessing**:
  - Categorical variables = encoded with OneHotEncoder
  - Target label = binarized with LabelBinarizer
  - Data split into train and test subsets with `train_test_split` 


---

## Evaluation Data

- **Held-out test set** created during training (80/20 split).
- **Consistent preprocessing** applied using the fitted encoder and label binarizer.
- **Slice evaluation** performed on each categorical feature (e.g., `sex=Male`, `education=Bachelors`) to analyze fairness and performance differences. Results logged in `slice_output.txt`.


---

## Metrics

The model is evaluated using **precision, recall, and F1 (fbeta with β=1)**.

- **Overall test set performance**: 
  - Precision: **0.7280**
  - Recall: **0.5702**
  - F1: **0.6395**

- **Slice metrics**:
  Detailed per-category results (e.g., performance on subsets like `sex=Female` or `education=Masters`) are saved in `slice_output.txt`. These help highlight groups where the model performs better or worse.


---

## Ethical Considerations

- The dataset contains sensitive attributes such as sex, race, and marital status.
- Using such features for real-world decision-making would probably not be good from a HR perspective.
- This project is just for learning — I’m not qualified to say how it should be used in real companies.


---

## Caveats and Recommendations

- **Simple model**: I used Logistic Regression because it keeps the model small and easy to save, not because it’s the most powerful option.
- **Limits**: The model was trained only on this dataset, so it may not work as well on other groups or in real-world settings.
- **Possible next steps**:
  - Try stronger models like Random Forest or Gradient Boosting.
  - Tune the settings (hyperparameters) and maybe scale the numeric features for better accuracy.
  - Look into ways to check and reduce bias, especially for sensitive features like race and sex.


---

## References

- Dataset: [UCI Adult Census Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

