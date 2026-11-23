
# Week 2 - Machine Learning: Regression and Classification

## Project Overview
This project focuses on supervised machine learning techniques, including regression and classification. It covers data preprocessing, exploratory data analysis, model training, evaluation, and model comparison.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Datasets
1. **Housing Prices Dataset:** Used to predict house prices.
2. **Titanic Dataset:** Used to predict passenger survival.

## Tasks Completed

### 1. Housing Price Prediction
- Performed exploratory data analysis.
- Applied Linear Regression.
- Evaluated the model using Mean Squared Error and R² score.
- Compared actual and predicted house prices.
- Saved the trained model.

**Results:**
- R² Score: 65.29%
- Model: Linear Regression

### 2. Titanic Survival Prediction
- Applied Logistic Regression.
- Generated a confusion matrix and classification report.
- Evaluated model accuracy.

**Accuracy:** 80.45%

### 3. Classification Model Comparison
Compared three classification algorithms:

| Model | Accuracy |
|---|---:|
| Decision Tree | 79.33% |
| Scaled KNN | 81.56% |
| Random Forest | 82.12% |

**Best test accuracy:** Random Forest - 82.12%

### 4. Decision Tree Hyperparameter Tuning
Tested different values of `max_depth` to observe their effect on model performance.

## Project Structure

```text
Week2_Machine_Learning/
├── dataset/
├── notebooks/
├── models/
├── outputs/
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

1. Clone this repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Python scripts from the project root directory. Example:

```bash
python notebooks/linear_regression_model.py
python notebooks/titanic_logistic_regression.py
python notebooks/random_forest_model.py
```

## Learning Outcomes
- Understanding supervised machine learning.
- Performing data preprocessing and exploratory data analysis.
- Building regression and classification models.
- Evaluating models using appropriate metrics.
- Comparing algorithms and tuning hyperparameters.

## Author
Varshini
