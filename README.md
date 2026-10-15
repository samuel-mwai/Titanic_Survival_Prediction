# 🚢 Titanic Survival Prediction — Machine Learning Project

## 📌 Project Overview

This project uses **Machine Learning to predict whether a passenger survived the Titanic disaster** based on passenger characteristics such as age, sex, passenger class, fare, and family relationships.

The project follows an end-to-end supervised machine learning workflow, including data exploration, preprocessing, feature engineering, model comparison, cross-validation, hyperparameter optimization, model evaluation, and prediction.

A Random Forest classifier was trained and saved for reuse, with additional functionality to generate predictions for the Kaggle Titanic test dataset.

## 🎯 Objectives

- Explore and understand the Titanic passenger dataset.
- Identify and handle missing values.
- Prepare numerical and categorical features for machine learning.
- Train and compare multiple classification algorithms.
- Apply cross-validation to evaluate model performance.
- Optimize hyperparameters using Randomized Search and Bayesian optimization with Optuna.
- Train and evaluate a final Random Forest model.
- Save the trained model using Joblib.
- Generate predictions and create a Kaggle-compatible submission file.

## 📊 Dataset

The project uses the Titanic passenger dataset.

The training dataset contains **891 passengers and 12 original columns**.

### Target variable

`Survived`

- `0` — Did not survive
- `1` — Survived

### Features used for prediction

| Feature | Description |
|---|---|
| `Pclass` | Passenger ticket class |
| `Sex` | Passenger's sex |
| `Age` | Passenger's age |
| `SibSp` | Number of siblings or spouses aboard |
| `Parch` | Number of parents or children aboard |
| `Fare` | Passenger fare |
| `Embarked` | Port of embarkation |

The notebook removes `PassengerId`, `Name`, and `Ticket` from the training features and drops `Cabin` because it contains a large proportion of missing values.

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas & NumPy** — Data manipulation
- **Matplotlib & Seaborn** — Exploratory data analysis and visualization
- **Scikit-learn** — Preprocessing, model training, cross-validation, and evaluation
- **XGBoost** — Imported as part of the modelling environment
- **Optuna** — Bayesian hyperparameter optimization
- **Joblib** — Saving and loading the trained model
- **Jupyter Notebook** — Development environment

## 🔄 Machine Learning Workflow

### 1. Data Exploration

The notebook performs initial data checks to examine:

- Dataset dimensions and column names
- Data types
- Duplicate records
- Missing values
- Categorical feature values
- Distribution of the target variable

The original dataset contains missing values in `Age`, `Cabin`, and `Embarked`.

### 2. Data Preprocessing

A Scikit-learn preprocessing pipeline is used to prepare the data.

**Numerical features**
- Missing values are imputed using the median.
- Features are standardized using `StandardScaler`.

**Categorical features**
- Missing values are imputed using the most frequent value.
- Features are encoded using `OneHotEncoder`.
- Unknown categories are handled with `handle_unknown="ignore"`.

A `ColumnTransformer` combines the numerical and categorical preprocessing pipelines.

### 3. Train-Test Split

The data is split into:

- **80% training data**
- **20% testing data**

The split uses `random_state=42` and stratification on the target variable to preserve the class distribution.

### 4. Model Training and Comparison

The notebook compares five classification algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting

Five-fold `StratifiedKFold` cross-validation is used to compare model performance using the F1-score and ROC-AUC.

### 5. Hyperparameter Optimization

The project explores two approaches to improve the Random Forest model:

**RandomizedSearchCV**

Randomly samples hyperparameter combinations to search for a better-performing model.

**Optuna**

Uses optimization trials to explore different values for parameters such as:

- Number of trees (`n_estimators`)
- Maximum tree depth (`max_depth`)
- Minimum samples required to split a node (`min_samples_split`)
- Minimum samples required at a leaf (`min_samples_leaf`)

The recorded Optuna study achieved a best cross-validation F1-score of approximately **0.7571**.

### 6. Final Model

A Random Forest classifier is trained using a preprocessing pipeline and selected hyperparameters.

The fitted pipeline includes both preprocessing and the classifier, making it possible to apply the same transformations when predicting on new passenger records.

## 📈 Model Evaluation

The final model was evaluated on the held-out test set.

| Metric | Class 0: Did not survive | Class 1: Survived |
|---|---:|---:|
| Precision | 0.81 | 0.82 |
| Recall | 0.91 | 0.65 |
| F1-score | 0.85 | 0.73 |

**Overall test accuracy: 81%**

The model performed better at identifying passengers who did not survive than at identifying passengers who survived. In particular, the recall for the survival class was 0.65, indicating that some actual survivors were incorrectly classified.

These results provide a useful baseline for further improvements.

## 💾 Saving and Loading the Model

The trained pipeline is saved with Joblib.

```python
import joblib

joblib.dump(
    rf_pipeline,
    "Models/titanic_model.joblib"
)
```

Load the saved model:

```python
model = joblib.load("Models/titanic_model.joblib")
```

The paths above are repository-relative examples; the notebook originally saved the model using an absolute Windows path.

## 🔮 Making Predictions

The saved model can predict survival for a new passenger.

```python
import pandas as pd
import joblib

model = joblib.load("Models/titanic_model.joblib")

new_passenger = pd.DataFrame({
    "Pclass": [1],
    "Sex": ["female"],
    "Age": [12],
    "SibSp": [2],
    "Parch": [1],
    "Fare": [54.72],
    "Embarked": ["S"]
})

prediction = model.predict(new_passenger)
probability = model.predict_proba(new_passenger)[:, 1]

print("Predicted survival:", prediction[0])
print("Survival probability:", probability[0])
```

In the notebook's example, the model predicted survival (`1`) with a predicted survival probability of approximately **93.64%**. This is a model estimate, not a guarantee of the actual outcome.

## 📁 Project Structure

A suggested GitHub repository structure is:

```text
Titanic_Survival_Prediction/
│
├── Data/
│   └── README.md
│
├── Models/
│   └── titanic_model.joblib
│
├── Notebooks/
│   └── titanic.ipynb
│
├── Submissions/
│   └── submission.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

The dataset and generated model files can be excluded from GitHub if they are large or unnecessary for reproducing the notebook. The `Data/README.md` file can explain where to obtain the dataset.

## 🚀 Getting Started

### Prerequisites

Install Python and Jupyter Notebook.

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna joblib jupyter
```

### Run the project

1. Clone the repository:

   ```bash
   git clone https://github.com/samuelmwai/Titanic_Survival_Prediction.git
   ```

2. Navigate into the project:

   ```bash
   cd Titanic_Survival_Prediction
   ```

3. Install the dependencies.

4. Place the Titanic training dataset in the expected data directory and update the notebook's file path if necessary.

5. Open Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

6. Open the Titanic notebook and execute the cells in order.

**Note:** The notebook uses local Windows paths. Update those paths or replace them with relative paths when running the project on another computer.

## 📤 Kaggle Submission

The notebook also includes a workflow for generating predictions on the separate Titanic test dataset.

It:

1. Loads the saved Random Forest pipeline.
2. Reads the Kaggle test data.
3. Selects the same seven features used during training.
4. Predicts survival for each passenger.
5. Creates a submission containing `PassengerId` and `Survived`.
6. Saves the result as `Submissions/submission.csv`.

The resulting CSV follows the expected two-column structure for a Titanic competition submission. A successful file generation does not, by itself, establish a Kaggle leaderboard score.

## 🔧 Future Improvements

- Improve recall for passengers who survived.
- Compare model performance using additional metrics and a confusion matrix.
- Further tune hyperparameters and validate the selected configuration.
- Investigate useful feature engineering, such as family size and passenger titles.
- Use a fully reproducible pipeline with relative file paths.
- Evaluate generalization through cross-validation and the held-out test set.
- Develop a Streamlit application for interactive survival predictions.

## 👨‍💻 Author

**Samuel Mwai**

Data Science & Analytics | Machine Learning | Python | Data Analysis

GitHub: https://github.com/samuelmwai

## 📄 License

No license has been specified yet. Add a suitable open-source license if you intend to permit others to reuse or modify the project.

---

*This project demonstrates the application of supervised machine learning, model evaluation, hyperparameter optimization, and model persistence to a binary classification problem.*
