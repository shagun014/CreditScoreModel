# CreditScoreModel

# Credit Scoring Machine Learning Model

## 🎯 Project Overview

This project implements a complete **Machine Learning pipeline** for **credit risk assessment (credit scoring)**. It encompasses synthetic data generation, robust preprocessing, feature engineering, training and comparison of multiple classification models, and comprehensive performance evaluation using various metrics and visualizations.

The goal is to classify loan applicants as either **'creditworthy' (1)** or **'not creditworthy' (0)**.

## ⚙️ Model Pipeline Steps

The `CreditScoringModel` class manages the entire workflow:

1.  **`generate_synthetic_data()`**: Creates a rich, synthetic dataset with a mix of numerical and categorical features, and a target variable (`creditworthy`) based on realistic credit risk logic.
2.  **`preprocess_data()`**: Handles missing value imputation (median for numerical, most frequent for categorical) and performs **Label Encoding** for categorical features.
3.  **`feature_engineering()`**: Creates new, potentially highly predictive features such as ratios (`income_to_age_ratio`) and binary risk indicators (`high_utilization`, `frequent_late_payments`).
4.  **`train_models()`**: Trains and evaluates **Logistic Regression**, **Decision Tree**, and **Random Forest** classifiers. Features are scaled for Logistic Regression.
5.  **`print_model_comparison()`**: Displays key performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC) for all models.
6.  **`plot_results()`**: Generates insightful plots, including **ROC Curves**, **Precision-Recall Curves**, a **Model Comparison Bar Chart**, and **Random Forest Feature Importance**.
7.  **`predict_creditworthiness()`**: Demonstrates how to use a trained model (defaulting to Random Forest) to make real-time predictions on new data.

## 📊 Key Features

* **Multi-Model Comparison:** Evaluates different model types (linear vs. tree-based).
* **Stratified Splitting:** Ensures the training and test sets maintain the original target variable's class balance.
* **Feature Importance:** Provides interpretability by showing which factors drive the Random Forest's decisions.
* **End-to-End Class:** Encapsulates all necessary steps within a single, reusable Python class.

## 🛠️ Requirements

The project is written in Python and requires the following libraries:

| Library | Purpose |
| :--- | :--- |
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computing and synthetic data generation |
| `matplotlib` & `seaborn` | Data visualization |
| `scikit-learn` | Model training, preprocessing, and evaluation metrics |

You can install the dependencies using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
