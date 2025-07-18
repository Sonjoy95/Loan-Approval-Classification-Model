# Loan Approval Prediction using Optimized XGBoost
## Table of Contents
- **Project Overview**
- **Problem Statement**
- **Dataset**
- **Methodology**
- **Results**
- **Contact**
### Project Overview
This project focuses on building and optimizing a classification model to predict loan approval outcomes. We leverage the power of the XGBoost algorithm, a highly efficient and flexible gradient boosting framework, coupled with robust hyperparameter tuning techniques to achieve high predictive accuracy.

The solution includes comprehensive data preprocessing steps, handling of class imbalance, and a meticulous hyperparameter optimization strategy to ensure robust performance on unseen data. A comparison with a Logistic Regression baseline is also presented to highlight the model's effectiveness in this domain.

### Problem Statement
In the financial sector, accurately predicting loan approval or denial is a critical task. It involves assessing various applicant attributes to determine creditworthiness and risk. This project aims to develop a predictive model that can accurately classify loan applications, providing insights for financial institutions and enabling more informed and efficient decision-making processes. A key challenge addressed is dealing with imbalanced datasets, which are common in loan approval scenarios where approved loans might significantly outnumber denied ones.

### Dataset
- **Total Observations:** 614
- **Nature:** Tabular data, containing various features related to loan applicants (e.g., credit history, income, education, loan amount, property area, etc.). The dataset was imbalanced, requiring specific techniques to handle this.
- **Splitting:** The dataset was split into:
  - **Training Set:** 496 observations (~81% of total), used for model training.
  - **Validation Set:** 62 observations (~10% of total), used for early stopping during hyperparameter tuning.
  - **Test Set:** 62 observations (~10% of total), completely held out for final, unbiased model evaluation.
- All splits were performed using StratifiedKFold and train_test_split with stratify to maintain class distribution.

### Methodology
The solution involves the following key steps:

1. **Data Splitting:** The raw data was split into training, validation, and testing sets to ensure proper model evaluation and prevent data leakage.
2. **Feature Scaling:** RobustScaler was applied to the features. This scaler is robust to outliers, making it suitable for datasets that may contain extreme values. The scaler was fitted only on the training data and then used to transform validation and test sets.
3. **Handling Class Imbalance:** The Synthetic Minority Over-sampling Technique (SMOTE) was applied to the training data (x_train_scaled and y_train) to address class imbalance in the loan approval outcomes. This helps the model learn from the minority class effectively without being biased towards the majority class. Crucially, SMOTE was applied only to the training data to prevent data leakage.
4. **Model Selection:**
  - **Baseline Model**: XGBoost Classifier, Logistic Regression, Random Forest was used as a simple yet effective baseline.
  - **Primary Model:** XGBoost Classifier, known for its performance and speed, was selected for its ability to handle complex relationships in the data.
5. **Hyperparameter Tuning:**
  - **Technique:** RandomizedSearchCV with StratifiedKFold (5 splits) was used for efficient exploration of the hyperparameter space.
  - **Objective:** Optimized for `roc_auc` score.
  - **Early Stopping:** XGBoost's `early_stopping_rounds` was utilized with the validation set (`x_val_scaled_df`, `y_val`) to prevent overfitting during individual training runs within the RandomizedSearchCV folds.
  - **Parameter Grid:** The hyperparameter search involved iterative refinement of the parameters to balance model complexity and regularization, leading to optimal performance
6. **Model Evaluation:** Performance metrics (F1-score, Accuracy, ROC-AUC) were calculated on both the training and the completely unseen test sets.

### Results
The hyperparameter tuning process successfully identified an optimal XGBoost model that achieved excellent performance in loan approval prediction on the unseen test data.

**Final Model Performance Comparison:**

| Model                 | F1-Score (Train) | F1-Score (Test) | Accuracy (Train) | Accuracy (Test) | ROC-AUC (Train) | ROC-AUC (Test) |
| :-------------------- | :--------------- | :-------------- | :--------------- | :-------------- | :-------------- | :------------- |
| **XGBoost Classifier** | 0.85             | **0.90** | 0.84             | **0.85** | 0.94            | **0.88** |
| Logistic Regression   | 0.87             | 0.87            | 0.80             | 0.82            | 0.78            | 0.87           |
| Random Forest         | 1.0              | 0.89        | 1.0             | 0.85            | 1.0          | 0.85       |

**Key Observations:**
  - The **XGBoost Classifier significantly outperforms the Logistic Regression baseline** across all key metrics (F1, Accuracy, ROC-AUC) on the unseen test data. This indicates its superior capability in capturing complex patterns within loan application data.
  - The final XGBoost model demonstrates **strong generalization capabilities**, with a very small difference between training and testing scores for F1-score and Accuracy.
  - The high test F1-score (0.90) indicates excellent precision and recall in identifying loan approval outcomes.
  - The test ROC-AUC (0.88) signifies a robust ability to distinguish between approved and denied loan applications across various thresholds.


### Getting Started
Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

**Prerequisites**
  - Python 3.x (Ensure that you have python 3.10 version to avoid library or package compatability issue)
  - Pip (Python package installer) (When you create virtual environment or install python, by default it installs pip and setuptools packages)

**Installation**
  1. **Clone the repository:**
     ```bash
     git clone https://github.com/Sonjoy95/Loan-Approval-Classification-Model.git
     cd Loan-Approval-Classification-Model
    ```
  2. **Create a virtual environment (recommended):**
     ```bash
     python -m venv venv
     # On Windows
     .\venv\Scripts\activate
     # On macOS/Linux
     source venv/bin/activate
     ```
  3. **Install the required packages:**
     ```bash
     pip install -r requirements.txt
     ```

### Contact

**Name:** Sanjay Chabukswar
**Email:** sanjaychabukswar1995@gmail.com
