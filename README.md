# Loan Approval Prediction: A Machine Learning Case Study

## Project Overview
This project aims to build a robust and accurate machine learning model to predict loan approval status. The primary challenge was addressing a highly imbalanced dataset and ensuring the model's reliability on new, unseen data. Through a rigorous and iterative process, this project demonstrates a complete machine learning workflow, from data preprocessing to model optimization and final evaluation.

## Problem Statement
In the financial sector, accurately predicting loan approval or denial is a critical task. It involves assessing various applicant attributes to determine creditworthiness and risk. This project aims to develop a predictive model that can accurately classify loan applications, providing insights for financial institutions and enabling more informed and efficient decision-making processes. A key challenge addressed is dealing with imbalanced datasets, which are common in loan approval scenarios where approved loans might significantly outnumber denied ones.

## Dataset
- **Total Observations:** 614
- **Nature:** Tabular data, containing various features related to loan applicants (e.g., credit history, income, education, loan amount, property area, etc.). The dataset was imbalanced, requiring specific techniques to handle this.
- **Splitting:** The dataset was split into:
  - **Training Set:** 491 observations (~80% of total), used for model training.
  - **Test Set:** 123 observations (~20% of total), completely held out for final, unbiased model evaluation.
- Splits were performed using StratifiedKFold and train_test_split with stratify to maintain class distribution.

## Key Features

-   **End-to-end Machine Learning Pipeline:** A complete workflow from data cleaning to model deployment.
-   **Robust Preprocessing:** Correctly handles missing values, inconsistent data, and feature engineering while preventing data leakage.
-   **Class Imbalance Handling:** Utilizes SMOTE (Synthetic Minority Over-sampling Technique) to ensure the model learns from all classes.
-   **Model Comparison:** Evaluates and compares the performance of multiple models, including Logistic Regression, Random Forest, and XGBoost.
-   **Data-Driven Conclusion:** The project's outcome provides a powerful lesson on why simpler models can outperform complex ones.

## Methodology

### 1. Data Preparation
The data was first split into training and testing sets to prevent **data leakage**. Missing values were imputed (using the median for numerical and mode for categorical features), and new features (`TotalIncome`, `Loan_to_Income_Ratio`) were engineered. Categorical features were one-hot encoded and numerical features were standardized using a scikit-learn pipeline.

### 2. Model Training & Evaluation
Initial models (Random Forest, XGBoost) showed severe **overfitting**, performing well on the training data but poorly on the test set. Extensive hyperparameter tuning was performed, but a significant performance gap persisted.

### 3. Final Model Selection
A simple **Logistic Regression** model was trained and, surprisingly, outperformed the tuned, complex models. The test scores for this model were even higher than its training scores, indicating excellent generalization and no signs of overfitting.

### 4. Optimization
The Logistic Regression model was then hyperparameter tuned, which significantly boosted its performance and confirmed it as the best-performing model for this project.

### 5. Model Evaluation 
Performance metrics (F1-score, Accuracy, ROC-AUC) were calculated on both the training and the completely unseen test sets.

## Final Results

The hyperparameter-tuned Logistic Regression model achieved the following performance on the unseen test set:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 0.8537 |
| **F1-Score** | **0.9032** |
| **ROC-AUC** | 0.8618 |

This result confirms the model's reliability and strong predictive power.

## Key Observations

* **Simplicity Wins:** A simple Logistic Regression model ultimately outperformed more complex models like Random Forest and XGBoost, which struggled with overfitting.
* **The Power of Process:** The project demonstrated the critical importance of a correct workflow, proving that preprocessing and feature engineering must be done *after* the train-test split to prevent data leakage.
* **Feature Engineering's Impact:** The custom-engineered features, such as `TotalIncome` and `Loan_to_Income_Ratio`, proved to be highly predictive and were instrumental in the final model's success.
* **Optimal Performance:** The final hyperparameter-tuned model achieved an outstanding test F1-Score of over 0.90, showcasing excellent performance and generalization.

## Getting Started
Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
  - Python 3.x (Ensure that you have python 3.10 version to avoid library or package compatability issue)
  - Pip (Python package installer) (When you create virtual environment or install python, by default it installs pip and setuptools packages)

### Installation
  1. **Create a virtual environment (recommended):**
     ```bash
     python -m venv venv
     # On Windows
     .\venv\Scripts\activate
     # On macOS/Linux
     source venv/bin/activate
     ```
  2. **Clone the repository:**
     ```bash
     git clone https://github.com/Sonjoy95/Loan-Approval-Classification-Model.git
     cd Loan-Approval-Classification-Model
     ```
  
  3. **Install the required packages:**
     ```bash
     pip install -r requirements.txt
     ```
  
  4. **Open Jupyter Notebook:**
     ```bash
     jupyter notebook
     ```
  
  5. Run the cells in the notebook to reproduce the results.

### Contact

  - **Name:** Sanjay Chabukswar
  - **Email:** sanjaychabukswar1995@gmail.com
