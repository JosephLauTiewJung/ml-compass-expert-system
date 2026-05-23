# ML Compass User Testing Scenarios

## Purpose

This user testing document checks whether ML Compass makes correct, explainable, and beginner-friendly recommendations across different machine learning project situations.

Each tester should enter the scenario details into the Streamlit form, submit the recommendation, and compare the system output against the expected decision.

## What to Record

For each scenario, record:

- Whether the predicted problem type is correct
- Whether the baseline model is suitable
- Whether the advanced model is suitable or safely avoided
- Whether the metrics match the data condition
- Whether warnings are clear and useful
- Whether the confidence level feels reasonable
- Any confusing question, label, or recommendation

## Test Scenarios

| ID | Scenario | Suggested Form Inputs | Expected Decision |
| --- | --- | --- | --- |
| UT-01 | Customer churn prediction from a CSV file | Target column: Yes; Goal: Predict a category; Data type: Tabular; Target: Category or label; Classes: 2; Rows: 5,000; Classes: Balanced; Priority: Balanced; Domain: General; Skill: Beginner/Intermediate | Problem type should be Classification, specifically Binary Classification. Recommend Logistic Regression or Decision Tree as baseline, Random Forest/XGBoost/LightGBM as advanced options, and Accuracy plus F1-Score as metrics. |
| UT-02 | House price prediction | Target column: Yes; Goal: Predict a number; Data type: Tabular; Target: Continuous number; Rows: 10,000; Outliers: Yes; Priority: Explainability; Domain: General | Problem type should be Regression. Recommend Linear Regression or Decision Tree Regressor as baseline. Since outliers exist, MAE should appear. R2 Score should also be recommended. |
| UT-03 | Numeric labels for student pass/fail | Target column: Yes; Goal: Predict a category; Data type: Tabular; Target: Numbers used as labels; Classes: 2; Rows: 800; Priority: Explainability; Domain: Educational | System should trigger the numeric trap rule and classify this as Classification, not Regression. It should warn that numeric labels are categories. It should prefer interpretable models such as Logistic Regression or Decision Tree. |
| UT-04 | Small image classification assignment | Target column: Yes; Goal: Predict a category; Data type: Image; Target: Category or label; Classes: More than 2; Rows: 500; GPU: No; Timeline: Short; Skill: Beginner | System may identify Classification/Image work, but it should warn that small data, no GPU, short timeline, and beginner skill make deep learning risky. It should avoid heavy models as first choice or clearly mark them as unsafe for a first model. |
| UT-05 | Product review sentiment analysis | Target column: Yes; Goal: Work with text or language; Data type: Text; Text complexity: Simple keyword or sentiment task; Target: Category or label; Classes: 2; Rows: 3,000; Classes: Balanced | Problem type should be NLP. Recommend TF-IDF + Naive Bayes, TF-IDF + Logistic Regression, or TF-IDF + SVM. Metrics should include Accuracy and F1-Score. |
| UT-06 | Legal document question answering | Target column: Unknown; Goal: Work with text or language; Data type: Text; Text complexity: Context, word order, or meaning matters; Rows: Unknown; Priority: Explainability; Domain: Legal; GPU: Unknown; Skill: Beginner | Problem type should be NLP with Medium or Low confidence because important inputs are unknown. It should prioritize caution and explainability because the domain is legal. It should warn about missing information and avoid presenting advanced transformer models as an easy first step. |
| UT-07 | Monthly sales forecasting | Target column: Yes; Goal: Predict future values; Data type: Time series; Target: Continuous number; Rows: 2,400; Outliers: No; Priority: Balanced | Problem type should be Time-Series Forecasting. Recommend Moving Average, ARIMA, or Prophet as baseline options. Metrics should include MAE, RMSE, or MAPE. Preprocessing should mention sorting by time and using a time-based train/test split. |
| UT-08 | Customer segmentation with no labels | Target column: No; Goal: Group similar records; Data type: Tabular; Rows: 20,000; Missing values: Mixed; Priority: Balanced | Problem type should be Clustering. Recommend K-Means or Hierarchical Clustering as baseline and DBSCAN or Gaussian Mixture Model as advanced options. Metrics should include Silhouette Score and Davies-Bouldin Index. |
| UT-09 | Credit card fraud detection | Target column: No or Unknown; Goal: Find rare or abnormal cases; Data type: Tabular; Rows: 100,000; Classes: Imbalanced or majority class over 80%; Domain: Financial; Priority: Explainability | Problem type should be Anomaly Detection. Recommend simple statistical baselines such as Z-Score/IQR and advanced methods such as Isolation Forest. Metrics should emphasize Precision, Recall, F1-Score, or PR-AUC rather than Accuracy. |
| UT-10 | Medical diagnosis with imbalanced classes | Target column: Yes; Goal: Predict a category; Data type: Tabular; Target: Category or label; Classes: 2; Rows: 8,000; Classes: Imbalanced; Majority class: 90%; Domain: Medical; Priority: Explainability | Problem type should be Classification. It should prioritize interpretable models and warn about imbalance. Metrics should include F1-Score, Precision, Recall, and PR-AUC. Accuracy alone should not be recommended as the main metric. |
| UT-11 | Direct formula calculation | Output can be calculated directly: Yes; Target column: Any; Goal: Predict a number; Data type: Tabular | System should recommend Rule-Based Logic or Standard Statistics instead of ML. Confidence should be High because the ML necessity gate is clear. |
| UT-12 | Incomplete beginner project description | Target column: I do not know; Goal: I do not know; Data type: I do not know; Dataset rows: Unknown; Class balance: Unknown; Priority: I do not know; Skill: Beginner | System should produce a temporary recommendation with Medium or Low confidence. It should clearly explain that important information is missing and suggest a safe beginner baseline rather than failing. |

## Acceptance Criteria

The user testing is successful if:

- At least 10 scenarios are tested by users.
- The system identifies the correct problem type in most clear-input scenarios.
- Numeric labels are not incorrectly treated as regression.
- Small datasets and limited resources reduce or block advanced model recommendations.
- Imbalanced classification scenarios do not rely only on Accuracy.
- Missing information produces a temporary recommendation instead of an error.
- Testers can understand why each model and metric was recommended.

## Tester Feedback Template

Use this format after each test:

```text
Scenario ID:
Tester name:
Was the problem type correct? Yes/No
Was the model recommendation suitable? Yes/No
Were the metrics suitable? Yes/No
Were warnings useful? Yes/No
Was anything confusing?
Suggested improvement:
```
