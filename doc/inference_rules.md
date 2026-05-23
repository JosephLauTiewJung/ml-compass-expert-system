# ML Compass — Identified Inference Rules

## A. Machine Learning Necessity Rules

1. **Machine Learning Necessity**
   IF output can be directly calculated using a formula OR follows a simple fixed rule, THEN recommend rule-based logic or standard statistics, not ML.

2. **Hidden Pattern Requirement**
   IF input-output relationship is not obvious AND the project needs to learn from past data, THEN ML is suitable.

## B. Priority Filter Rules

3. **Transparency Priority**
   IF explainability is required OR the domain is medical, legal, financial, or educational, THEN prioritize interpretable models such as Logistic Regression, Decision Tree, or rule-based models.

4. **Performance Priority**
   IF highest accuracy is the main priority AND explainability is less important AND sufficient data/resources exist, THEN recommend stronger models such as Random Forest, XGBoost, LightGBM, CNN, LSTM, BERT, or Transformer.

## C. Task-Based Rules

5. **Classification Task**
   IF the objective is to predict a category AND labeled data is available, THEN recommend classification methods.

6. **Regression Task**
   IF the objective is to predict a continuous numerical value AND labeled data is available, THEN recommend regression methods.

7. **Clustering Task**
   IF the dataset has no labels AND the goal is to group similar data points, THEN recommend clustering methods.

8. **Anomaly Detection Task**
   IF the goal is to identify rare, abnormal, or unusual patterns, THEN recommend anomaly detection methods.

9. **Time Series Task**
   IF the data contains a time component AND the objective is to predict future values or trends, THEN recommend time-series forecasting methods.

10. **NLP / LLM Task**
    IF the objective involves text understanding, generation, summarization, translation, or question answering, THEN recommend NLP or LLM-based methods.

## D. Numeric Trap Rules

11. **Numeric Category Rule**
    IF output is represented by numbers BUT the numbers are only category labels AND differences do not represent real quantity, THEN treat as classification.

12. **Continuous Quantity Rule**
    IF output is numerical AND differences between values represent a real measurable quantity, THEN treat as regression.

## E. Data Type-Based Rules

13. **Tabular Data**
    IF data is structured/tabular such as CSV, Excel, or database table, THEN recommend traditional ML methods such as Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, or LightGBM.

14. **Simple Text Data**
    IF data is text AND the task is simple AND dataset size is small/moderate, THEN recommend TF-IDF with Naive Bayes, Logistic Regression, or SVM.

15. **Complex Text Data**
    IF data is text AND context, word order, or sentence meaning is important OR the dataset is large, THEN recommend LSTM, BERT, or Transformer.

16. **Image Data**
    IF data type is image, THEN recommend CNN, ResNet, or Vision Transformer.

17. **Time Series Data**
    IF data is arranged by time order AND past values affect future values, THEN recommend ARIMA, Prophet, XGBoost with lag features, LSTM, or GRU.

18. **Multi-Modal Data**
    IF the project has multiple data types, THEN identify the dominant data type and either use one simple model or extract/combine features from each type.

## F. Learning Type-Based Rules

19. **Supervised Learning**
    IF labeled data is available AND there is a clear target column, THEN recommend supervised learning.

20. **Unlabeled Data — Clustering**
    IF there are no labels AND the goal is grouping or segmentation, THEN recommend clustering.

21. **Unlabeled Data — Anomaly Detection**
    IF there are no labels AND the goal is detecting rare or abnormal cases, THEN recommend anomaly detection.

22. **Semi-Supervised Approach**
    IF there are no labels BUT labeling is possible through rules, experts, or partial annotation, THEN recommend semi-supervised learning or pseudo-labeling.

23. **Rule-Based Labeling**
    IF there is no target column BUT the student can create labels using domain rules, THEN recommend rule-based labeling before supervised training.

24. **Pseudo-Labeling**
    IF there are no labels AND clustering produces meaningful groups, THEN use cluster IDs as pseudo-labels for supervised classification.

## G. Dataset Size Rules

25. **Small Dataset**
    IF dataset size is small, THEN recommend simple models such as Logistic Regression, Decision Tree, Naive Bayes, SVM, or KNN.

26. **Medium Dataset**
    IF dataset size is moderate, THEN recommend ensemble models such as Random Forest, XGBoost, or LightGBM.

27. **Large Dataset**
    IF dataset size is large AND the problem is complex, THEN recommend deep learning models such as CNN, LSTM, GRU, BERT, or Transformer.

28. **Overfitting Warning**
    IF dataset is small AND selected model is complex, THEN warn about overfitting and recommend a simpler baseline first.

29. **Baseline Model Rule**
    IF the student has not trained any model yet, THEN recommend starting with a simple baseline model before advanced models.

## H. Preprocessing Rules

30. **Scaling for Distance and Gradient Models**
    IF the selected model is SVM, KNN, Logistic Regression, or Neural Network, THEN recommend feature scaling.

31. **Scaling for Tree-Based Models**
    IF the selected model is Decision Tree, Random Forest, XGBoost, or LightGBM, THEN feature scaling is optional.

32. **Ordinal Encoding**
    IF categorical feature has natural order, THEN recommend Label Encoding.

33. **Nominal Encoding**
    IF categorical feature has no natural order, THEN recommend One-Hot Encoding.

34. **High Cardinality Encoding**
    IF categorical feature has too many unique values, THEN recommend Target Encoding or another suitable encoding method and avoid excessive one-hot encoding.

35. **Missing Value — Low Missing Percentage**
    IF missing values are less than 5% of dataset, THEN remove affected rows if it does not reduce dataset too much.

36. **Missing Numerical Values**
    IF missing values are numerical, THEN recommend Median Imputation, especially when outliers exist.

37. **Missing Categorical Values**
    IF missing values are categorical, THEN recommend Mode Imputation or create an “Unknown” category.

## I. Evaluation Metric Rules

38. **Balanced Classification**
    IF classification dataset has balanced classes, THEN recommend Accuracy.

39. **Imbalanced Classification**
    IF classification dataset is imbalanced, THEN recommend F1-Score, Precision, Recall, or PR-AUC and do not rely only on Accuracy.

40. **High False Negative Cost**
    IF missing a positive case is dangerous, THEN recommend Recall.

41. **High False Positive Cost**
    IF false alarms are costly, THEN recommend Precision.

42. **Regression with Normal Error Distribution**
    IF regression errors are normally distributed AND large errors should be penalized more, THEN recommend RMSE.

43. **Regression with Outliers**
    IF regression dataset contains many outliers, THEN recommend MAE.

44. **Regression Explanation**
    IF the student wants to know how much variance the model explains, THEN recommend R² Score.

## J. Practical Constraint Rules

45. **Resource Limitation**
    IF computational resources or time are limited, THEN recommend lightweight models such as Logistic Regression, Decision Tree, Naive Bayes, or simple SVM.

46. **No GPU Available**
    IF no GPU is available AND dataset is not very large, THEN avoid heavy deep learning and recommend Scikit-Learn-based models.

47. **Limited Programming Skill**
    IF student has limited programming experience, THEN recommend easier models such as Logistic Regression, Decision Tree, Random Forest, or Naive Bayes.

48. **Interpretability Requirement**
    IF explainability is required, THEN recommend Decision Tree, Logistic Regression, or Linear Regression, and optionally SHAP/LIME.

49. **Student Project Feasibility**
    IF project timeline is limited AND student has limited experience, THEN prioritize simple models, easy implementation, and clear metrics.

50. **Course Requirement Rule**
    IF course/lecturer requires model reasoning, THEN avoid black-box-only solutions and include an interpretable model or explanation method.

## K. Final Validation Rules

51. **Dataset Support Check**
    IF selected model does not match dataset size, THEN reject it and suggest a better-fit model.

52. **Implementation Capability Check**
    IF student cannot realistically implement, debug, or explain the model, THEN recommend a simpler method.

53. **Interpretability Check**
    IF domain requires explanation AND selected model is black-box, THEN suggest an interpretable alternative or add SHAP/LIME.

54. **Metric Clarity Check**
    IF selected metric does not match task type or data distribution, THEN reject it and recommend a better metric.

55. **Time Feasibility Check**
    IF the complete pipeline cannot finish within the timeline, THEN recommend a simpler model and shorter workflow.

56. **Resource Efficiency Check**
    IF selected model needs too much RAM, GPU, cloud cost, or training time, THEN recommend a more resource-efficient model.

## L. Final Recommendation Rule

57. **Approved Recommendation**
    IF task type is correct, model matches data type, model fits dataset size, preprocessing is suitable, metric is correct, method is feasible, and it matches explainability/performance needs, THEN approve the recommendation and provide model, preprocessing steps, evaluation metrics, and explanation.

---

# Improved Inference Rule Design

The original rules are enough for a working prototype, but the system will be stronger if the rules include conflict handling, confidence scoring, missing input handling, and clearer thresholds.

## 1. Rule Priority Layer

Some rules can conflict. For example, one rule may recommend deep learning because the data is image-based, while another rule may reject deep learning because the dataset is small or the student has no GPU.

### Priority Order

1. **ML Necessity Gate** — decide whether ML is needed at all.
2. **Task Type Rules** — classify the problem as classification, regression, clustering, anomaly detection, time series, or NLP.
3. **Data Type Rules** — choose model family based on tabular, text, image, or time-series data.
4. **Dataset Size Rules** — limit model complexity.
5. **Practical Constraint Rules** — check GPU, time, programming skill, and course requirement.
6. **Metric Rules** — choose evaluation metric.
7. **Final Validation Rules** — approve, reject, or warn.

### Improved Rule P1: Dataset Size Overrides Model Complexity

IF one rule recommends a complex model
AND dataset size is small
THEN prioritize the small-dataset rule
AND recommend a simpler baseline first.

### Improved Rule P2: Practical Constraint Overrides Advanced Recommendation

IF a model requires high computational resources
AND the student has limited time, no GPU, or limited programming skill
THEN reject the advanced model as the first recommendation
AND suggest a simpler alternative.

## 2. Confidence Score Rules

Instead of giving only one recommendation, the system should also show how confident it is.

### Improved Rule C1: High Confidence Recommendation

IF task type, data type, dataset size, metric, and constraints all point to the same model family
THEN confidence = High.

### Improved Rule C2: Medium Confidence Recommendation

IF task type and data type are clear
BUT dataset size, class balance, missing values, or constraints are unknown
THEN confidence = Medium
AND ask follow-up questions.

### Improved Rule C3: Low Confidence Recommendation

IF important inputs are missing
OR multiple rules strongly conflict
THEN confidence = Low
AND provide a warning before recommending.

## 3. Missing Input Handling Rules

The system should not fail when the student gives incomplete answers.

### Improved Rule M1: Missing Target Column Information

IF the user does not know whether a target column exists
THEN ask the user to identify what they want to predict
AND infer whether a target column is needed.

### Improved Rule M2: Missing Dataset Size

IF dataset size is unknown
THEN recommend a simple baseline model first
AND mark advanced model recommendations as uncertain.

### Improved Rule M3: Missing Class Balance

IF classification task is selected
AND class balance is unknown
THEN recommend Accuracy plus F1-score
AND warn that Accuracy alone may be misleading.

### Improved Rule M4: Missing Data Type

IF data type is unknown
THEN ask whether the data is mainly table, text, image, audio, or time-ordered values.

## 4. Threshold Rules

The current rules mention small, medium, and large datasets, but the system needs rough numbers.

### Suggested Dataset Size Thresholds

* **Small dataset:** fewer than 1,000 rows
* **Medium dataset:** 1,000 to 100,000 rows
* **Large dataset:** more than 100,000 rows

### Improved Rule T1: Small Dataset Threshold

IF dataset rows < 1,000
THEN recommend Logistic Regression, Decision Tree, Naive Bayes, KNN, or SVM
AND avoid deep learning as the first model.

### Improved Rule T2: Medium Dataset Threshold

IF dataset rows are between 1,000 and 100,000
THEN recommend Random Forest, XGBoost, LightGBM, or SVM depending on task type.

### Improved Rule T3: Large Dataset Threshold

IF dataset rows > 100,000
AND data type is text, image, audio, or sequential
THEN deep learning may be considered.

## 5. Better Model Ranking Rules

Instead of recommending one model, ML Compass should rank models.

### Improved Rule R1: Always Recommend Baseline First

IF this is a beginner project
THEN provide at least one simple baseline model before advanced models.

### Improved Rule R2: Recommend Advanced Model as Upgrade

IF baseline model is suitable
AND dataset size/resources are enough
THEN recommend an advanced model as the second-stage improvement.

### Improved Rule R3: Explain Trade-Off

IF two models are recommended
THEN explain the trade-off using accuracy, explainability, training time, and implementation difficulty.

## 6. Improved Task-Specific Rules

### Improved Rule TS1: Binary vs Multi-Class Classification

IF task is classification
AND target column has 2 unique classes
THEN identify the task as Binary Classification.

IF task is classification
AND target column has more than 2 unique classes
THEN identify the task as Multi-Class Classification.

### Improved Rule TS2: Multi-Label Classification

IF one data point can belong to multiple labels at the same time
THEN identify the task as Multi-Label Classification
AND recommend suitable methods such as One-vs-Rest Logistic Regression or tree-based models.

### Improved Rule TS3: Forecasting vs Time-Based Regression

IF time is only one feature
AND the goal is not predicting future ordered values
THEN treat as normal regression/classification.

IF chronological order affects the prediction
AND the goal is future prediction
THEN treat as time-series forecasting.

## 7. Data Quality Rules

### Improved Rule DQ1: Duplicate Rows

IF many duplicate rows exist
THEN warn the user
AND recommend removing duplicates before training.

### Improved Rule DQ2: Data Leakage

IF a feature directly reveals the target answer
THEN warn about data leakage
AND remove that feature before training.

### Improved Rule DQ3: Too Many Missing Values

IF a feature has more than 40% missing values
THEN recommend dropping the feature
OR ask the user whether the feature is important.

### Improved Rule DQ4: High Imbalance Warning

IF the largest class is more than 80% of the dataset
THEN mark the classification problem as imbalanced
AND recommend F1-score, Recall, Precision, PR-AUC, or class-weighted models.

## 8. Better Output Rule

### Improved Rule O1: Structured Recommendation Output

IF final recommendation is approved
THEN output:

1. Problem type
2. Confidence level
3. Recommended baseline model
4. Recommended advanced model
5. Preprocessing steps
6. Evaluation metrics
7. Warning messages
8. Simple explanation
9. Starter code suggestion

## 9. Improved Final Rule

### Improved Rule FINAL-2: Safe Final Recommendation

IF the system has enough information
AND no high-priority rule conflict exists
THEN provide a final recommendation.

IF important information is missing
THEN provide a temporary recommendation
AND clearly state what information is missing.

IF the recommendation may be risky or uncertain
THEN show a warning and suggest a safer beginner-friendly option.
