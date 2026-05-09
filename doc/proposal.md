# Project Proposal: ML Compass — Expert System for Machine Learning Model Selection

## 1. Project Title

**ML Compass: An Expert System to Help Students Choose Suitable Machine Learning Algorithms**

---

## 2. Introduction

Many machine learning beginners struggle to choose the correct machine learning algorithm when they receive an assignment, homework task, or Kaggle-style problem. Students may not know whether their problem is classification, regression, clustering, or natural language processing.

Because of this, they may choose unsuitable models, use wrong evaluation metrics, or skip important preprocessing steps.

**ML Compass** is an expert system designed to guide students step by step in selecting suitable machine learning algorithms. The system asks simple guided questions about the dataset and problem, then recommends suitable machine learning models, evaluation metrics, preprocessing steps, and starter code.

The purpose of ML Compass is not to replace a machine learning expert. Instead, it works like a beginner-friendly assistant that helps students make better first decisions when starting a machine learning project.

---

## 3. Project Objectives

The objectives of this project are to:

* Help students identify the type of machine learning problem
* Recommend suitable machine learning algorithms
* Suggest correct evaluation metrics
* Provide preprocessing advice
* Explain the reason behind each recommendation
* Help students complete assignments and Kaggle competitions more confidently

---

## 4. Target Users

| User                       | Purpose                     |
| -------------------------- | --------------------------- |
| Machine learning beginners | Learn how to choose models  |
| College students           | Complete ML assignments     |
| Kaggle beginners           | Select baseline models      |
| Data science learners      | Understand ML workflow      |
| Teachers / tutors          | Use as a teaching assistant |

---

## 5. Literature Review: Related and Recent Available Expert Systems

Expert systems are computer-based systems that imitate the decision-making ability of human experts. They usually contain a knowledge base, an inference engine, and a user interface.

Expert systems are commonly used in areas such as:

* Medical diagnosis
* Education
* Engineering
* Agriculture
* Business decision support
* Data analysis

In the machine learning field, many tools already help users automate or simplify model selection. Examples include AutoML platforms such as:

* Auto-sklearn
* TPOT
* H2O AutoML
* Google AutoML
* Microsoft Azure AutoML

These systems can automatically test multiple models and select the best-performing one based on metrics.

However, many AutoML systems focus mainly on automation and performance. They may not clearly teach beginners why a model is selected, why a metric is suitable, or what preprocessing steps are needed.

This creates a gap for students who want to learn the reasoning behind model selection.

**ML Compass** focuses on the educational side. Instead of fully automating machine learning, it guides students using expert rules and explanations. It helps users understand the basic decision-making process behind machine learning model selection.

---

## 6. Specific Domain / Unique Use Case

The specific domain of this expert system is **machine learning model selection for beginner-level machine learning tasks**.

The unique use case is helping students choose suitable algorithms for common machine learning problem types without needing advanced machine learning knowledge.

### Supported Problem Types

* Classification
* Regression
* Clustering
* Time-series forecasting
* Basic natural language processing tasks

### Example Scenario

The system asks:

* Do you have a target column?
* Is the target column numerical or categorical?
* Is your data tabular, text, image, or time-series?
* Is the dataset balanced or imbalanced?

Based on the answers, the system recommends suitable models, metrics, preprocessing steps, and starter code.

---

## 7. Subject Matter Expert from the Field / Industry

The knowledge in the expert system should be supported by a subject matter expert in machine learning, data science, or artificial intelligence.

In this case, we invited **[professor name]**, a professor from **[department]** in **University Malaya**.

| Field           | Details              |
| --------------- | -------------------- |
| Name            | [professor name]     |
| Position        | [professor position] |
| Department      | [department]         |
| Expertise       | [expertise]          |
| Role in Project | [role]               |

---

## 8. Expert System Architecture Design

### 8.1 User Interface

The user interface allows users to:

* Input problem description
* Answer guided questions
* View preprocessing suggestions
* View recommended models
* View explanation
* View recommended metrics
* View starter code

### 8.2 Question Engine

The question engine asks guided questions such as:

* Do you have a target column?
* What is the target column?
* Is the target numerical or categorical?
* What type of data do you have?
* Do you need high accuracy or high explainability?

### 8.3 Inference Engine

The inference engine is a rule-based inference engine used to choose suitable machine learning models and algorithms.

Example rule:

```
if has_target and target_type == "numeric":
    problem_type = "regression"

elif has_target and target_type == "categorical":
    problem_type = "classification"

elif not has_target:
    problem_type = "unsupervised_learning"
```

### 8.4 Knowledge Base

The knowledge base stores expert domain knowledge. The knowledge can be stored using JSON format.

Example knowledge base:

```
{
  "classification": {
    "tabular": {
      "models": ["Logistic Regression", "Random Forest", "XGBoost"],
      "metrics": ["Accuracy", "F1-score", "ROC-AUC"]
    }
  },
  "regression": {
    "tabular": {
      "models": ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor"],
      "metrics": ["MAE", "RMSE", "R2 Score"]
    }
  }
}
```

### 8.5 Explanation Engine

The explanation engine explains how the expert system reaches its conclusion and why it chooses specific machine learning models.

Example explanation:

> Your target column has two classes, so this is a binary classification problem. Since your data is tabular and you need explainability, Logistic Regression is recommended as a baseline model.

### 8.6 Code Generator

The code generator provides code templates for users.

Example code template for Logistic Regression:

```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))
```

---

## 9. Example Output

### Problem Type

**Binary Classification**

### Reason

Your target column has two classes, so this is a binary classification problem.

### Recommended Models

1. **Logistic Regression**

   * Good baseline model
   * Easy to explain

2. **Random Forest**

   * Good for tabular data
   * Handles non-linear relationships

3. **XGBoost**

   * Powerful for Kaggle-style problems
   * Often gives high accuracy

### Recommended Metrics

* Accuracy
* F1-score
* Precision
* Recall
* ROC-AUC

### Preprocessing Suggestions

* Handle missing values
* Encode categorical features
* Scale numerical features if using Logistic Regression
* Check class imbalance

---

## 10. System Workflow

1. User enters problem description or dataset information.
2. System asks guided questions.
3. User answers questions.
4. Inference engine identifies the machine learning problem type.
5. Knowledge base provides suitable models, metrics, and preprocessing steps.
6. Explanation engine explains the recommendation.
7. Code generator provides beginner-friendly starter code.
8. User reviews the output.

---

## 11. Implementation: Selection and Mastery of Expert System Software and Tools

The expert system can be implemented using **Python** because Python is widely used in machine learning and data science. It has many libraries for rule-based systems, web applications, and machine learning.

For example, we can use the **Experta** library to create a rule-based inference engine.

### Recommended Tools and Software

| Component             | Tool / Software                   |
| --------------------- | --------------------------------- |
| Frontend              | Streamlit                         |
| Backend Logic         | Python                            |
| Rule Engine           | Experta or custom if-else rules   |
| Knowledge Base        | JSON / YAML                       |
| Explanation           | Simple text reason                |
| Google Forms / Survey | Collect end-user testing feedback |

---

## 12. Project Roles

| Role               | Member               | Responsibility                                                                                                                        |
| ------------------ | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Domain Expert      | [placeholder]        | Provide the core machine learning and expert domain knowledge, including rules for model selection, metrics, and preprocessing steps. |
| Project Manager    | [placeholder]        | Oversee the project, manage timelines, and coordinate communication among team members.                                               |
| Knowledge Engineer | [placeholder]        | Extract, structure, and formalize the domain knowledge into rules for the Knowledge Base and design the Inference Engine logic.       |
| Programmer         | Joseph Lau Tiew Jung | Develop and implement the expert system components such as Frontend, Backend, and Rule Engine using Python and selected tools.        |
| End User           | [placeholder]        | Provide testing and feedback to validate the system's usability and effectiveness for machine learning beginners.                     |

---

## 13. Gantt Chart

[placeholder]

---

## 14. Testing: Human Expert and End-User Involvement

Testing is important to make sure **ML Compass** gives useful and correct recommendations.

### 14.1 Human Expert Testing

[placeholder]

### 14.2 End-User Testing

[placeholder]

### 14.3 Testing Criteria

[placeholder]

### 14.4 Sample Test Case

[placeholder]

---

## 15. Discussion / Results

[placeholder]

---

## 16. Future Suggestions

[placeholder]

---

## 17. Conclusion

[placeholder]

---

## 18. Appendix

[placeholder]

---

## 19. References

[placeholder]

---

## 20. Project Scope

### In Scope

To prevent this expert system from becoming too large and complex, it only supports:

* Classification
* Regression
* Clustering
* Time-series forecasting
* Basic NLP

### Out of Scope

The system does not support:

* Computer vision
* Reinforcement learning
* Automatic dataset detection
* LLM integration
* Multiple model recommendations

---

## 21. Blockers / Challenges

Possible blockers include:

* What if the user picks the wrong feature?
* Are there any mechanisms to detect and warn users about wrong input?
* Numeric feature is not always a regression problem. For example, `1` can mean `true` and `0` can mean `false`.
* How should the system handle rule conflicts?
* How should the system handle uncertainty in decision-making?
