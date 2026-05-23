import json
from pathlib import Path

import streamlit as st

from inference_engine.inference_engine import infer_recommendation


APP_DIR = Path(__file__).parent
KNOWLEDGE_BASE_PATH = APP_DIR / "knowledge_base" / "knowledge_base.json"


@st.cache_data
def load_knowledge_base():
    """Load the rule-based knowledge base."""
    with KNOWLEDGE_BASE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def unknown_bool(label, key, default="unknown"):
    options = ["unknown", True, False]
    labels = {
        "unknown": "I do not know",
        True: "Yes",
        False: "No",
    }
    index = options.index(default)
    return st.radio(label, options=options, format_func=lambda value: labels[value], index=index, horizontal=True, key=key)


def optional_number(label, key, min_value=0, max_value=None):
    known = st.checkbox(f"I know the {label.lower()}", key=f"{key}_known")
    if not known:
        return "unknown"
    kwargs = {"min_value": min_value, "step": 1, "key": key}
    if max_value is not None:
        kwargs["max_value"] = max_value
    return st.number_input(label, **kwargs)


def build_form_payload():
    with st.form("problem_form"):
        problem_description = st.text_area(
            "Problem description",
            placeholder="Example: I have customer records and want to predict whether each customer will churn.",
        )

        st.markdown("### Problem Setup")
        output_can_be_calculated_by_formula = unknown_bool(
            "Can the output be calculated directly with a formula or fixed rule?",
            "formula_gate",
            default=False,
        )
        has_target_column = unknown_bool("Do you have a target column or labels?", "has_target_column")
        goal = st.selectbox(
            "What is the main goal?",
            options=["unknown", "classification", "regression", "grouping", "anomaly_detection", "forecasting", "nlp"],
            format_func=lambda value: {
                "unknown": "I do not know",
                "classification": "Predict a category",
                "regression": "Predict a number",
                "grouping": "Group similar records",
                "anomaly_detection": "Find rare or abnormal cases",
                "forecasting": "Predict future values",
                "nlp": "Work with text or language",
            }[value],
        )
        data_type = st.selectbox(
            "What is the main data type?",
            options=["unknown", "tabular", "text", "image", "time_series", "multimodal"],
            format_func=lambda value: {
                "unknown": "I do not know",
                "tabular": "Tabular data",
                "text": "Text",
                "image": "Image",
                "time_series": "Time series",
                "multimodal": "Multiple data types",
            }[value],
        )

        text_complexity = "unknown"
        if data_type == "text":
            text_complexity = st.radio(
                "How complex is the text task?",
                options=["simple", "complex", "unknown"],
                format_func=lambda value: {
                    "simple": "Simple keyword or sentiment task",
                    "complex": "Context, word order, or meaning matters",
                    "unknown": "I do not know",
                }[value],
                horizontal=True,
            )

        st.markdown("### Target Details")
        target_type = "unknown"
        target_unique_classes = "unknown"
        can_have_multiple_labels = "unknown"
        if has_target_column is not False:
            target_type = st.radio(
                "What kind of target do you have?",
                options=["unknown", "categorical", "continuous_numeric", "numeric_label"],
                format_func=lambda value: {
                    "unknown": "I do not know",
                    "categorical": "Category or label",
                    "continuous_numeric": "Continuous number",
                    "numeric_label": "Numbers used as labels",
                }[value],
            )

            if target_type in {"categorical", "numeric_label"}:
                class_count_choice = st.radio(
                    "How many target classes are there?",
                    options=["unknown", "two", "many"],
                    format_func=lambda value: {
                        "unknown": "I do not know",
                        "two": "2",
                        "many": "More than 2",
                    }[value],
                    horizontal=True,
                )
                target_unique_classes = {"unknown": "unknown", "two": 2, "many": 3}[class_count_choice]
                can_have_multiple_labels = unknown_bool(
                    "Can one record belong to multiple labels at the same time?",
                    "can_have_multiple_labels",
                    default=False,
                )

        st.markdown("### Dataset and Constraints")
        dataset_rows = optional_number("Dataset rows", "dataset_rows", min_value=1)
        class_balance = st.radio(
            "Are the classes balanced?",
            options=["unknown", "balanced", "imbalanced"],
            format_func=lambda value: {
                "unknown": "I do not know",
                "balanced": "Balanced",
                "imbalanced": "Imbalanced",
            }[value],
            horizontal=True,
        )
        majority_class_percentage = optional_number(
            "Majority class percentage",
            "majority_class_percentage",
            min_value=0,
            max_value=100,
        )
        missing_value_percentage = optional_number(
            "Missing value percentage",
            "missing_value_percentage",
            min_value=0,
            max_value=100,
        )
        missing_value_type = st.radio(
            "What type of missing values are present?",
            options=["unknown", "numerical", "categorical", "mixed"],
            format_func=lambda value: {
                "unknown": "I do not know or none",
                "numerical": "Numerical",
                "categorical": "Categorical",
                "mixed": "Mixed",
            }[value],
            horizontal=True,
        )
        has_outliers = unknown_bool("Does the dataset have important outliers?", "has_outliers")

        st.markdown("### Project Practicality")
        priority = st.radio(
            "What is your priority?",
            options=["balanced", "transparency", "performance", "unknown"],
            format_func=lambda value: {
                "balanced": "Balanced",
                "transparency": "Explainability",
                "performance": "Highest accuracy",
                "unknown": "I do not know",
            }[value],
            horizontal=True,
        )
        domain = st.selectbox(
            "What domain is the project in?",
            options=["general", "medical", "legal", "financial", "educational", "unknown"],
            format_func=lambda value: {
                "general": "General",
                "medical": "Medical",
                "legal": "Legal",
                "financial": "Financial",
                "educational": "Educational",
                "unknown": "I do not know",
            }[value],
        )
        has_gpu = unknown_bool("Do you have GPU access?", "has_gpu")
        timeline = st.radio(
            "How much time do you have?",
            options=["unknown", "short", "medium", "long"],
            format_func=lambda value: {
                "unknown": "I do not know",
                "short": "Short",
                "medium": "Medium",
                "long": "Long",
            }[value],
            horizontal=True,
        )
        programming_skill = st.radio(
            "Programming skill level",
            options=["beginner", "intermediate", "advanced", "unknown"],
            format_func=lambda value: {
                "beginner": "Beginner",
                "intermediate": "Intermediate",
                "advanced": "Advanced",
                "unknown": "I do not know",
            }[value],
            horizontal=True,
        )

        submitted = st.form_submit_button("Get recommendation")

    payload = {
        "problem_description": problem_description,
        "output_can_be_calculated_by_formula": output_can_be_calculated_by_formula,
        "has_target_column": has_target_column,
        "target_type": target_type,
        "target_unique_classes": target_unique_classes,
        "can_have_multiple_labels": can_have_multiple_labels,
        "data_type": data_type,
        "text_complexity": text_complexity,
        "dataset_rows": dataset_rows,
        "class_balance": class_balance,
        "majority_class_percentage": majority_class_percentage,
        "missing_value_percentage": missing_value_percentage,
        "missing_value_type": missing_value_type,
        "has_outliers": has_outliers,
        "goal": goal,
        "priority": priority,
        "domain": domain,
        "has_gpu": has_gpu,
        "timeline": timeline,
        "programming_skill": programming_skill,
    }
    return submitted, payload


def render_recommendation(result):
    problem_title = result.get("problem_type", "unknown").replace("_", " ").title()
    st.subheader(problem_title)

    confidence = result.get("confidence_level", "unknown")
    status = result.get("status", "unknown").replace("_", " ").title()
    st.caption(f"Confidence: {confidence.title()} | Status: {status}")

    if result.get("classification_type"):
        st.write(f"Classification type: {result['classification_type'].replace('_', ' ').title()}")

    if result.get("explanation"):
        st.write(result["explanation"])

    for warning in result.get("warnings", []):
        st.warning(warning)

    model_tab, metric_tab, prep_tab, code_tab = st.tabs(
        ["Models", "Metrics", "Preprocessing", "Starter Code"]
    )

    with model_tab:
        st.markdown("**Baseline model**")
        for model in result.get("recommended_baseline_model", []):
            st.markdown(f"- {model}")

        advanced_models = result.get("recommended_advanced_model", [])
        if advanced_models:
            st.markdown("**Advanced model**")
            for model in advanced_models:
                st.markdown(f"- {model}")

    with metric_tab:
        for metric in result.get("evaluation_metrics", []):
            st.markdown(f"- {metric}")

    with prep_tab:
        for step in result.get("preprocessing_steps", []):
            st.markdown(f"- {step}")

    with code_tab:
        st.code(result.get("starter_code_suggestion") or "# No starter code available.", language="python")


def main():
    st.set_page_config(page_title="ML Compass", page_icon="ML", layout="centered")

    knowledge_base = load_knowledge_base()

    st.title("ML Compass")
    st.caption("A beginner-friendly expert system for machine learning model selection.")

    submitted, payload = build_form_payload()

    if submitted:
        if payload["problem_description"]:
            st.markdown("### Input Summary")
            st.write(payload["problem_description"])

        st.markdown("### Recommendation")
        result = infer_recommendation(payload, knowledge_base)
        render_recommendation(result)


if __name__ == "__main__":
    main()
