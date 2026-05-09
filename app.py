import json
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).parent
KNOWLEDGE_BASE_PATH = APP_DIR / "knowledge_base" / "knowledge_base.json"


@st.cache_data
def load_knowledge_base():
    """read knowledge from knowledge base"""
    with KNOWLEDGE_BASE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def infer_problem_type(data_type, has_target, target_type, class_count, is_time_ordered):
    """determine the problem type based on the users input"""
    warnings = []

    if data_type == "time_series":
        return "time_series_forecasting", warnings

    if data_type == "text":
        return "basic_nlp", warnings

    if not has_target:
        return "clustering", warnings

    if target_type == "categorical":
        return "classification", warnings

    if target_type == "numeric" and class_count == "two":
        warnings.append(
            "A numeric target with only two values may represent classes, such as 0 and 1. "
            "Treating this as classification is usually better for beginner projects."
        )
        return "classification", warnings

    if target_type == "numeric" and is_time_ordered:
        return "time_series_forecasting", warnings

    if target_type == "numeric":
        return "regression", warnings

    return "classification", warnings


def get_recommendation(knowledge_base, problem_type, data_type, priority, is_imbalanced):
    """get the ML models recommendation by using the inference engine"""
    problem_rules = knowledge_base["problem_types"][problem_type]
    data_key = data_type if data_type in problem_rules["data_types"] else "tabular"
    recommendation = problem_rules["data_types"][data_key]

    models = recommendation["models"]
    if priority == "explainability":
        models = sorted(models, key=lambda model: model["explainability_rank"])
    else:
        models = sorted(models, key=lambda model: model["accuracy_rank"])

    metrics = list(recommendation["metrics"])
    if problem_type == "classification" and is_imbalanced:
        metrics = ["F1-score", "Precision", "Recall"] + [
            metric for metric in metrics if metric not in {"F1-score", "Precision", "Recall"}
        ]

    return {
        "title": problem_rules["title"],
        "reason": recommendation["reason"],
        "models": models,
        "metrics": metrics,
        "preprocessing": recommendation["preprocessing"],
        "starter_code": recommendation["starter_code"],
    }


def render_recommendation(result, warnings):
    """display the ML recommendation to the users"""
    st.subheader(result["title"])
    st.write(result["reason"])

    if warnings:
        for warning in warnings:
            st.warning(warning)

    model_tab, metric_tab, prep_tab, code_tab = st.tabs(
        ["Recommended Models", "Metrics", "Preprocessing", "Starter Code"]
    )

    with model_tab:
        for index, model in enumerate(result["models"], start=1):
            st.markdown(f"**{index}. {model['name']}**")
            st.write(model["why"])

    with metric_tab:
        for metric in result["metrics"]:
            st.markdown(f"- {metric}")

    with prep_tab:
        for step in result["preprocessing"]:
            st.markdown(f"- {step}")

    with code_tab:
        st.code(result["starter_code"], language="python")


def render_sample_size_note(sample_count):
    """render the sample note. This is just for demo
    TODO: delete this function later
    """
    if sample_count == "small":
        st.info(
            "With fewer than 50 rows, keep the model simple and treat results cautiously. "
            "Collecting more data may improve reliability more than changing algorithms."
        )
    elif sample_count == "medium":
        st.info("With 50 to 10k rows, start with simple baseline models before trying more complex ones.")
    else:
        st.info("With more than 10k rows, tree-based models and stronger validation splits become more practical.")


def main():
    st.set_page_config(page_title="ML Compass", page_icon="ML", layout="centered")

    knowledge_base = load_knowledge_base()

    st.title("ML Compass")
    st.caption("A beginner-friendly expert system for machine learning model selection.")

    with st.form("problem_form"):
        problem_description = st.text_area(
            "Problem description",
            placeholder="Example: I have customer records and want to predict whether each customer will churn.",
        )

        has_target_label = st.radio(
            "Q1. Do you have a target column (labels) you want to predict?",
            options=["Yes", "No"],
            horizontal=True,
        )
        has_target = has_target_label == "Yes"

        data_type = st.selectbox(
            "Q2. What is the format of your dataset?",
            options=["tabular", "text", "time_series"],
            format_func=lambda value: {
                "tabular": "Tabular (CSV)",
                "text": "Text",
                "time_series": "Time-series",
            }[value],
        )

        target_type = "categorical"
        class_count = "many"
        is_time_ordered = data_type == "time_series"
        is_imbalanced = False

        if has_target:
            target_type = st.radio(
                "Q3. Is the target column a number or a category/label?",
                options=["numeric", "categorical"],
                format_func=lambda value: {
                    "numeric": "Number, such as price",
                    "categorical": "Category, such as spam/ham",
                }[value],
            )

            if target_type == "categorical":
                class_count = st.radio(
                    "Q4. How many unique categories are in your target?",
                    options=["two", "many"],
                    format_func=lambda value: {
                        "two": "2",
                        "many": "More than 2",
                    }[value],
                    horizontal=True,
                )
            else:
                class_count = "continuous"
                is_time_ordered_label = st.radio(
                    "Q5. Is your data ordered by date or time?",
                    options=["Yes", "No"],
                    horizontal=True,
                )
                is_time_ordered = is_time_ordered_label == "Yes"

            is_imbalanced = st.checkbox("The classes are imbalanced")

        sample_count = st.radio(
            "Q6. How many data samples (rows) do you have?",
            options=["small", "medium", "large"],
            format_func=lambda value: {
                "small": "< 50",
                "medium": "50 - 10k",
                "large": "> 10k",
            }[value],
            horizontal=True,
        )

        priority = st.radio(
            "Q7. What is your priority?",
            options=["accuracy", "explainability"],
            format_func=lambda value: {
                "accuracy": "High Accuracy",
                "explainability": "Easy to Explain (White-box)",
            }[value],
            horizontal=True,
        )

        submitted = st.form_submit_button("Get recommendation")

    if submitted:
        problem_type, warnings = infer_problem_type(
            data_type=data_type,
            has_target=has_target,
            target_type=target_type,
            class_count=class_count,
            is_time_ordered=is_time_ordered,
        )
        result = get_recommendation(
            knowledge_base=knowledge_base,
            problem_type=problem_type,
            data_type=data_type,
            priority=priority,
            is_imbalanced=is_imbalanced,
        )

        if problem_description:
            st.markdown("### Input Summary")
            st.write(problem_description)

        st.markdown("### Recommendation")
        render_sample_size_note(sample_count)
        render_recommendation(result, warnings)


if __name__ == "__main__":
    main()
