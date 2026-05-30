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


def load_custom_css():
    # Streamlit exposes limited theming hooks, so the app uses scoped CSS
    # selectors and app-specific classes for the custom visual treatment.
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --ml-bg: #F6F4E8;
            --ml-surface: #FBFAF1;
            --ml-soft: #E5EEE4;
            --ml-primary: #C0E1D2;
            --ml-accent: #DC9B9B;
            --ml-text: #1F2D23;
            --ml-muted: #475B4E;
            --ml-border: #C0D6C9;
            --ml-shadow: 0 18px 55px rgba(31, 45, 35, 0.10);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(192, 225, 210, 0.30), transparent 28rem),
                radial-gradient(circle at bottom right, rgba(220, 155, 155, 0.18), transparent 24rem),
                var(--ml-bg);
            color: var(--ml-text);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3, h4, h5, h6, p, label, span {
            letter-spacing: 0;
        }

        div[data-testid="stForm"] {
            background: rgba(251, 250, 241, 0.94);
            border: 1px solid rgba(192, 214, 201, 0.85);
            border-radius: 12px;
            box-shadow: var(--ml-shadow);
            padding: 24px;
        }

        div[data-testid="stForm"] [data-testid="stVerticalBlock"] {
            gap: 0.75rem;
        }

        .ml-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.5rem;
            margin-bottom: 1.25rem;
        }

        .ml-brand {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .ml-logo {
            width: 58px;
            height: 58px;
            display: grid;
            place-items: center;
            border: 3px solid var(--ml-primary);
            border-radius: 14px;
            color: #2E7D67;
            font-weight: 800;
            transform: rotate(45deg);
            background: rgba(251, 250, 241, 0.80);
        }

        .ml-logo span {
            transform: rotate(-45deg);
            display: block;
            font-size: 1.5rem;
        }

        .ml-title {
            font-size: clamp(2.2rem, 5vw, 3.5rem);
            line-height: 1;
            font-weight: 800;
            color: #0E312A;
            margin: 0;
        }

        .ml-subtitle {
            margin: 0.4rem 0 0;
            color: var(--ml-muted);
            font-size: 1.05rem;
        }

        .ml-badge {
            width: 58px;
            height: 58px;
            display: grid;
            place-items: center;
            border-radius: 14px;
            background: var(--ml-soft);
            border: 1px solid var(--ml-border);
            color: #1E6E5A;
            font-weight: 800;
        }

        .ml-section-title {
            color: #397D6C;
            font-size: 1.25rem;
            font-weight: 700;
            margin: 1.35rem 0 0.2rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(192, 214, 201, 0.62);
        }

        .ml-section-title.first {
            border-top: 0;
            padding-top: 0;
            margin-top: 0.35rem;
        }

        .ml-result-card {
            background: rgba(251, 250, 241, 0.96);
            border: 1px solid rgba(192, 214, 201, 0.9);
            border-radius: 12px;
            box-shadow: var(--ml-shadow);
            padding: 24px;
            margin-top: 1.5rem;
        }

        .ml-result-title {
            font-size: 2rem;
            font-weight: 800;
            color: #0E312A;
            margin: 0 0 0.75rem;
        }

        .ml-status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .ml-pill {
            border: 1px solid var(--ml-border);
            border-radius: 999px;
            background: var(--ml-soft);
            color: var(--ml-text);
            padding: 0.45rem 0.75rem;
            font-size: 0.9rem;
            font-weight: 600;
        }

        .ml-model-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 0.75rem;
            margin-top: 0.5rem;
        }

        .ml-mini-card {
            border: 1px solid var(--ml-border);
            border-radius: 8px;
            background: rgba(229, 238, 228, 0.42);
            padding: 0.85rem 0.95rem;
            font-weight: 600;
            color: var(--ml-text);
        }

        .ml-note {
            border-left: 4px solid var(--ml-primary);
            background: rgba(229, 238, 228, 0.55);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            color: var(--ml-muted);
            margin: 0.75rem 0 1rem;
        }

        div[data-testid="stTextArea"] textarea,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            border-color: rgba(192, 214, 201, 0.95);
            border-radius: 8px;
            background-color: rgba(255, 255, 250, 0.86);
        }

        div[data-testid="stTextArea"] textarea {
            min-height: 96px;
        }

        div[data-testid="stRadio"] label,
        div[data-testid="stCheckbox"] label {
            color: var(--ml-text);
        }

        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stCaptionContainer"] {
            color: var(--ml-muted);
        }

        button[kind="primaryFormSubmit"] {
            background: linear-gradient(180deg, #F2B2B4, var(--ml-accent));
            color: var(--ml-text);
            border: 0;
            border-radius: 8px;
            padding: 0.75rem 1.3rem;
            font-weight: 800;
            box-shadow: 0 10px 28px rgba(220, 155, 155, 0.32);
        }

        button[kind="primaryFormSubmit"]:hover {
            background: linear-gradient(180deg, #F4BFC1, #E4A4A4);
            color: var(--ml-text);
            border: 0;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
            border-color: rgba(220, 155, 155, 0.42);
        }

        div[data-testid="stTabs"] button {
            color: var(--ml-muted);
            font-weight: 700;
        }

        @media (max-width: 720px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .ml-header {
                align-items: flex-start;
            }

            .ml-badge {
                display: none;
            }

            .ml-title {
                font-size: 2.15rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_title(title, first=False):
    # Keep section headings consistent while avoiding repeated raw HTML in the
    # form-building code.
    class_name = "ml-section-title first" if first else "ml-section-title"
    st.markdown(f'<div class="{class_name}">{title}</div>', unsafe_allow_html=True)


def unknown_bool(label, key, default="unknown"):
    # The inference engine distinguishes "unknown" from False, so binary form
    # controls need a third explicit option instead of a checkbox.
    options = ["unknown", True, False]
    labels = {
        "unknown": "I do not know",
        True: "Yes",
        False: "No",
    }
    index = options.index(default)
    return st.radio(label, options=options, format_func=lambda value: labels[value], index=index, horizontal=True, key=key)


def optional_number(label, key, min_value=0, max_value=None, default_value=None):
    # Streamlit number inputs always return a number; this companion checkbox
    # lets users pass the engine's UNKNOWN marker when the value is unavailable.
    unknown = st.checkbox(f"I do not know the {label.lower()}", key=f"{key}_unknown")
    if unknown:
        st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            value=default_value or min_value,
            step=1,
            key=f"{key}_disabled",
            disabled=True,
        )
        return "unknown"

    return st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=default_value or min_value,
        step=1,
        key=key,
    )


def render_header():
    st.markdown(
        """
        <div class="ml-header">
            <div class="ml-brand">
                <div class="ml-logo"><span>ML</span></div>
                <div>
                    <h1 class="ml-title">ML Compass</h1>
                    <p class="ml-subtitle">A beginner-friendly expert system for machine learning model selection.</p>
                </div>
            </div>
            <div class="ml-badge">M C</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_form_payload():
    with st.form("problem_form"):
        # All widgets live inside one form so the inference engine only runs
        # after the user submits a complete snapshot of their answers.
        problem_description = st.text_area(
            "Problem description",
            placeholder="Example: I have customer records and want to predict whether each customer will churn.",
        )

        section_title("Problem Setup", first=True)
        setup_col_1, setup_col_2 = st.columns(2)
        with setup_col_1:
            output_can_be_calculated_by_formula = unknown_bool(
                "Can the output be calculated directly with a formula or fixed rule?",
                "formula_gate",
                default=False,
            )
        with setup_col_2:
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
            # Text complexity only matters for NLP model selection, so keep this
            # field hidden for non-text datasets.
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

        section_title("Target Details")
        target_type = "unknown"
        target_unique_classes = "unknown"
        can_have_multiple_labels = "unknown"
        if has_target_column is not False:
            # If the user is unsure whether labels exist, still collect target
            # details and let the engine lower confidence if answers conflict.
            target_type = st.radio(
                "What kind of target do you have?",
                options=["unknown", "categorical", "continuous_numeric", "numeric_label"],
                format_func=lambda value: {
                    "unknown": "I do not know",
                    "categorical": "Category or label",
                    "continuous_numeric": "Continuous number",
                    "numeric_label": "Numbers used as labels",
                }[value],
                horizontal=True,
            )

            if target_type in {"categorical", "numeric_label"}:
                # The engine only needs binary versus multi-class, so the UI
                # maps "More than 2" to a representative count instead of
                # forcing beginners to provide an exact class total.
                target_col_1, target_col_2 = st.columns(2)
                with target_col_1:
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
                with target_col_2:
                    can_have_multiple_labels = unknown_bool(
                        "Can one record belong to multiple labels at the same time?",
                        "can_have_multiple_labels",
                        default=False,
                    )

        section_title("Dataset and Constraints")
        data_col_1, data_col_2 = st.columns(2)
        with data_col_1:
            dataset_rows = optional_number("Dataset rows", "dataset_rows", min_value=1, default_value=1000)
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
                default_value=50,
            )
        with data_col_2:
            missing_value_percentage = optional_number(
                "Missing value percentage",
                "missing_value_percentage",
                min_value=0,
                max_value=100,
                default_value=0,
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

        section_title("Project Practicality")
        practical_col_1, practical_col_2 = st.columns(2)
        with practical_col_1:
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
            has_gpu = unknown_bool("Do you have GPU access?", "has_gpu")
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
        with practical_col_2:
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

        st.markdown('<div class="ml-note">Unknown values are allowed. ML Compass will mark uncertain recommendations clearly.</div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("Get recommendation")

    # Keep payload keys aligned with normalize_input() in the inference engine;
    # display-only fields may be included but are ignored by the rules.
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


def render_model_cards(models):
    # The same compact card grid is reused for models and metric names.
    if not models:
        st.caption("No model suggested for this section.")
        return

    cards = "".join(f'<div class="ml-mini-card">{model}</div>' for model in models)
    st.markdown(f'<div class="ml-model-grid">{cards}</div>', unsafe_allow_html=True)


def render_rule_trace(trace):
    # Expand the first few rules to expose the reasoning path without making
    # long traces overwhelming.
    if not trace:
        st.caption("No rule trace available.")
        return

    for index, item in enumerate(trace, start=1):
        title = f'{index}. {item.get("rule_id", "RULE")} - {item.get("rule_name", "Unnamed rule")}'
        with st.expander(title, expanded=index <= 3):
            st.markdown(f'**Category:** {item.get("category", "unknown")}')
            st.markdown(f'**Decision:** {item.get("decision", "No decision recorded.")}')
            st.markdown(f'**Reason:** {item.get("reason", "No reason recorded.")}')
            st.markdown(f'**Impact:** {item.get("impact", "No impact recorded.")}')
            evidence = item.get("evidence") or {}
            if evidence:
                st.markdown("**Evidence**")
                st.json(evidence)


def render_recommendation(result):
    # Result rendering is defensive because recommendations can be temporary or
    # rule-based and may omit fields that ordinary ML recommendations include.
    problem_title = result.get("problem_type", "unknown").replace("_", " ").title()
    confidence = result.get("confidence_level", "unknown")
    status = result.get("status", "unknown").replace("_", " ").title()

    st.markdown('<div class="ml-result-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="ml-result-title">{problem_title}</div>', unsafe_allow_html=True)

    pills = [
        f"Confidence: {confidence.title()}",
        f"Status: {status}",
    ]
    if result.get("classification_type"):
        pills.append(result["classification_type"].replace("_", " ").title())
    pill_markup = "".join(f'<span class="ml-pill">{pill}</span>' for pill in pills)
    st.markdown(f'<div class="ml-status-row">{pill_markup}</div>', unsafe_allow_html=True)

    if result.get("explanation"):
        st.markdown(f'<div class="ml-note">{result["explanation"]}</div>', unsafe_allow_html=True)

    for warning in result.get("warnings", []):
        st.warning(warning)

    model_tab, metric_tab, prep_tab, code_tab, trace_tab = st.tabs(
        ["Models", "Metrics", "Preprocessing", "Starter Code", "Rule Trace"]
    )

    with model_tab:
        st.markdown("**Baseline model**")
        render_model_cards(result.get("recommended_baseline_model", []))

        advanced_models = result.get("recommended_advanced_model", [])
        if advanced_models:
            st.markdown("**Advanced model**")
            render_model_cards(advanced_models)

    with metric_tab:
        render_model_cards(result.get("evaluation_metrics", []))

    with prep_tab:
        for step in result.get("preprocessing_steps", []):
            st.markdown(f"- {step}")

    with code_tab:
        st.code(result.get("starter_code_suggestion") or "# No starter code available.", language="python")

    with trace_tab:
        render_rule_trace(result.get("explanation_trace", []))

    st.markdown("</div>", unsafe_allow_html=True)

# main program start here
def main():
    st.set_page_config(page_title="ML Compass", page_icon="ML", layout="wide")
    load_custom_css()

    # The knowledge base is cached, while each submitted payload is evaluated
    # fresh so changes in form answers immediately update the recommendation.
    knowledge_base = load_knowledge_base()

    render_header()
    submitted, payload = build_form_payload()

    if submitted:
        if payload["problem_description"]:
            st.markdown("### Input Summary")
            st.write(payload["problem_description"])

        result = infer_recommendation(payload, knowledge_base)
        render_recommendation(result)


if __name__ == "__main__":
    main()
