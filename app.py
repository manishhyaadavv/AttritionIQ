import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from scipy.sparse import issparse

st.set_page_config(
    page_title="AttritionIQ | HR Analytics",
    page_icon="📊",
    layout="wide"
)

# Load original HR dataset
@st.cache_data
def load_data():
    return pd.read_csv(
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

try:
    df = load_data()
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------



# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

/* App Background */
.stApp {
    background: linear-gradient(135deg, #0B1120 0%, #111C32 100%);
    color: #E2E8F0;
}

/* Main Layout */
.block-container {
    padding: 2rem 2.5rem 3rem 2.5rem;
    max-width: 1550px;
}

/* PREMIUM SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0B1426 0%,
        #111F38 55%,
        #0B1426 100%
    ) !important;

    border-right: 1px solid #263D5C;
}

/* Sidebar Padding */
section[data-testid="stSidebar"] .block-container {
    padding: 2rem 1rem;
}

/* Sidebar Brand */
section[data-testid="stSidebar"] h1 {
    color: #F8FAFC !important;
    font-size: 27px !important;
    font-weight: 850 !important;
    letter-spacing: -0.8px;
}

/* Sidebar Description */
section[data-testid="stSidebar"] p {
    color: #94A3B8 !important;
    font-size: 13px;
}

/* Navigation Heading */
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #60A5FA !important;
    font-size: 13px !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

/* Navigation Options */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 7px;
}

/* Individual Navigation Item */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid transparent;
    border-radius: 9px;
    padding: 9px 10px;
    transition: all 0.2s ease;
}

/* Hover Effect */
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #1E3A5F;
    border-color: #3B82F6;
    transform: translateX(4px);
}

/* Selected Navigation Item */
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(
        90deg,
        #1D4ED8,
        #2563EB
    ) !important;

    border: 1px solid #60A5FA;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25);
}

/* Navigation Text */
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: #E2E8F0 !important;
    font-weight: 550;
    font-size: 13px;
}

/* Sidebar Divider */
section[data-testid="stSidebar"] hr {
    border-color: #334155;
    margin-top: 25px;
    margin-bottom: 25px;
}

/* Sidebar Information Card */
section[data-testid="stSidebar"] .stAlert {
    background: linear-gradient(
        135deg,
        #1E3A5F,
        #172B4D
    );
    border: 1px solid #31547D;
    border-radius: 12px;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #F8FAFC !important;
}

/* Sidebar Navigation */
section[data-testid="stSidebar"] label {
    color: #CBD5E1 !important;
    font-weight: 500;
}

/* Headings */
h1 {
    color: #F8FAFC !important;
    font-size: 40px !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2 {
    color: #60A5FA !important;
    font-size: 27px !important;
    font-weight: 700 !important;
}

h3 {
    color: #BFDBFE !important;
    font-size: 21px !important;
    font-weight: 650 !important;
}

/* KPI Cards */
.metric-card {
    background: linear-gradient(145deg, #1E293B, #263B59);
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #334E70;
    box-shadow: 0 5px 18px rgba(0,0,0,0.20);
    transition: all 0.25s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: #60A5FA;
    box-shadow: 0 8px 24px rgba(37,99,235,0.18);
}

.metric-title {
    color: #CBD5E1;
    font-size: 14px;
    font-weight: 500;
}

.metric-value {
    color: #60A5FA;
    font-size: 32px;
    font-weight: 800;
}

/* Streamlit Metric Widgets */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1E293B, #263B59);
    border: 1px solid #334E70;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

div[data-testid="stMetricLabel"] {
    color: #CBD5E1 !important;
}

div[data-testid="stMetricValue"] {
    color: #60A5FA !important;
    font-weight: 800;
}

/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #2563EB, #3B82F6);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 9px 20px;
    font-weight: 600;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #1D4ED8, #2563EB);
    border: 1px solid #60A5FA;
    transform: translateY(-2px);
}

/* Premium Input Fields */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #17243A !important;
    border: 1px solid #3B5273 !important;
    border-radius: 10px !important;
    min-height: 42px;
    transition: all 0.2s ease;
}

/* Input Focus Effect */
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border: 1px solid #60A5FA !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15);
}

/* Input Text */
div[data-baseweb="select"] *,
div[data-baseweb="input"] input {
    color: #F8FAFC !important;
}

/* Input Labels */
.stSelectbox label,
.stNumberInput label {
    color: #BFDBFE !important;
    font-weight: 600 !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: #17243A;
    border: 1px solid #334E70;
    border-radius: 12px;
}

/* DataFrames */
div[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 10px;
    overflow: hidden;
}

/* Containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #334E70;
    border-radius: 14px;
}

/* Dividers */
hr {
    border-color: #334155;
    margin-top: 25px;
    margin-bottom: 25px;
}

/* Text */
p, label {
    color: #E2E8F0;
}

/* Links */
a {
    color: #60A5FA !important;
}

/* Captions */
.stCaption {
    color: #94A3B8 !important;
}

/* Smooth UI */
button, input, select {
    transition: all 0.2s ease;
}
/* Prediction Form Section Headings */
div[data-testid="stForm"] h3 {
    color: #60A5FA !important;
    font-size: 19px !important;
    font-weight: 750 !important;
    padding: 10px 12px;
    margin-top: 12px;
    margin-bottom: 16px;
    background: rgba(59, 130, 246, 0.10);
    border-left: 4px solid #3B82F6;
    border-radius: 6px;
}

/* Prediction Form Container */
div[data-testid="stForm"] {
    background: rgba(30, 41, 59, 0.25);
    padding: 18px;
    border: 1px solid #334E70;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_models():
    return joblib.load("attrition_models.pkl")

try:
    trained_models = load_models()

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📊 AttritionIQ")
st.sidebar.caption("Employee Retention Analytics")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Employee Prediction",
        "Model Comparison",
        "Feature Importance",
        "SHAP Explainability",
        "Advanced Evaluation",
        "Employee Risk Segmentation",
        "Employee Risk Comparison",
        "Employee Retention Simulator"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "AI-powered employee attrition prediction "
    "and retention analytics."
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📊 AttritionIQ")
st.write(
    "Employee Attrition Prediction & Retention Analytics"
)
st.divider()

# --------------------------------------------------

# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    total_employees = len(df)
    employees_left = (df["Attrition"] == "Yes").sum()
    employees_stayed = (df["Attrition"] == "No").sum()
    attrition_rate = (employees_left / total_employees) * 100

    st.markdown("""
    <div style="
        padding: 22px 25px;
        background: linear-gradient(110deg, #172B4D, #1E3A5F);
        border: 1px solid #31547D;
        border-radius: 15px;
        margin-bottom: 25px;
    ">
        <h2 style="color:#F8FAFC; margin:0;">
            📊 HR Analytics Overview
        </h2>
        <p style="color:#BFDBFE; margin:8px 0 0 0;">
            Monitor workforce trends and employee retention insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    cards = [
        ("👥", "Total Employees", f"{total_employees:,}", "Workforce Size"),
        ("📉", "Employees Left", f"{employees_left:,}", "Attrition Count"),
        ("🤝", "Employees Stayed", f"{employees_stayed:,}", "Retained Employees"),
        ("📊", "Attrition Rate", f"{attrition_rate:.2f}%", "Overall Attrition")
    ]

    for col, card in zip([c1, c2, c3, c4], cards):
        icon, title, value, subtitle = card

        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:27px;">{icon}</div>
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
                <div style="
                    color:#94A3B8;
                    font-size:12px;
                    margin-top:8px;
                ">{subtitle}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.subheader("Employee Attrition Distribution")
    import matplotlib.pyplot as plt

    # Dynamic Attrition Data
    total = len(df)
    stayed = (df["Attrition"] == "No").sum()
    left = (df["Attrition"] == "Yes").sum()

    chart_data = pd.DataFrame({
        "Status": ["Stayed", "Left"],
        "Employees": [stayed, left]
    })

    chart_data["Percentage"] = (
        chart_data["Employees"] / total * 100
    )

    fig, ax = plt.subplots(figsize=(11, 4))

    fig.patch.set_facecolor("#111C32")
    ax.set_facecolor("#111C32")

    bars = ax.barh(
        chart_data["Status"],
        chart_data["Employees"],
        color=["#3B82F6", "#F87171"],
        height=0.48
    )

    ax.invert_yaxis()

    ax.set_title(
        "Workforce Retention Overview",
        color="white",
        fontsize=17,
        fontweight="bold",
        pad=20
    )

    ax.set_xlabel(
        "Number of Employees",
        color="#CBD5E1",
        fontsize=11
    )

    ax.tick_params(axis="x", colors="#CBD5E1")
    ax.tick_params(axis="y", colors="white", labelsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")

    ax.grid(axis="x", linestyle="--", alpha=0.18)
    ax.set_axisbelow(True)

    for bar, percentage in zip(
        bars, chart_data["Percentage"]
    ):
        width = bar.get_width()

        ax.text(
            width + total * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width):,}  ({percentage:.1f}%)",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold"
        )

    ax.set_xlim(0, total * 1.18)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "**Dataset:** IBM HR Analytics Employee "
            "Attrition & Performance"
        )

    with col2:
        st.info(
            "**Observation:** 16.12% of employees "
            "in the dataset have left the company."
        )
    st.divider()

    st.subheader("Department-wise Attrition Analysis")

    department_analysis = (
        df.groupby("Department")["Attrition"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Attrition Rate (%)")
    )

    fig, ax = plt.subplots(figsize=(10, 3.8))

    fig.patch.set_facecolor("#111C32")
    ax.set_facecolor("#111C32")

    bars = ax.barh(
        department_analysis["Department"],
        department_analysis["Attrition Rate (%)"],
        color=["#60A5FA", "#34D399", "#FBBF24"],
        height=0.5
    )

    ax.invert_yaxis()

    ax.set_xlabel("Attrition Rate (%)", color="#CBD5E1")
    ax.tick_params(axis="x", colors="#CBD5E1")
    ax.tick_params(axis="y", colors="white", labelsize=11)

    ax.set_xlim(0, 25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")

    ax.grid(axis="x", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}%",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=10
        )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("📋 View Department-wise Attrition Details"):
        st.dataframe(
            department_analysis.round(2),
            use_container_width=True,
            hide_index=True
        )
    st.divider()

    st.subheader("Job Role-wise Attrition Analysis")

    role_analysis = (
        df.groupby("JobRole")["Attrition"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Attrition Rate (%)")
        .sort_values("Attrition Rate (%)", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    fig.patch.set_facecolor("#111C32")
    ax.set_facecolor("#111C32")

    bars = ax.barh(
        role_analysis["JobRole"],
        role_analysis["Attrition Rate (%)"],
        color="#60A5FA",
        height=0.6
    )

    ax.invert_yaxis()

    ax.set_xlabel("Attrition Rate (%)", color="#CBD5E1")
    ax.tick_params(axis="x", colors="#CBD5E1")
    ax.tick_params(axis="y", colors="white", labelsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")

    ax.grid(axis="x", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)

    ax.set_xlim(0, role_analysis["Attrition Rate (%)"].max() + 8)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}%",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold"
        )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("📋 View Job Role-wise Attrition Details"):
        st.dataframe(
            role_analysis.round(2),
            use_container_width=True,
            hide_index=True
        )
    st.divider()

    st.subheader("Age Group-wise Attrition Analysis")

    df["Age Group"] = pd.cut(
        df["Age"],
        bins=[17, 25, 35, 45, 55, 65],
        labels=[
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56-65"
        ]
    )

    age_analysis = (
        df.groupby("Age Group", observed=False)["Attrition"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Attrition Rate (%)")
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    fig.patch.set_facecolor("#111C32")
    ax.set_facecolor("#111C32")

    bars = ax.barh(
        age_analysis["Age Group"].astype(str),
        age_analysis["Attrition Rate (%)"],
        color="#60A5FA",
        height=0.55
    )

    ax.invert_yaxis()

    ax.set_xlabel("Attrition Rate (%)", color="#CBD5E1")
    ax.tick_params(axis="x", colors="#CBD5E1")
    ax.tick_params(axis="y", colors="white", labelsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")

    ax.grid(axis="x", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)

    ax.set_xlim(0, age_analysis["Attrition Rate (%)"].max() + 8)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}%",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("📋 View Age Group-wise Attrition Details"):
        st.dataframe(
            age_analysis.round(2),
            use_container_width=True,
            hide_index=True
        )
    st.divider()

    st.subheader("Gender-wise Attrition Analysis")

    gender_analysis = (
        df.groupby("Gender")["Attrition"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Attrition Rate (%)")
    )

    fig, ax = plt.subplots(figsize=(10, 3.5))

    fig.patch.set_facecolor("#111C32")
    ax.set_facecolor("#111C32")

    bars = ax.barh(
        gender_analysis["Gender"],
        gender_analysis["Attrition Rate (%)"],
        color=["#60A5FA", "#F472B6"],
        height=0.5
    )

    ax.invert_yaxis()

    ax.set_xlabel("Attrition Rate (%)", color="#CBD5E1")
    ax.tick_params(axis="x", colors="#CBD5E1")
    ax.tick_params(axis="y", colors="white", labelsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")

    ax.grid(axis="x", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)

    ax.set_xlim(0, gender_analysis["Attrition Rate (%)"].max() + 8)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}%",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("📋 View Gender-wise Attrition Details"):
        st.dataframe(
            gender_analysis.round(2),
            use_container_width=True,
            hide_index=True
        )
    st.divider()
    st.subheader("🎯 Interactive HR Analytics Filters")

    st.caption(
        "Filter employees by department and job role "
        "to explore attrition patterns."
    )

    with st.container(border=True):

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            selected_department = st.selectbox(
                "🏢 Select Department",
                ["All"] + sorted(df["Department"].unique().tolist())
            )

        with filter_col2:
            selected_role = st.selectbox(
                "💼 Select Job Role",
                ["All"] + sorted(df["JobRole"].unique().tolist())
            )

    filtered_df = df.copy()

    if selected_department != "All":
        filtered_df = filtered_df[
            filtered_df["Department"] == selected_department
        ]

    if selected_role != "All":
        filtered_df = filtered_df[
            filtered_df["JobRole"] == selected_role
        ]

    total_filtered = len(filtered_df)
    left_filtered = (filtered_df["Attrition"] == "Yes").sum()

    rate_filtered = (
        left_filtered / total_filtered * 100
        if total_filtered > 0 else 0
    )

    st.markdown("### 📊 Filtered HR Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Employees",
        f"{total_filtered:,}"
    )

    c2.metric(
        "Employees Left",
        f"{left_filtered:,}"
    )

    c3.metric(
        "Attrition Rate",
        f"{rate_filtered:.2f}%"
    )

    st.divider()

    st.markdown("### 👥 Employee Records")

    st.caption("Showing up to 20 employee records based on selected filters.")

    display_columns = [
        "Age",
        "Attrition",
        "Department",
        "JobRole",
        "Gender",
        "MonthlyIncome",
        "OverTime",
        "YearsAtCompany",
        "JobSatisfaction"
    ]

    st.dataframe(
        filtered_df[display_columns].head(20),
        use_container_width=True,
        hide_index=True
    )
elif page == "Advanced Evaluation":

    st.subheader("📊 Advanced Model Evaluation")

    st.caption(
        "Detailed performance analysis of all trained "
        "machine learning models."
    )

    # Prepare test data
    X = df.drop(
        columns=[
            "Attrition",
            "EmployeeNumber",
            "EmployeeCount",
            "Over18",
            "StandardHours",
            "Age Group"
        ],
        errors="ignore"
    )

    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------------
    # CLASSIFICATION REPORT
    # -----------------------------------

    st.markdown("### 🎯 Classification Report")

    selected_name = st.selectbox(
        "Choose Model for Evaluation",
        list(trained_models.keys())
    )

    selected_model = trained_models[selected_name]

    predictions = selected_model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    # KPI Cards
    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Accuracy",
        f"{report['accuracy']:.2%}"
    )

    m2.metric(
        "Precision",
        f"{report['Yes']['precision']:.2%}"
    )

    m3.metric(
        "Recall",
        f"{report['Yes']['recall']:.2%}"
    )

    m4.metric(
        "F1 Score",
        f"{report['Yes']['f1-score']:.2%}"
    )

    # Report Table
    report_df = pd.DataFrame(report).transpose()

    report_df = report_df.loc[
        ["No", "Yes", "macro avg", "weighted avg"],
        ["precision", "recall", "f1-score", "support"]
    ]

    st.dataframe(
        report_df.style.format({
            "precision": "{:.2%}",
            "recall": "{:.2%}",
            "f1-score": "{:.2%}",
            "support": "{:.0f}"
        }),
        use_container_width=True
    )

    st.info(
        f"Currently evaluating **{selected_name}**. "
        "Precision and Recall above focus on employees "
        "who actually left (Attrition = Yes)."
    )

    st.divider()

    # -----------------------------------
    # ROC CURVE
    # -----------------------------------

    st.markdown("### 📈 ROC Curve Comparison")

    fig, ax = plt.subplots(figsize=(10, 5.5))

    fig.patch.set_facecolor("#111C32")
    ax.set_facecolor("#111C32")

    curve_colors = ["#60A5FA", "#FBBF24", "#34D399"]

    for i, (name, model) in enumerate(trained_models.items()):

        probabilities = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities,
            pos_label="Yes"
        )

        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            linewidth=2.5,
            color=curve_colors[i % len(curve_colors)],
            label=f"{name} (AUC = {roc_auc:.3f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#F87171",
        linewidth=1.5,
        label="Random Classifier"
    )

    ax.set_xlabel(
        "False Positive Rate",
        color="#CBD5E1",
        fontsize=11
    )

    ax.set_ylabel(
        "True Positive Rate",
        color="#CBD5E1",
        fontsize=11
    )

    ax.set_title(
        "ROC Curve Comparison",
        color="white",
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    ax.tick_params(colors="#CBD5E1")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax.grid(
        True,
        linestyle="--",
        alpha=0.18
    )

    legend = ax.legend(
        facecolor="#1E293B",
        edgecolor="#334155",
        fontsize=9
    )

    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "A higher ROC-AUC generally indicates better "
        "separation between employees who leave and stay."
    )

    st.divider()

    # -----------------------------------
    # PRECISION-RECALL CURVE
    # -----------------------------------

    st.markdown("### 🎯 Precision-Recall Curve")

    fig, ax = plt.subplots(figsize=(10, 5.5))

    fig.patch.set_facecolor("#111C32")
    ax.set_facecolor("#111C32")

    for i, (name, model) in enumerate(trained_models.items()):

        probabilities = model.predict_proba(X_test)[:, 1]

        precision, recall, _ = precision_recall_curve(
            y_test,
            probabilities,
            pos_label="Yes"
        )

        ap = average_precision_score(
            y_test,
            probabilities,
            pos_label="Yes"
        )

        ax.plot(
            recall,
            precision,
            linewidth=2.5,
            color=curve_colors[i % len(curve_colors)],
            label=f"{name} (AP = {ap:.3f})"
        )

    ax.set_xlabel(
        "Recall",
        color="#CBD5E1",
        fontsize=11
    )

    ax.set_ylabel(
        "Precision",
        color="#CBD5E1",
        fontsize=11
    )

    ax.set_title(
        "Precision-Recall Curve Comparison",
        color="white",
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    ax.tick_params(colors="#CBD5E1")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax.grid(
        True,
        linestyle="--",
        alpha=0.18
    )

    legend = ax.legend(
        facecolor="#1E293B",
        edgecolor="#334155",
        fontsize=9
    )

    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Average Precision (AP) summarizes the "
        "Precision-Recall curve. It is especially useful "
        "when the target classes are imbalanced."
    )

# --------------------------------------------------
# EMPLOYEE PREDICTION
# --------------------------------------------------

elif page == "Employee Prediction":

    st.subheader("🔍 Employee Attrition Prediction")

    st.write(
        "Enter employee information to estimate "
        "attrition probability."
    )

    with st.form("employee_form"):

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### 👤 Personal Information")

            age = st.number_input("Age", 18, 60, 30)

            gender = st.selectbox(
                "Gender", ["Male", "Female"]
            )

            marital_status = st.selectbox(
                "Marital Status",
                ["Single", "Married", "Divorced"]
            )

            distance_from_home = st.number_input(
                "Distance From Home", 1, 30, 5
            )

            education = st.selectbox(
                "Education", [1, 2, 3, 4, 5]
            )

            education_field = st.selectbox(
                "Education Field",
                [
                    "Life Sciences",
                    "Medical",
                    "Marketing",
                    "Technical Degree",
                    "Other",
                    "Human Resources"
                ]
            )

            st.markdown("### 💼 Job Information")

            department = st.selectbox(
                "Department",
                [
                    "Sales",
                    "Research & Development",
                    "Human Resources"
                ]
            )

            job_role = st.selectbox(
                "Job Role",
                [
                    "Sales Executive",
                    "Research Scientist",
                    "Laboratory Technician",
                    "Manufacturing Director",
                    "Healthcare Representative",
                    "Manager",
                    "Sales Representative",
                    "Research Director",
                    "Human Resources"
                ]
            )

            job_level = st.selectbox(
                "Job Level", [1, 2, 3, 4, 5]
            )

            business_travel = st.selectbox(
                "Business Travel",
                [
                    "Non-Travel",
                    "Travel_Rarely",
                    "Travel_Frequently"
                ]
            )

            overtime = st.selectbox(
                "OverTime", ["No", "Yes"]
            )

            job_involvement = st.selectbox(
                "Job Involvement", [1, 2, 3, 4]
            )

            job_satisfaction = st.selectbox(
                "Job Satisfaction", [1, 2, 3, 4]
            )

            environment_satisfaction = st.selectbox(
                "Environment Satisfaction", [1, 2, 3, 4]
            )

            relationship_satisfaction = st.selectbox(
                "Relationship Satisfaction", [1, 2, 3, 4]
            )

        with col2:

            st.markdown("### 💰 Compensation & Benefits")

            monthly_income = st.number_input(
                "Monthly Income", 1000, 20000, 5000
            )

            daily_rate = st.number_input(
                "Daily Rate", 100, 1500, 800
            )

            hourly_rate = st.number_input(
                "Hourly Rate", 30, 100, 65
            )

            monthly_rate = st.number_input(
                "Monthly Rate", 2000, 27000, 14000
            )

            percent_salary_hike = st.number_input(
                "Percent Salary Hike", 10, 25, 15
            )

            stock_option_level = st.selectbox(
                "Stock Option Level", [0, 1, 2, 3]
            )

            performance_rating = st.selectbox(
                "Performance Rating", [3, 4]
            )

            st.markdown("### 📈 Experience & Career Growth")

            total_working_years = st.number_input(
                "Total Working Years", 0, 40, 8
            )

            years_at_company = st.number_input(
                "Years At Company", 0, 40, 5
            )

            years_in_current_role = st.number_input(
                "Years In Current Role", 0, 20, 3
            )

            years_with_curr_manager = st.number_input(
                "Years With Current Manager", 0, 20, 3
            )

            years_since_last_promotion = st.number_input(
                "Years Since Last Promotion", 0, 15, 1
            )

            num_companies_worked = st.number_input(
                "Number Of Companies Worked", 0, 20, 2
            )

            training_times_last_year = st.number_input(
                "Training Times Last Year", 0, 6, 3
            )

            work_life_balance = st.selectbox(
                "Work Life Balance", [1, 2, 3, 4]
            )

        submitted = st.form_submit_button(
            "🚀 Predict Attrition",
            use_container_width=True
        )

    if submitted:

        employee = pd.DataFrame([{
            "Age": age,
            "BusinessTravel": business_travel,
            "DailyRate": daily_rate,
            "Department": department,
            "DistanceFromHome": distance_from_home,
            "Education": education,
            "EducationField": education_field,
            "EnvironmentSatisfaction": environment_satisfaction,
            "Gender": gender,
            "HourlyRate": hourly_rate,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level,
            "JobRole": job_role,
            "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital_status,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies_worked,
            "OverTime": overtime,
            "PercentSalaryHike": percent_salary_hike,
            "PerformanceRating": performance_rating,
            "RelationshipSatisfaction": relationship_satisfaction,
            "StockOptionLevel": stock_option_level,
            "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times_last_year,
            "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_current_role,
            "YearsSinceLastPromotion": years_since_last_promotion,
            "YearsWithCurrManager": years_with_curr_manager
        }])

        selected_model = trained_models["Gradient Boosting"]

        prediction = selected_model.predict(employee)[0]

        probability = selected_model.predict_proba(
            employee
        )[0][1]

        st.session_state["last_employee"] = employee
        st.session_state["last_prediction"] = prediction
        st.session_state["last_probability"] = probability

        st.divider()
        st.subheader("Prediction Result")

        c1, c2 = st.columns(2)

        with c1:

            if prediction == "Yes":
                st.error("⚠️ Higher predicted attrition risk")
            else:
                st.success("✅ Lower predicted attrition risk")

        with c2:

            st.metric(
                "Predicted Attrition Probability",
                f"{probability * 100:.2f}%"
            )

        st.progress(float(probability))

        st.caption(
            "This prediction is an estimate from the trained model, "
            "not a certainty or an HR decision."
        )

        st.info(
            "Open the SHAP Explainability page to inspect "
            "the model factors behind this prediction."
        )

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

elif page == "Model Comparison":

    st.subheader("📈 Machine Learning Model Comparison")
    st.caption(
        "Compare the performance of three trained models "
        "across five evaluation metrics."
    )

    results = pd.DataFrame({
        "Model": [
            "Random Forest",
            "Extra Trees",
            "Gradient Boosting"
        ],
        "Accuracy": [0.8503, 0.8537, 0.7993],
        "Precision": [0.7143, 0.6667, 0.3960],
        "Recall": [0.1064, 0.1702, 0.4894],
        "F1 Score": [0.1852, 0.2712, 0.4381],
        "ROC-AUC": [0.7842, 0.8032, 0.7694]
    })

    # TOP PERFORMANCE HIGHLIGHTS
    st.markdown("### 🏆 Performance Highlights")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Best Recall</div>
            <div class="metric-value">48.94%</div>
            <p>Gradient Boosting</p>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Best F1 Score</div>
            <div class="metric-value">43.81%</div>
            <p>Gradient Boosting</p>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Best ROC-AUC</div>
            <div class="metric-value">0.8032</div>
            <p>Extra Trees</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # MODEL COMPARISON TABLE
    st.markdown("### 📊 Detailed Model Performance")

    styled_results = (
        results.style
        .format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}",
            "ROC-AUC": "{:.4f}"
        })
        .background_gradient(
            subset=[
                "Accuracy", "Precision", "Recall",
                "F1 Score", "ROC-AUC"
            ],
            cmap="Blues"
        )
        .set_properties(**{
            "text-align": "center",
            "padding": "12px"
        })
    )

    st.dataframe(
        styled_results,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # INTERACTIVE PERFORMANCE CHART
    st.markdown("### 📉 Interactive Performance Analysis")

    metric_choice = st.selectbox(
        "Choose a performance metric",
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    )

    import matplotlib.pyplot as plt

    chart_df = results.sort_values(
        by=metric_choice,
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    fig.patch.set_facecolor("#111C32")
    ax.set_facecolor("#111C32")

    values = chart_df[metric_choice]
    best_index = values.idxmax()

    colors = [
        "#22C55E" if idx == best_index else "#3B82F6"
        for idx in chart_df.index
    ]

    bars = ax.barh(
        chart_df["Model"],
        values,
        color=colors,
        height=0.55
    )

    ax.set_title(
        f"{metric_choice} Comparison",
        color="white",
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    ax.set_xlabel(
        "Score",
        color="#CBD5E1"
    )

    ax.tick_params(axis="x", colors="#CBD5E1")
    ax.tick_params(axis="y", colors="white", labelsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")

    ax.grid(axis="x", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        label = (
            f"{value:.4f}"
            if metric_choice == "ROC-AUC"
            else f"{value:.2%}"
        )

        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            color="white",
            fontweight="bold",
            fontsize=11
        )

    ax.set_xlim(0, max(values) * 1.25)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # MODEL INSIGHTS
    st.markdown("### 💡 Key Findings")

    st.success(
        "Gradient Boosting achieved the highest Recall (48.94%) "
        "and F1 Score (43.81%). It is useful when identifying "
        "more employees who may be at risk of attrition is a priority."
    )

    st.info(
        "Extra Trees achieved the highest ROC-AUC (0.8032), "
        "indicating the strongest overall ranking performance "
        "among the three models in this evaluation."
    )

    st.caption(
        "Note: Model performance is based on the current "
        "train-test evaluation. Results may vary on new data."
    )
# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

elif page == "Feature Importance":

    st.subheader("⭐ Feature Importance Analysis")
    st.caption(
        "Understand which employee attributes contribute most "
        "to the Gradient Boosting model's predictions."
    )

    model = trained_models["Gradient Boosting"]

    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importances = model.named_steps[
        "classifier"
    ].feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(
        "Importance",
        ascending=False
    ).head(15).copy()

    # CLEAN FEATURE NAMES
    importance_df["Feature"] = (
        importance_df["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("_", " ", regex=False)
    )

    # TOP 3 FEATURE CARDS
    st.markdown("### 🏆 Top 3 Influential Features")

    top3 = importance_df.head(3)
    c1, c2, c3 = st.columns(3)

    for col, (_, row) in zip(
        [c1, c2, c3],
        top3.iterrows()
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">
                    {row['Feature']}
                </div>
                <div class="metric-value">
                    {row['Importance']:.2%}
                </div>
                <p>Feature Importance</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # HORIZONTAL FEATURE IMPORTANCE CHART
    st.markdown("### 📊 Top 15 Feature Importance Ranking")

    import matplotlib.pyplot as plt

    plot_df = importance_df.sort_values(
        "Importance",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    fig.patch.set_facecolor("#111C32")
    ax.set_facecolor("#111C32")

    bars = ax.barh(
        plot_df["Feature"],
        plot_df["Importance"],
        color="#3B82F6",
        height=0.65
    )

    ax.set_title(
        "Most Important Attrition Prediction Features",
        color="white",
        fontsize=15,
        fontweight="bold",
        pad=18
    )

    ax.set_xlabel(
        "Importance Score",
        color="#CBD5E1"
    )

    ax.tick_params(axis="x", colors="#CBD5E1")
    ax.tick_params(axis="y", colors="white", labelsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")

    ax.grid(axis="x", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2%}",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold"
        )

    ax.set_xlim(
        0,
        plot_df["Importance"].max() * 1.25
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # DETAILED TABLE
    st.markdown("### 💡 Key Insights")

    insights = [
        (
            "💰 Monthly Income",
            "9.62%",
            "Highest model importance among the selected features."
        ),
        (
            "👤 Age",
            "8.39%",
            "Age is another important feature used by the model."
        ),
        (
            "⏰ Overtime",
            "8.14%",
            "Overtime status contributes significantly to model predictions."
        )
    ]

    cols = st.columns(3)

    for col, (title, value, desc) in zip(cols, insights):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value">{value}</div>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.info(
        "Note: Feature importance shows the contribution of "
        "features to model predictions. It does not prove causation."
    )

    st.divider()

    # HR RETENTION RECOMMENDATIONS
    st.markdown("### 💡 HR Retention Recommendations")

    st.caption(
        "Practical retention strategies based on the analysis "
        "and observed employee patterns."
    )

    recommendations = [
        (
            "⏰ Workload Management",
            "Review overtime patterns and improve workload "
            "distribution across teams."
        ),
        (
            "😊 Employee Satisfaction",
            "Conduct regular feedback sessions and satisfaction "
            "surveys to understand employee concerns."
        ),
        (
            "💰 Compensation Review",
            "Review compensation structures and career growth "
            "opportunities."
        ),
        (
            "📈 Career Development",
            "Support employees through training, mentoring "
            "and internal mobility."
        ),
        (
            "🤝 Employee Engagement",
            "Monitor workplace experience and manager-employee "
            "relationships."
        )
    ]

    rec_cols = st.columns(2)

    for i, (title, description) in enumerate(recommendations):
        with rec_cols[i % 2]:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:15px;">
                <div style="color:#60A5FA;
                            font-size:17px;
                            font-weight:700;
                            margin-bottom:10px;">
                    {title}
                </div>
                <div style="color:#CBD5E1;
                            font-size:14px;">
                    {description}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.info(
        "Note: Feature importance indicates how much a feature "
        "contributes to the model's predictive decisions. "
        "It does not establish a causal relationship with attrition."
    )
# --------------------------------------------------
# SHAP EXPLAINABILITY
# --------------------------------------------------

elif page == "SHAP Explainability":

    st.subheader("🧠 Explainable AI — SHAP Analysis")

    st.write(
        "Understand how different employee attributes "
        "influence the model's attrition predictions."
    )

    model = trained_models["Gradient Boosting"]
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    # ----------------------------------
    # GLOBAL FEATURE IMPORTANCE
    # ----------------------------------

    st.markdown("### 🌍 Global Feature Importance")

    global_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(
        "Importance",
        ascending=False
    ).head(10)

    global_importance["Feature"] = (
        global_importance["Feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )

    top_cols = st.columns(3)

    for col, (_, row) in zip(
        top_cols,
        global_importance.head(3).iterrows()
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">
                    ⭐ {row['Feature']}
                </div>
                <div class="metric-value">
                    {row['Importance']:.2%}
                </div>
                <p>Model Feature Importance</p>
            </div>
            """, unsafe_allow_html=True)

    st.write("")

    fig, ax = plt.subplots(figsize=(10, 5))

    fig.patch.set_facecolor("#111C32")
    ax.set_facecolor("#111C32")

    plot_data = global_importance.sort_values(
        "Importance",
        ascending=True
    )

    bars = ax.barh(
        plot_data["Feature"],
        plot_data["Importance"],
        color="#3B82F6",
        height=0.6
    )

    ax.set_xlabel("Feature Importance", color="white")
    ax.set_title(
        "Top 10 Global Features",
        color="white",
        fontsize=15,
        fontweight="bold"
    )

    ax.tick_params(colors="white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")

    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2%}",
            va="center",
            color="white",
            fontsize=9
        )

    ax.set_xlim(
        0,
        plot_data["Importance"].max() * 1.18
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Global feature importance shows which features "
        "the model relies on most across its predictions. "
        "It does not prove causation."
    )

    st.divider()

    # ----------------------------------
    # INDIVIDUAL EMPLOYEE EXPLANATION
    # ----------------------------------

    st.markdown("### 👤 Individual Employee Explanation")

    if "last_employee" not in st.session_state:

        st.warning(
            "No employee prediction found. "
            "Visit Employee Prediction, submit an employee profile, "
            "and return here to view its SHAP explanation."
        )

    else:

        employee = st.session_state["last_employee"]
        prediction = st.session_state["last_prediction"]
        probability = st.session_state["last_probability"]

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Attrition Status",
                prediction
            )

        with col2:
            st.metric(
                "Estimated Attrition Probability",
                f"{probability * 100:.2f}%"
            )

        try:

            transformed_employee = preprocessor.transform(employee)

            if issparse(transformed_employee):
                transformed_employee = transformed_employee.toarray()

            explainer = shap.TreeExplainer(classifier)

            shap_values = explainer.shap_values(
                transformed_employee
            )

            shap_values = np.asarray(shap_values)

            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, -1]

            local_values = shap_values[0]

            explanation_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": local_values
            })

            explanation_df["Absolute Impact"] = (
                explanation_df["SHAP Value"].abs()
            )

            explanation_df = explanation_df.sort_values(
                "Absolute Impact",
                ascending=False
            ).head(12)

            explanation_df["Feature"] = (
                explanation_df["Feature"]
                .str.replace("num__", "", regex=False)
                .str.replace("cat__", "", regex=False)
            )

            explanation_df = explanation_df.sort_values(
                "SHAP Value"
            )

            st.markdown("#### 🔎 Factors Influencing This Prediction")

            fig, ax = plt.subplots(figsize=(10, 6))

            fig.patch.set_facecolor("#111C32")
            ax.set_facecolor("#111C32")

            colors = [
                "#F87171" if value > 0 else "#34D399"
                for value in explanation_df["SHAP Value"]
            ]

            ax.barh(
                explanation_df["Feature"],
                explanation_df["SHAP Value"],
                color=colors
            )

            ax.axvline(
                0,
                color="white",
                linewidth=1
            )

            ax.set_xlabel(
                "SHAP Value",
                color="white"
            )

            ax.set_title(
                "Individual Employee SHAP Impact",
                color="white",
                fontsize=14,
                fontweight="bold"
            )

            ax.tick_params(colors="white")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_color("#334155")

            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("#### 📋 Detailed SHAP Values")

            display_explanation = explanation_df[
                ["Feature", "SHAP Value"]
            ].copy()

            display_explanation["SHAP Value"] = (
                display_explanation["SHAP Value"].map(
                    lambda x: f"{x:.5f}"
                )
            )

            st.dataframe(
                display_explanation,
                use_container_width=True,
                hide_index=True
            )

            st.success(
                "Green bars indicate negative SHAP values; "
                "red bars indicate positive SHAP values."
            )

            st.caption(
                "Positive SHAP values push the model output "
                "towards attrition, while negative values push "
                "it away from attrition. SHAP explains model "
                "behaviour, not real-world causation."
            )

        except Exception as e:

            st.error(
                f"SHAP explanation could not be generated: {e}"
            )

# --------------------------------------------------
# --------------------------------------------------
# EMPLOYEE RISK SEGMENTATION
# --------------------------------------------------

elif page == "Employee Risk Segmentation":

    st.subheader("🚦 Employee Attrition Risk Segmentation")

    st.caption(
        "AI-powered identification of employees with potential "
        "attrition risk using the Gradient Boosting model."
    )

    # -----------------------------------
    # RISK PREDICTION
    # -----------------------------------

    risk_df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

    risk_X = risk_df.drop(
        columns=[
            "Attrition",
            "EmployeeNumber",
            "EmployeeCount",
            "Over18",
            "StandardHours"
        ]
    )

    risk_model = trained_models["Gradient Boosting"]

    risk_probabilities = risk_model.predict_proba(risk_X)[:, 1]

    risk_df["Risk Probability"] = risk_probabilities * 100

    def assign_risk(probability):
        if probability >= 60:
            return "High Risk"
        elif probability >= 30:
            return "Medium Risk"
        else:
            return "Low Risk"

    risk_df["Risk Category"] = risk_df[
        "Risk Probability"
    ].apply(assign_risk)

    high_risk_df = risk_df[
        risk_df["Risk Category"] == "High Risk"
    ].sort_values(
        "Risk Probability",
        ascending=False
    )

    medium_risk_df = risk_df[
        risk_df["Risk Category"] == "Medium Risk"
    ]

    low_risk_df = risk_df[
        risk_df["Risk Category"] == "Low Risk"
    ]

    high = len(high_risk_df)
    medium = len(medium_risk_df)
    low = len(low_risk_df)

    # -----------------------------------
    # RISK OVERVIEW CARDS
    # -----------------------------------

    st.markdown("### 📊 Workforce Risk Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div style="
            background:linear-gradient(135deg,#451A1A,#7F1D1D);
            padding:22px;
            border-radius:15px;
            border:1px solid #EF4444;
        ">
            <div style="color:#FCA5A5;font-size:15px;">
                🔴 HIGH RISK
            </div>
            <div style="font-size:34px;font-weight:800;color:white;">
                {high:,}
            </div>
            <div style="color:#FECACA;font-size:12px;">
                Probability ≥ 60%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="
            background:linear-gradient(135deg,#422006,#78350F);
            padding:22px;
            border-radius:15px;
            border:1px solid #F59E0B;
        ">
            <div style="color:#FCD34D;font-size:15px;">
                🟡 MEDIUM RISK
            </div>
            <div style="font-size:34px;font-weight:800;color:white;">
                {medium:,}
            </div>
            <div style="color:#FDE68A;font-size:12px;">
                Probability 30%–59.99%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div style="
            background:linear-gradient(135deg,#064E3B,#065F46);
            padding:22px;
            border-radius:15px;
            border:1px solid #10B981;
        ">
            <div style="color:#6EE7B7;font-size:15px;">
                🟢 LOW RISK
            </div>
            <div style="font-size:34px;font-weight:800;color:white;">
                {low:,}
            </div>
            <div style="color:#A7F3D0;font-size:12px;">
                Probability below 30%
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # -----------------------------------
    # RISK DISTRIBUTION CHART
    # -----------------------------------

    st.markdown("### 📈 Employee Risk Distribution")

    fig, ax = plt.subplots(figsize=(10, 4))

    fig.patch.set_facecolor("#111C32")
    ax.set_facecolor("#111C32")

    categories = ["High Risk", "Medium Risk", "Low Risk"]
    values = [high, medium, low]
    colors = ["#F87171", "#FBBF24", "#34D399"]

    bars = ax.barh(
        categories,
        values,
        color=colors,
        height=0.55
    )

    ax.invert_yaxis()

    ax.set_title(
        "Workforce Risk Segmentation",
        color="white",
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel(
        "Number of Employees",
        color="#CBD5E1"
    )

    ax.tick_params(colors="#CBD5E1")

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.2
    )

    ax.set_axisbelow(True)

    for bar in bars:
        width = bar.get_width()

        ax.text(
            width + 8,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width):,}",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=11
        )

    ax.set_xlim(0, max(values) * 1.15)

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # -----------------------------------
    # HIGH RISK EMPLOYEE TABLE
    # -----------------------------------

    st.markdown("### 🔴 High Risk Employee Profiles")

    st.caption(
        "Employees with the highest predicted attrition probabilities."
    )

    display_risk_df = high_risk_df[
        [
            "Age",
            "Department",
            "JobRole",
            "OverTime",
            "YearsAtCompany",
            "Risk Probability",
            "Risk Category"
        ]
    ].copy()

    display_risk_df["Risk Probability"] = (
        display_risk_df["Risk Probability"].map(
            lambda x: f"{x:.2f}%"
        )
    )

    st.dataframe(
        display_risk_df,
        use_container_width=True,
        hide_index=True,
        height=350
    )

    st.caption(
        f"Showing {high:,} employees classified as High Risk."
    )

    st.divider()

    # -----------------------------------
    # DOWNLOAD REPORTS
    # -----------------------------------

    st.markdown("### 📥 Download HR Risk Reports")

    d1, d2 = st.columns(2)

    with d1:
        high_risk_csv = high_risk_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download High-Risk Employees",
            data=high_risk_csv,
            file_name="high_risk_employees.csv",
            mime="text/csv",
            use_container_width=True
        )

    with d2:
        all_risk_csv = risk_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Complete Risk Report",
            data=all_risk_csv,
            file_name="employee_risk_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.divider()

    # -----------------------------------
    # HR RETENTION ACTION CENTER
    # -----------------------------------

    st.markdown("### 🎯 HR Retention Action Center")

    st.caption(
        "Suggested retention actions based on the model's risk analysis."
    )

    high_risk_count = len(high_risk_df)

    overtime_count = (
        high_risk_df["OverTime"].eq("Yes").sum()
    )

    overtime_percentage = (
        overtime_count / high_risk_count * 100
        if high_risk_count > 0 else 0
    )

    if high_risk_count > 0:
        top_department = (
            high_risk_df["Department"].value_counts().idxmax()
        )
    else:
        top_department = "None"

    a1, a2, a3 = st.columns(3)

    a1.metric(
        "High-Risk Employees",
        f"{high_risk_count:,}"
    )

    a2.metric(
        "High-Risk Employees with Overtime",
        f"{overtime_count:,}"
    )

    a3.metric(
        "Overtime Among High-Risk Employees",
        f"{overtime_percentage:.2f}%"
    )

    st.markdown("#### Recommended HR Actions")

    action1, action2 = st.columns(2)

    with action1:
        st.warning(
            "**1. Workload Management**\n\n"
            "Review overtime patterns and improve workload "
            "distribution among high-risk employees."
        )

        st.info(
            "**2. Employee Engagement**\n\n"
            "Conduct regular feedback sessions and employee "
            "satisfaction surveys."
        )

    with action2:
        st.success(
            "**3. Department Review**\n\n"
            f"Prioritize retention initiatives in the "
            f"{top_department} department, which has the "
            "highest number of high-risk employees."
        )

        st.info(
            "**4. Career Development**\n\n"
            "Provide mentoring, training and internal "
            "growth opportunities."
        )

    st.caption(
        "Note: Risk categories represent model estimates, "
        "not certainty that an employee will leave. "
        "Recommendations do not establish causation."
    )
# --------------------------------------------------
# EMPLOYEE RISK COMPARISON
# --------------------------------------------------

elif page == "Employee Risk Comparison":

    st.subheader("⚖️ Employee Attrition Risk Comparison")

    st.caption(
        "Compare employee attrition risk using the Gradient Boosting model."
    )

    compare_df = pd.read_csv(
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

    compare_X = compare_df.drop(
        columns=[
            "Attrition",
            "EmployeeNumber",
            "EmployeeCount",
            "Over18",
            "StandardHours"
        ]
    )

    compare_model = trained_models["Gradient Boosting"]

    employee_ids = compare_df["EmployeeNumber"].tolist()

    st.markdown("### 👥 Select Employees")

    col1, col2 = st.columns(2)

    with col1:
        emp1_id = st.selectbox(
            "Employee 1",
            employee_ids,
            index=0
        )

    with col2:
        emp2_id = st.selectbox(
            "Employee 2",
            employee_ids,
            index=1
        )

    if emp1_id == emp2_id:

        st.warning("Please select two different employees.")

    else:

        idx1 = compare_df.index[
            compare_df["EmployeeNumber"] == emp1_id
        ][0]

        idx2 = compare_df.index[
            compare_df["EmployeeNumber"] == emp2_id
        ][0]

        employee1 = compare_X.loc[[idx1]]
        employee2 = compare_X.loc[[idx2]]

        prob1 = compare_model.predict_proba(employee1)[0][1] * 100
        prob2 = compare_model.predict_proba(employee2)[0][1] * 100

        def get_risk(prob):
            if prob >= 60:
                return "High Risk"
            elif prob >= 30:
                return "Medium Risk"
            return "Low Risk"

        risk1 = get_risk(prob1)
        risk2 = get_risk(prob2)

        st.divider()

        # RISK SUMMARY CARDS

        st.markdown("### 📊 Attrition Risk Overview")

        col1, col2 = st.columns(2)

        with col1:

            color1 = (
                "#F87171" if risk1 == "High Risk"
                else "#FBBF24" if risk1 == "Medium Risk"
                else "#34D399"
            )

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg,#1E293B,#263B59);
                padding:25px;
                border-radius:15px;
                border:1px solid {color1};
                min-height:145px;
            ">
                <div style="color:#CBD5E1;font-size:14px;">
                    EMPLOYEE {emp1_id}
                </div>
                <div style="
                    font-size:34px;
                    font-weight:800;
                    color:{color1};
                    margin-top:10px;
                ">
                    {prob1:.2f}%
                </div>
                <div style="color:{color1};font-weight:600;">
                    {risk1}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:

            color2 = (
                "#F87171" if risk2 == "High Risk"
                else "#FBBF24" if risk2 == "Medium Risk"
                else "#34D399"
            )

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg,#1E293B,#263B59);
                padding:25px;
                border-radius:15px;
                border:1px solid {color2};
                min-height:145px;
            ">
                <div style="color:#CBD5E1;font-size:14px;">
                    EMPLOYEE {emp2_id}
                </div>
                <div style="
                    font-size:34px;
                    font-weight:800;
                    color:{color2};
                    margin-top:10px;
                ">
                    {prob2:.2f}%
                </div>
                <div style="color:{color2};font-weight:600;">
                    {risk2}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # COMPARISON CHART

        st.markdown("### 📈 Employee Risk Comparison")

        fig, ax = plt.subplots(figsize=(10, 3.5))

        fig.patch.set_facecolor("#111C32")
        ax.set_facecolor("#111C32")

        employees = [
            f"Employee {emp1_id}",
            f"Employee {emp2_id}"
        ]

        probabilities = [prob1, prob2]

        colors = [color1, color2]

        bars = ax.barh(
            employees,
            probabilities,
            color=colors,
            height=0.45
        )

        ax.invert_yaxis()

        ax.set_xlim(0, 110)

        ax.set_xlabel(
            "Predicted Attrition Probability (%)",
            color="#CBD5E1"
        )

        ax.tick_params(axis="x", colors="#CBD5E1")
        ax.tick_params(axis="y", colors="white", labelsize=11)

        ax.set_title(
            "Employee Risk Comparison",
            color="white",
            fontsize=14,
            fontweight="bold",
            pad=15
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#334155")

        ax.grid(axis="x", linestyle="--", alpha=0.2)
        ax.set_axisbelow(True)

        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 2,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.2f}%",
                va="center",
                color="white",
                fontweight="bold"
            )

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        # EMPLOYEE PROFILE

        st.markdown("### 👤 Employee Profile Comparison")

        profile_columns = [
            "Age",
            "Department",
            "JobRole",
            "OverTime",
            "YearsAtCompany",
            "MonthlyIncome",
            "JobSatisfaction"
        ]

        profile_df = pd.DataFrame({
            "Employee Feature": profile_columns,
            f"Employee {emp1_id}": [
                compare_df.loc[idx1, col]
                for col in profile_columns
            ],
            f"Employee {emp2_id}": [
                compare_df.loc[idx2, col]
                for col in profile_columns
            ]
        })

        profile_df = profile_df.astype(str)

        st.dataframe(
            profile_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # COMPARISON INSIGHT

        st.markdown("### 💡 Model Insight")

        difference = abs(prob1 - prob2)

        if prob1 > prob2:
            higher_employee = emp1_id
        elif prob2 > prob1:
            higher_employee = emp2_id
        else:
            higher_employee = None

        if higher_employee is not None:

            st.warning(
                f"Employee {higher_employee} has a higher "
                f"predicted attrition risk by {difference:.2f} "
                "percentage points."
            )

        else:

            st.success(
                "Both employees have the same predicted "
                "attrition probability."
            )

        st.caption(
            "Note: These probabilities are model estimates, "
            "not certainty or proof that an employee will leave."
        )

elif page == "Employee Retention Simulator":

    st.subheader("🎯 Employee Retention Simulator")

    st.caption(
        "Explore how workplace improvements may change "
        "an employee's predicted attrition risk."
    )

    # Load dataset
    sim_df = pd.read_csv(
        "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

    sim_X = sim_df.drop(
        columns=[
            "Attrition",
            "EmployeeNumber",
            "EmployeeCount",
            "Over18",
            "StandardHours"
        ]
    )

    simulator_model = trained_models["Gradient Boosting"]

    # EMPLOYEE SELECTION

    st.markdown("### 👤 Select Employee")

    employee_index = st.selectbox(
        "Choose an employee to simulate",
        sim_df.index,
        format_func=lambda x: (
            f"Employee {sim_df.loc[x, 'EmployeeNumber']}"
        )
    )

    original_employee = sim_X.iloc[[employee_index]].copy()

    employee_age = sim_df.loc[employee_index, "Age"]
    employee_role = sim_df.loc[employee_index, "JobRole"]
    employee_department = sim_df.loc[employee_index, "Department"]

    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,#1E293B,#263B59);
        padding:20px;
        border-radius:14px;
        border:1px solid #334E73;
        margin-top:10px;
        margin-bottom:20px;
    ">
        <div style="color:#60A5FA;font-size:13px;font-weight:600;">
            SELECTED EMPLOYEE PROFILE
        </div>
        <div style="font-size:21px;font-weight:700;color:white;margin-top:8px;">
            Employee {sim_df.loc[employee_index, 'EmployeeNumber']}
        </div>
        <div style="color:#CBD5E1;margin-top:8px;">
            Age: {employee_age} &nbsp; | &nbsp;
            Department: {employee_department}
        </div>
        <div style="color:#CBD5E1;margin-top:5px;">
            Job Role: {employee_role}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # SIMULATION CONTROLS

    st.markdown("### 🛠️ Simulate Retention Improvements")

    st.caption(
        "Adjust the workplace factors below and observe "
        "how the model's prediction changes."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### ⏰ Workload")

        overtime = st.selectbox(
            "Overtime",
            ["Yes", "No"],
            index=["Yes", "No"].index(
                original_employee.iloc[0]["OverTime"]
            )
        )

    with col2:
        st.markdown("#### 😊 Job Satisfaction")

        satisfaction = st.slider(
            "Satisfaction Level",
            1, 4,
            int(original_employee.iloc[0]["JobSatisfaction"])
        )

    with col3:
        st.markdown("#### ⚖️ Work-Life Balance")

        work_balance = st.slider(
            "Balance Level",
            1, 4,
            int(original_employee.iloc[0]["WorkLifeBalance"])
        )

    # MODIFY EMPLOYEE PROFILE

    modified_employee = original_employee.copy()

    modified_employee.loc[
        modified_employee.index[0], "OverTime"
    ] = overtime

    modified_employee.loc[
        modified_employee.index[0], "JobSatisfaction"
    ] = satisfaction

    modified_employee.loc[
        modified_employee.index[0], "WorkLifeBalance"
    ] = work_balance

    # PREDICTIONS

    before_probability = (
        simulator_model.predict_proba(original_employee)[0][1] * 100
    )

    after_probability = (
        simulator_model.predict_proba(modified_employee)[0][1] * 100
    )

    risk_change = after_probability - before_probability

    st.divider()

    # BEFORE VS AFTER

    st.markdown("### 📊 Before vs After Risk Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div style="
            background:#1E293B;
            padding:20px;
            border-radius:13px;
            border:1px solid #475569;
            min-height:120px;
        ">
            <div style="color:#CBD5E1;font-size:13px;">
                ORIGINAL ATTRITION RISK
            </div>
            <div style="font-size:29px;font-weight:800;color:#F87171;margin-top:12px;">
                {before_probability:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        after_color = "#34D399" if risk_change < 0 else "#F87171"

        st.markdown(f"""
        <div style="
            background:#1E293B;
            padding:20px;
            border-radius:13px;
            border:1px solid #475569;
            min-height:120px;
        ">
            <div style="color:#CBD5E1;font-size:13px;">
                SIMULATED ATTRITION RISK
            </div>
            <div style="font-size:29px;font-weight:800;color:{after_color};margin-top:12px;">
                {after_probability:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        change_color = "#34D399" if difference > 0 else (
            "#F87171" if difference < 0 else "#CBD5E1"
        )

        st.markdown(f"""
        <div style="
            background:#1E293B;
            padding:20px;
            border-radius:13px;
            border:1px solid #475569;
            min-height:120px;
        ">
            <div style="color:#CBD5E1;font-size:13px;">
                ESTIMATED RISK REDUCTION
            </div>
            <div style="font-size:29px;font-weight:800;color:{change_color};margin-top:12px;">
                {difference:+.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # COMPARISON CHART

    st.markdown("### 📈 Risk Comparison Chart")

    fig, ax = plt.subplots(figsize=(9, 3.5))

    fig.patch.set_facecolor("#111C32")
    ax.set_facecolor("#111C32")

    labels = ["Original Risk", "Simulated Risk"]
    values = [before_probability, after_probability]

    bars = ax.barh(
        labels,
        values,
        color=["#F87171", "#34D399"],
        height=0.45
    )

    ax.invert_yaxis()
    ax.set_xlim(0, 110)

    ax.set_xlabel(
        "Attrition Probability (%)",
        color="#CBD5E1"
    )

    ax.tick_params(axis="x", colors="#CBD5E1")
    ax.tick_params(axis="y", colors="white")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")

    ax.grid(axis="x", linestyle="--", alpha=0.2)
    ax.set_axisbelow(True)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 2,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}%",
            va="center",
            color="white",
            fontweight="bold"
        )

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # FINAL RESULT

    st.divider()

    st.markdown("### 💡 Simulation Result")

    if risk_change < 0:

        st.success(
            f"Predicted attrition risk decreased by "
            f"{abs(risk_change):.2f} percentage points "
            "after the simulated changes."
        )

    elif risk_change > 0:

        st.warning(
            f"Predicted attrition risk increased by "
            f"{risk_change:.2f} percentage points "
            "after the simulated changes."
        )

    else:

        st.info(
            "The simulated changes did not alter "
            "the predicted attrition risk."
        )

    st.caption(
        "Important: Simulation results are model estimates. "
        "They do not guarantee employee retention or establish "
        "a causal relationship."
    )
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AttritionIQ | Employee Attrition Prediction "
    "& Retention Analytics | Academic ML Project"
)