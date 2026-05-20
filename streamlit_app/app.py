import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Loan Default Risk Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>

    .main {
        background-color: #f5f7fa;
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-size: 18px;
        font-weight: bold;
        background-color: #2563eb;
        color: white;
    }

    .prediction-card {
        padding: 25px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 20px;
    }

    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# API CONFIG
# =========================================================
API_URL = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/"

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.title("⚙️ Dashboard")

    st.markdown("---")

    st.markdown("""
    ### 📌 About Project

    This application predicts the probability of a customer defaulting on a loan using:

    - Logistic Regression
    - MLflow Model Registry
    - FastAPI Backend
    - DVC Pipeline
    - Streamlit Frontend
    - Dockerized Architecture
    """)

    st.markdown("---")

    # Health Check
    st.subheader("🔍 API Health Check")

    try:
        response = requests.get(HEALTH_URL, timeout=5)

        if response.status_code == 200:
            st.success("FastAPI Server Running")
        else:
            st.error("FastAPI Server Error")

    except Exception:
        st.error("FastAPI Server Not Reachable")

    st.markdown("---")

    st.subheader("👨‍💻 Tech Stack")

    tech_df = pd.DataFrame({
        "Technology": [
            "FastAPI",
            "Streamlit",
            "MLflow",
            "DVC",
            "Docker",
            "Scikit-Learn"
        ]
    })

    st.dataframe(tech_df, use_container_width=True)

# =========================================================
# HEADER
# =========================================================

st.title("🏦 Loan Default Risk Prediction System")

st.markdown(
    """
    Predict whether a customer is likely to default on a loan using a production-style MLOps pipeline.
    """
)

st.markdown("---")

# =========================================================
# INPUT FORM
# =========================================================

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)

        income = st.number_input("Income", min_value=0, value=60000)

        loan_amount = st.number_input("Loan Amount", min_value=0, value=20000)

        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)

        months_employed = st.number_input("Months Employed", min_value=0, value=24)

    with col2:

        num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=3)

        interest_rate = st.number_input("Interest Rate", min_value=0.0, value=12.5)

        loan_term = st.number_input("Loan Term (Months)", min_value=1, value=36)

        dti_ratio = st.slider("DTI Ratio", 0.0, 1.0, 0.3)

        education = st.selectbox(
            "Education",
            ["High School", "Bachelor's", "Master's", "PhD"]
        )

    with col3:

        employment_type = st.selectbox(
            "Employment Type",
            ["Full-time", "Part-time", "Self-employed", "Unemployed"]
        )

        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Divorced"]
        )

        has_mortgage = st.selectbox(
            "Has Mortgage",
            ["Yes", "No"]
        )

        has_dependents = st.selectbox(
            "Has Dependents",
            ["Yes", "No"]
        )

        loan_purpose = st.selectbox(
            "Loan Purpose",
            [
                "Business",
                "Education",
                "Personal",
                "Auto",
                "Home"
            ]
        )

        has_cosigner = st.selectbox(
            "Has Co-Signer",
            ["Yes", "No"]
        )

    submit_button = st.form_submit_button("🚀 Predict Loan Default Risk")


# =========================================================
# PREDICTION LOGIC
# =========================================================

if submit_button:

    payload = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner
    }

    with st.spinner("Analyzing customer risk profile..."):

        try:
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:

                result = response.json()

                prediction = result["prediction"]
                probability = result["default_probability"]

                st.markdown("---")
                st.subheader("📊 Prediction Results")

                # =========================================================
                # METRICS
                # =========================================================

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="Prediction",
                        value="Default" if prediction == 1 else "No Default"
                    )

                with col2:
                    st.metric(
                        label="Default Probability",
                        value=f"{probability * 100:.2f}%"
                    )

                with col3:
                    risk_level = (
                        "High Risk"
                        if probability > 0.7
                        else "Medium Risk"
                        if probability > 0.4
                        else "Low Risk"
                    )

                    st.metric(
                        label="Risk Level",
                        value=risk_level
                    )

                # =========================================================
                # GAUGE CHART
                # =========================================================

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        title={"text": "Loan Default Probability"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"thickness": 0.3},
                            "steps": [
                                {"range": [0, 40], "color": "lightgreen"},
                                {"range": [40, 70], "color": "yellow"},
                                {"range": [70, 100], "color": "red"}
                            ],
                        },
                    )
                )

                fig.update_layout(height=400)

                st.plotly_chart(fig, use_container_width=True)

                # =========================================================
                # RESULT CARD
                # =========================================================

                if prediction == 1:
                    st.error(
                        "⚠️ Customer has HIGH probability of loan default."
                    )
                else:
                    st.success(
                        "✅ Customer is likely to repay the loan successfully."
                    )

                # =========================================================
                # EXPANDABLE SECTION
                # =========================================================

                with st.expander("🔎 View Submitted Data"):
                    st.json(payload)

                # =========================================================
                # SESSION HISTORY
                # =========================================================

                if "history" not in st.session_state:
                    st.session_state.history = []

                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "prediction": prediction,
                    "probability": probability
                })

                history_df = pd.DataFrame(st.session_state.history)

                st.subheader("📜 Prediction History")
                st.dataframe(history_df, use_container_width=True)

            else:
                st.error(f"API Error: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI backend.")

        except Exception as e:
            st.error(f"Unexpected Error: {e}")    
