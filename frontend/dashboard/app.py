import streamlit as st
import requests
import pandas as pd

AGENT_URL = "http://localhost:8001"
MODEL_URL = "http://localhost:8000"

st.set_page_config(page_title="Drift Triage Co‑Pilot", layout="wide")
st.title("🩺 Drift Triage Co‑Pilot – Dashboard")

# ------------------------------------------------------------
# Helper to reset drift after human decision
# ------------------------------------------------------------
def reset_drift():
    try:
        requests.post(f"{MODEL_URL}/api/v1/reset-drift", timeout=3)
    except:
        pass

# ------------------------------------------------------------
# Sidebar – Connection status
# ------------------------------------------------------------
st.sidebar.header("⚙️ Connection Status")

try:
    r = requests.get(f"{AGENT_URL}/health", timeout=3)
    agent_ok = r.status_code == 200
except:
    agent_ok = False
st.sidebar.write("🔹 Agent:", "✅" if agent_ok else "❌")

try:
    r = requests.get(f"{MODEL_URL}/api/v1/health", timeout=3)
    model_ok = r.status_code == 200
except:
    model_ok = False
st.sidebar.write("🔹 Model Service:", "✅" if model_ok else "❌")

# ------------------------------------------------------------
# Model & Drift Status
# ------------------------------------------------------------
st.header("📊 Model & Drift Status")

if model_ok:
    try:
        metrics = requests.get(f"{MODEL_URL}/api/v1/metrics", timeout=5).json()
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Version", metrics.get("model", {}).get("production_version", "?"))
        col2.metric("Threshold", f"{metrics.get('model', {}).get('threshold', 0):.3f}")
        drift = metrics.get("drift", {})
        if drift:
            col3.metric("Drift Severity", drift.get("severity", "none"))
            st.write("**Drift details** – PSI:", drift.get("psi_numeric", "?"),
                     "| Chi²:", drift.get("chi2_categorical", "?"),
                     "| Output drift:", drift.get("output_drift", "?"))
        else:
            col3.metric("Drift Severity", "not computed")
        st.write("📈 Predictions stored:", metrics.get("predictions", {}).get("total_stored", "?"))
    except:
        st.warning("Could not fetch model metrics.")
else:
    st.warning("Model service is down.")

# ------------------------------------------------------------
# Prediction Input
# ------------------------------------------------------------
st.header("🔮 Send a Prediction")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=42)
        job = st.selectbox("Job", ["technician", "admin.", "services", "management", "blue-collar",
                                   "entrepreneur", "housemaid", "retired", "student", "unemployed", "unknown"])
        marital = st.selectbox("Marital", ["married", "single", "divorced", "unknown"])
        education = st.selectbox("Education", ["university.degree", "high.school", "basic.9y", "professional.course", "unknown"])
        default = st.selectbox("Default", ["no", "yes", "unknown"])
        housing = st.selectbox("Housing", ["yes", "no", "unknown"])

    with col2:
        loan = st.selectbox("Loan", ["yes", "no", "unknown"])
        contact = st.selectbox("Contact", ["cellular", "telephone"])
        month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
        day_of_week = st.selectbox("Day of Week", ["mon", "tue", "wed", "thu", "fri"])
        campaign = st.number_input("Campaign", min_value=0, max_value=50, value=1)
        pdays = st.number_input("Pdays", min_value=-1, max_value=999, value=999)
        previous = st.number_input("Previous", min_value=0, max_value=10, value=0)

    with col3:
        poutcome = st.selectbox("Poutcome", ["nonexistent", "success", "failure"])
        emp_var_rate = st.number_input("emp.var.rate", value=-1.8, format="%.3f")
        cons_price_idx = st.number_input("cons.price.idx", value=93.075, format="%.3f")
        cons_conf_idx = st.number_input("cons.conf.idx", value=-36.1, format="%.3f")
        euribor3m = st.number_input("euribor3m", value=4.857, format="%.3f")
        nr_employed = st.number_input("nr.employed", value=5191.0, format="%.3f")

    submitted = st.form_submit_button("🔮 Predict")

if submitted:
    payload = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp_var_rate": emp_var_rate,
        "cons_price_idx": cons_price_idx,
        "cons_conf_idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr_employed": nr_employed
    }
    with st.spinner("Predicting..."):
        try:
            resp = requests.post(f"{MODEL_URL}/api/v1/predict", json=payload, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                pred_label = "Subscribe" if result["prediction"] == 1 else "Won't subscribe"
                st.success(f"Prediction: {pred_label} (confidence: {result['probability']:.3f})")
            else:
                st.error(f"Prediction failed: {resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

# ------------------------------------------------------------
# Pending Investigations (HIL Inbox)
# ------------------------------------------------------------
st.header("📬 Pending Investigations (HIL Inbox)")

if agent_ok:
    try:
        pending = requests.get(f"{AGENT_URL}/webhook/pending", timeout=5).json()
    except:
        st.error("Failed to fetch pending investigations.")
        pending = []

    if not pending:
        st.info("No pending investigations.")
    else:
        for inv in pending:
            with st.expander(f"{inv['id']} – {inv.get('proposed_action', '?')} (model {inv.get('model_version','?')})"):
                st.write("**Triage result:**", inv.get("triage_result", ""))
                st.write("**Action proposed:**", inv.get("proposed_action", ""))
                col1, col2 = st.columns(2)
                if col1.button("✅ Approve", key=f"approve_{inv['id']}"):
                    try:
                        resp = requests.post(
                            f"{AGENT_URL}/webhook/investigations/{inv['id']}/approve",
                            json={"approve": True}, timeout=10
                        )
                        if resp.status_code == 200:
                            st.success("Approved!")
                            reset_drift()          # ← allow new drift alerts
                            st.rerun()
                        else:
                            st.error(f"Approval failed: {resp.text}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")
                if col2.button("❌ Reject", key=f"reject_{inv['id']}"):
                    try:
                        resp = requests.post(
                            f"{AGENT_URL}/webhook/investigations/{inv['id']}/reject",
                            json={"approve": False}, timeout=10
                        )
                        if resp.status_code == 200:
                            st.success("Rejected.")
                            reset_drift()          # ← allow new drift alerts
                            st.rerun()
                        else:
                            st.error(f"Rejection failed: {resp.text}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")
else:
    st.warning("Agent is down.")

# ------------------------------------------------------------
# Queue Stats
# ------------------------------------------------------------
st.header("⚡ Job Queue Stats")

if agent_ok:
    try:
        stats = requests.get(f"{AGENT_URL}/webhook/queue/stats", timeout=5).json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Queue Depth", stats["queue_depth"])
        c2.metric("DLQ Depth", stats["dead_letter_depth"])
        c3.metric("Processed Jobs", stats["processed_count"])
    except:
        st.warning("Could not fetch queue stats.")
else:
    st.warning("Agent is down.")

# ------------------------------------------------------------
# All Investigations
# ------------------------------------------------------------
st.header("📋 All Investigations")
if agent_ok:
    try:
        all_inv = requests.get(f"{AGENT_URL}/webhook/investigations", timeout=5).json()
        if all_inv:
            df = pd.DataFrame(all_inv)
            df = df[["id", "model_version", "status", "proposed_action", "created_at"]]
            st.dataframe(df.tail(20), use_container_width=True)
        else:
            st.info("No investigations yet.")
    except:
        st.warning("Could not fetch investigation list.")
else:
    st.warning("Agent is down.")