import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- UI ----------------
st.set_page_config(page_title="FinSight AI", layout="wide")
st.title("💰 FinSight AI - Financial Anomaly Detection System")

# ---------------- LOAD DATA ----------------
uploaded_file = st.file_uploader("Upload Financial CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded → using sample dataset")
    df = pd.read_csv("Personal_Finance_Dataset.csv")

# ---------------- CLEAN ----------------
df = df.dropna()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
df = df.dropna()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# ---------------- FILTER EXPENSES ----------------
expense_df = df[df["Type"].str.lower() == "expense"].copy()

# ---------------- CATEGORY FREQUENCY (IMPORTANT IMPROVEMENT) ----------------
category_counts = expense_df["Category"].value_counts().to_dict()

expense_df["category_freq"] = expense_df["Category"].map(category_counts)
expense_df["category_rarity"] = 1 / (expense_df["category_freq"] + 1)

# ---------------- DATE FEATURES ----------------
expense_df["day"] = expense_df["Date"].dt.day
expense_df["month"] = expense_df["Date"].dt.month
expense_df["weekday"] = expense_df["Date"].dt.weekday

# ---------------- NLP FEATURES ----------------
vectorizer = TfidfVectorizer(max_features=30)

text_matrix = vectorizer.fit_transform(
    expense_df["Transaction Description"].astype(str)
).toarray()

text_df = pd.DataFrame(text_matrix)

# ---------------- FEATURE COMBINATION ----------------
X = pd.concat([
    expense_df[[
        "Amount",
        "day",
        "month",
        "weekday",
        "category_rarity"
    ]].reset_index(drop=True),
    text_df.reset_index(drop=True)
], axis=1)

# FIX sklearn requirement
X.columns = X.columns.astype(str)

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- MODEL ----------------
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

pred = model.fit_predict(X_scaled)

scores = model.decision_function(X_scaled)

# ---------------- ANOMALY FLAG ----------------
expense_df["anomaly"] = np.where(pred == -1, 1, 0)

# ---------------- RISK SCORE (0–100 normalized) ----------------
expense_df["risk_score"] = (
    (scores.max() - scores) /
    (scores.max() - scores.min() + 1e-9)
) * 100

# ---------------- DASHBOARD ----------------
st.subheader("💰 Financial Summary")

income = df[df["Type"].str.lower() == "income"]["Amount"].sum()
expense = df[df["Type"].str.lower() == "expense"]["Amount"].sum()
balance = income - expense

col1, col2, col3 = st.columns(3)

col1.metric("Total Income", f"₹{income:,.2f}")
col2.metric("Total Expense", f"₹{expense:,.2f}")
col3.metric("Net Balance", f"₹{balance:,.2f}")

# ---------------- ANOMALIES ----------------
st.subheader("🚨 Unusual Expense Transactions")

anomalies = expense_df[expense_df["anomaly"] == 1]

st.dataframe(anomalies[[
    "Date",
    "Transaction Description",
    "Category",
    "Amount",
    "risk_score"
]].sort_values("risk_score", ascending=False))

# ---------------- INSIGHT ----------------
st.subheader("🧠 AI Insight")

if len(anomalies) > 0:
    top = anomalies.sort_values("risk_score", ascending=False).iloc[0]

    st.error(f"""
Detected {len(anomalies)} unusual expense transactions.

🔴 Highest Risk Transaction:
- Amount: ₹{top['Amount']}
- Category: {top['Category']}
- Risk Score: {top['risk_score']:.2f}

Interpretation:
This transaction is statistically unusual compared to normal spending patterns.
It indicates deviation from behavior, NOT fraud.

Note:
Low-value transactions can also be flagged if they are rare or behaviorally different.
""")

else:
    st.success("No unusual spending detected.")

# ---------------- CATEGORY ANALYSIS ----------------
st.subheader("📊 Category-wise Spending")

cat = expense_df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
st.bar_chart(cat)

# ---------------- FULL DATA ----------------
st.subheader("📋 Full Expense Data")

st.dataframe(expense_df)