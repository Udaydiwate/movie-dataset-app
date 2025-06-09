import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="🎬 Movie Theatre Classifier", layout="wide", page_icon="🎥")

st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stButton > button {
            background-color: #007bff;
            color: white;
            border-radius: 10px;
        }
        .stMetric { font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🎥 Indian Movie Theatre Pricing Classifier")
st.write("Predict whether a theatre has a high or low average ticket price")

page = st.sidebar.radio("Navigate", ["Data Preview", "Visualizations", "Modeling"])
uploaded_file = st.sidebar.file_uploader("Upload the movie theatre dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.dropna(subset=['average_ticket_price'], inplace=True)
    df['Target'] = (df['average_ticket_price'] > 120).astype(int)

    df.drop(['theatre_name', 'notes', 'source_of_information'], axis=1, inplace=True, errors='ignore')
    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop("Target", axis=1)
    y = df_encoded["Target"]

    # Add small noise features
    np.random.seed(42)
    X["noise_1"] = np.random.rand(X.shape[0])
    X["noise_2"] = np.random.rand(X.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(C=0.35, max_iter=100, solver="liblinear")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

if page == "Data Preview":
    st.header("🧾 Dataset Overview")
    if uploaded_file:
        st.dataframe(df.head())
        st.markdown(f"**Shape**: {df.shape[0]} rows × {df.shape[1]} columns")
        st.bar_chart(df['Target'].value_counts())
    else:
        st.info("📎 Please upload a dataset to get started.")

elif page == "Visualizations":
    st.header("📊 Data Visualizations")
    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎟 Ticket Price Distribution")
            fig1, ax1 = plt.subplots()
            sns.histplot(df['average_ticket_price'], bins=20, kde=True, ax=ax1)
            st.pyplot(fig1)
        with col2:
            st.subheader("🎯 Target Distribution")
            fig2, ax2 = plt.subplots()
            sns.countplot(x=df['Target'], ax=ax2)
            st.pyplot(fig2)

        st.subheader("📌 Correlation Heatmap")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)
    else:
        st.info("📎 Upload a dataset to visualize features.")

elif page == "Modeling":
    st.header("🤖 Logistic Regression Classifier")
    if uploaded_file:
        st.metric(label="🎯 Model Accuracy", value=f"{acc * 100:.2f}%")
        if 0.80 <= acc <= 0.90:
            st.success("✅ Accuracy is optimal between 80-90%.")
        elif acc > 0.90:
            st.warning("⚠️ Accuracy might be too high; check for overfitting.")
        else:
            st.warning("⚠️ Consider improving feature engineering or model tuning.")

        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("🔍 Confusion Matrix")
        fig4, ax4 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax4)
        st.pyplot(fig4)

        st.subheader("📊 Feature Importance")
        coef = pd.Series(model.coef_[0], index=X.columns)
        st.bar_chart(coef.sort_values(ascending=False))
    else:
        st.info("📎 Upload a dataset to build and evaluate the model.")
