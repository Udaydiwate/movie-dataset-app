import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Theatre Ticket Classifier", layout="wide")

# Sidebar
page = st.sidebar.selectbox("Select Page", ["Data Preview", "Visualizations", "Modeling"])
uploaded_file = st.sidebar.file_uploader("Upload your theatre dataset CSV", type=["csv"])

# Shared logic
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.dropna()
    df["Target"] = (df["average_ticket_price"] > 120).astype(int)

    df = df.drop(["theatre_name", "notes", "source_of_information"], axis=1, errors='ignore')

    # Encode categoricals
    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop("Target", axis=1)
    y = df_encoded["Target"]

    # Drop strong predictors
    for col in ["average_ticket_price", "calculated_ticket_prices"]:
        if col in X.columns:
            X = X.drop(col, axis=1)

    # Add controlled noise
    for i in range(2):
        np.random.seed(i)
        X[f"noise_{i}"] = np.random.rand(X.shape[0])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# Page 1 - Data Preview
if page == "Data Preview":
    st.title("🎟️ Theatre Dataset Preview")
    if uploaded_file is not None:
        st.dataframe(df.head())
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.bar_chart(df["Target"].value_counts())
    else:
        st.info("Upload a CSV to begin.")

# Page 2 - Visualizations
elif page == "Visualizations":
    st.title("📈 Visualizations")
    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            if "total_seats" in df.columns:
                st.write("Total Seats Distribution")
                fig1, ax1 = plt.subplots()
                sns.histplot(df["total_seats"], kde=True, ax=ax1)
                st.pyplot(fig1)

        with col2:
            if "no_screens" in df.columns:
                st.write("No. of Screens vs Target")
                fig2, ax2 = plt.subplots()
                sns.boxplot(x="Target", y="no_screens", data=df, ax=ax2)
                st.pyplot(fig2)

        st.subheader("Correlation Heatmap")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)
    else:
        st.info("Upload a file to see plots.")

# Page 3 - Modeling
elif page == "Modeling":
    st.title("🤖 Logistic Regression Model")
    if uploaded_file is not None:
        model = LogisticRegression(max_iter=100, C=0.05, solver="liblinear")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.metric(label="🎯 Model Accuracy", value=f"{acc:.2f}")

        if acc > 0.90:
            st.warning("⚠️ Accuracy above 90%. Consider dropping more strong features.")
        elif acc < 0.80:
            st.warning("⚠️ Accuracy below 80%. Add more predictive features.")
        else:
            st.success("✅ Accuracy is between 80% and 90% — good for generalization.")

        # Classification Report
        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        fig4, ax4 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax4)
        st.pyplot(fig4)

        # Feature Importance
        st.subheader("📊 Feature Importance")
        coef = pd.Series(model.coef_[0], index=X.columns)
        st.bar_chart(coef.sort_values(ascending=False))
    else:
        st.info("Upload dataset to train the model.")
