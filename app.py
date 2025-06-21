import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="🎥 Indian Movie Theatre Price Predictor", layout="wide")

st.title("🎥 Indian Movie Theatre Price Predictor")
st.write("Predict whether the theatre pricing is **Low**, **Medium** or **High** based on features.")

# Load dataset
try:
    df = pd.read_csv("indian-movie-theatres.csv")
except FileNotFoundError:
    st.error("❌ 'indian-movie-theatres.csv' not found. Please ensure it's in the app folder.")
    st.stop()

# Basic cleaning
df.dropna(subset=["average_ticket_price"], inplace=True)

# Define price category
def classify_price(price):
    if price < 100:
        return "Low"
    elif 100 <= price < 150:
        return "Medium"
    else:
        return "High"

df["Price_Category"] = df["average_ticket_price"].apply(classify_price)

# Drop irrelevant columns if they exist
drop_cols = ['theatre_name', 'notes', 'source_of_information']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# Handle missing numerical columns
df["total_seats"] = df["total_seats"].fillna(df["total_seats"].median())
df["no_screens"] = df["no_screens"].fillna(df["no_screens"].median())

# Store all category levels for consistency later
all_types = df['type'].fillna('Unknown').unique().tolist() if 'type' in df.columns else []
all_cities = df['city'].fillna('Unknown').unique().tolist() if 'city' in df.columns else []

# Encode categorical columns
if 'type' in df.columns:
    df['type'] = df['type'].fillna('Unknown')
    df = pd.get_dummies(df, columns=['type'])
if 'city' in df.columns:
    df['city'] = df['city'].fillna('Unknown')
    df = pd.get_dummies(df, columns=['city'])

# Feature selection
selected_features = ['total_seats', 'no_screens'] + [col for col in df.columns if col.startswith('type_') or col.startswith('city_')]
X = df[selected_features]
y = df["Price_Category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(multi_class='ovr', solver='liblinear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["Data Preview", "Visualizations", "Model", "Predict"])

# Page: Data Preview
if page == "Data Preview":
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())
    st.markdown(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
    st.write(df["Price_Category"].value_counts())

# Page: Visualizations
elif page == "Visualizations":
    st.subheader("📊 Visualizations")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["average_ticket_price"], bins=20, kde=True, ax=ax1)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.countplot(x="Price_Category", data=df, ax=ax2)
    st.pyplot(fig2)

# Page: Model
elif page == "Model":
    st.subheader("🤖 Model Evaluation")
    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{acc * 100:.2f}%")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.subheader("📉 Confusion Matrix")
    fig3, ax3 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", ax=ax3)
    st.pyplot(fig3)

# Page: Predict
elif page == "Predict":
    st.subheader("🎯 Make a Prediction")

    user_input = {
        'total_seats': int(st.number_input("Total Seats", value=100)),
        'no_screens': int(st.number_input("Number of Screens", value=1))
    }

    selected_type = st.selectbox("Select Theatre Type", all_types)
    selected_city = st.selectbox("Select City", all_cities)

    # One-hot encode the dropdown selections
    for val in all_types:
        col_name = f'type_{val}'
        user_input[col_name] = 1 if val == selected_type else 0

    for val in all_cities:
        col_name = f'city_{val}'
        user_input[col_name] = 1 if val == selected_city else 0

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        # Ensure all features match training features
        for col in selected_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[selected_features]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        confidence = model.predict_proba(input_scaled).max()

        st.success(f"🏷 Predicted Price Category: **{prediction}**")
        st.info(f"🔍 Confidence: {confidence * 100:.2f}%")
