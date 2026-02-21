import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Load Model & Encoder
# ----------------------------
model = joblib.load("disease_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ----------------------------
# Load Datasets
# ----------------------------
symptom_description = pd.read_csv(
    "data/symptom_Description.csv",
    header=None,
    names=["Disease", "Description"]
)

symptom_precaution = pd.read_csv(
    "data/symptom_precaution.csv",
    header=None,
    names=["Disease", "Precaution1", "Precaution2", "Precaution3", "Precaution4"]
)

severity_df = pd.read_csv(
    "data/Symptom_severity.csv",
    header=None,
    names=["Symptom", "Weight"]
)

training_data = pd.read_csv("data/Training.csv")
symptom_list = list(training_data.columns[:-1])

# Create symptom index
symptom_dict = {symptom: i for i, symptom in enumerate(symptom_list)}

# ----------------------------
# Prediction Function
# ----------------------------
def predict_disease(symptoms):
    input_data = np.zeros(len(symptom_list))
    for symptom in symptoms:
        if symptom in symptom_dict:
            input_data[symptom_dict[symptom]] = 1

    prediction = model.predict([input_data])
    disease = label_encoder.inverse_transform(prediction)[0]
    return disease

def get_description(disease):
    row = symptom_description[symptom_description["Disease"] == disease]
    if not row.empty:
        return row["Description"].values[0]
    return "Description not available."

def get_precautions(disease):
    row = symptom_precaution[symptom_precaution["Disease"] == disease]
    if not row.empty:
        return row.iloc[0, 1:].dropna().tolist()
    return ["No precautions available."]

# ----------------------------
# Pages
# ----------------------------
def chatbot_page():
    st.title("🩺 Healthcare Chatbot")
    st.write("Select your symptoms and get a predicted disease with description and precautions.")

    selected_symptoms = st.multiselect(
        "Select Symptoms:",
        symptom_list
    )

    if st.button("🔍 Predict Disease"):
        if selected_symptoms:
            predicted_disease = predict_disease(selected_symptoms)

            st.success(f"🦠 Predicted Disease: {predicted_disease}")

            description = get_description(predicted_disease)
            precautions = get_precautions(predicted_disease)

            st.markdown("### 📝 Description")
            st.write(description)

            st.markdown("### ✅ Precautions")
            for i, p in enumerate(precautions, 1):
                st.write(f"{i}. {p}")

            st.info("🩹 Advice: Please consult a medical professional for proper diagnosis.")
        else:
            st.warning("⚠️ Please select at least one symptom.")


def health_tips_page():
    st.title("💡 General Health Tips")

    tips = [
        "🥗 Eat a balanced diet with fruits and vegetables.",
        "💧 Drink plenty of water daily.",
        "🏃 Exercise regularly.",
        "😴 Get enough sleep.",
        "🚭 Avoid smoking and alcohol.",
        "🩺 Go for regular health checkups."
    ]

    for tip in tips:
        st.write(f"- {tip}")


def about_page():
    st.title("ℹ️ About the Healthcare Chatbot")

    st.write("""
    This Healthcare Chatbot predicts possible diseases based on user-selected symptoms
    using a Machine Learning model trained on symptom-disease data.
    
    It also provides:
    - Disease description
    - Precautionary measures
    - General health tips
    
    This application is built using **Python, Machine Learning, and Streamlit**.
    """)

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("📚 Navigation")
menu = st.sidebar.radio("Go to:", ["Healthcare Chatbot", "General Health Tips", "About"])

if menu == "Healthcare Chatbot":
    chatbot_page()
elif menu == "General Health Tips":
    health_tips_page()
elif menu == "About":
    about_page()
