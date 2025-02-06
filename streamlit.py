import streamlit as st

# Dummy Prediction Function
def predict_risk(age, cholesterol):
    return "High Risk" if age > 50 and cholesterol > 200 else "Low Risk"

# Streamlit UI
st.title("Heart Attack Prediction")

# Input Fields
age = st.number_input("Enter Age:", min_value=0, max_value=120, step=1)
cholesterol = st.number_input("Enter Cholesterol Level:", min_value=50, max_value=500, step=1)

# Prediction Button
if st.button("Predict"):
    prediction = predict_risk(age, cholesterol)
    st.write(f"### Prediction: {prediction}")
