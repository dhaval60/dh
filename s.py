import streamlit as st
import pickle
import numpy as np

# ----------------------------------------
# Helper functions
# ----------------------------------------

def load_model(model_filename):
    """Load the ANN model from a .pkl file"""
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_force(model, input_yarn):
    """Predict Force using the loaded model and yarn input"""
    input_array = np.array([[input_yarn]])
    prediction = model.predict(input_array)
    return prediction[0]

# ----------------------------------------
# Streamlit App
# ----------------------------------------

st.set_page_config(page_title="Yarn to Force Predictor", page_icon="🧵")

st.title("🧵 Yarn to Force Predictor App")
st.write("Enter the Yarn value to predict the Force using the ANN model.")

# 1. Input Yarn
st.subheader("🔢 Input Yarn Parameter")
yarn = st.number_input("Yarn Value", min_value=0.0, format="%.4f")

# 2. Predict Button
if st.button("Predict Force"):
    try:
        model = load_model("b_force.pkl")
        force = predict_force(model, yarn)
        st.success(f"🔮 Predicted Force: {force:.4f} N")
    except FileNotFoundError:
        st.error("❗ Model file 'b_force.pkl' not found in the folder.")
    except Exception as e:
        st.error(f"⚠ Error: {str(e)}")

st.caption("Made with ❤ using Streamlit")
