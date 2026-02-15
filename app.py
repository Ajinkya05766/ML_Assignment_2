import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

st.set_page_config(page_title="Wine Quality Classification", layout="wide")

@st.cache_resource
def load_models():
    model_names = ["Logistic_Regression", "Decision_Tree", "KNN", "Naive_Bayes", "Random_Forest", "XGBoost"]
    models = {}
    for name in model_names:
        try:
            models[name.replace("_", " ")] = joblib.load(f"{name}.pkl")
        except:
            pass
    return models

def main():
    st.title("🍷 Wine Quality Classification")
    st.write("Predict wine quality using ML models. Built for Assignment 2.")

    models = load_models()
    model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file and model_name:
        data = pd.read_csv(uploaded_file)
        
        # Simple preprocessing to match training (ensure 12 cols if possible)
        # Drop ID or target if present for prediction
        if 'quality' in data.columns:
            X = data.drop('quality', axis=1)
            y_true = data['quality']
        else:
            X = data
            y_true = None
            
        # Ensure only numeric columns are used if model expects scaler
        X = X.select_dtypes(include=[np.number])
        
        model = models[model_name]
        try:
            pred = model.predict(X)
            
            st.subheader("Predictions")
            st.write(pred[:10])
            
            if y_true is not None:
                st.subheader("Metrics")
                acc = accuracy_score(y_true, pred)
                st.metric("Accuracy", f"{acc:.4f}")
        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
