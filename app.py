import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Wine Quality Classification", layout="wide")

@st.cache_resource
def load_models():
    """Load all trained models"""
    model_files = {
        "Logistic Regression": "LogisticRegression.pkl",
        "Decision Tree": "DecisionTree.pkl",
        "KNN": "KNN.pkl",
        "Naive Bayes": "NaiveBayes.pkl",
        "Random Forest": "RandomForest.pkl",
        "XGBoost": "XGBoost.pkl"
    }
    models = {}
    for name, filename in model_files.items():
        try:
            models[name] = joblib.load(filename)
            st.sidebar.success(f"✅ {name} loaded")
        except Exception as e:
            st.sidebar.warning(f"⚠️ {name}: {filename} not found")
    return models

def main():
    st.title("🍷 Wine Quality Classification")
    st.markdown("**Interactive demo for ML Assignment 2**")
    
    # Load models
    models = load_models()
    
    # Sidebar controls
    st.sidebar.header("Model Selection")
    model_name = st.sidebar.selectbox("Choose model:", list(models.keys()) if models else ["Demo Mode"])
    
    # File uploader
    uploaded_file = st.file_uploader("📁 Upload test CSV (12 features)", type=["csv"])
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(data)} rows")
        st.dataframe(data.head())
        
        if model_name in models and model_name != "Demo Mode":
            model = models[model_name]
            try:
                # Predict
                X = data.select_dtypes(include=[np.number])
                predictions = model.predict(X)
                
                st.subheader(f"🔮 Predictions ({model_name})")
                st.write(predictions[:10])
                
                # Metrics if quality column exists
                if 'quality' in data.columns:
                    y_true = data['quality'].values[:len(predictions)]
                    acc = accuracy_score(y_true, predictions[:len(y_true)])
                    st.metric("Accuracy", f"{acc:.3f}")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            # Demo mode
            np.random.seed(42)
            predictions = np.random.choice([0,1,2,3,4,5], len(data))
            st.subheader("🔮 Demo Predictions")
            st.write(predictions[:10])
            
            st.balloons()
    
    st.markdown("---")
    st.markdown("*Models trained using UCI Wine Quality dataset (12 features, 2000 samples)*")

if __name__ == "__main__":
    main()
