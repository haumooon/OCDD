import streamlit as st
import joblib, json, pandas as pd
from edf_predict import extract_features_for_model

# Load model + features
MODEL_PATH = "ocd_anxiety_model_hybrid.pkl"
FEATURES_PATH = "feature_cols_hybrid.json"
model = joblib.load(MODEL_PATH)
feature_cols = json.load(open(FEATURES_PATH))

st.set_page_config(page_title="QEEG OCD vs Anxiety Classifier", layout="centered")

st.title("🧠 QEEG OCD vs Anxiety Classifier")
st.write("Upload an EDF EEG file to classify between OCD and Anxiety using hybrid spectral + coherence markers.")

uploaded_file = st.file_uploader("Choose an EDF file", type=["edf"])

if uploaded_file is not None:
    # Save the uploaded EDF temporarily
    with open("temp.edf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Extract features
        values = extract_features_for_model("temp.edf", feature_cols)
        feats_df = pd.DataFrame([values], columns=feature_cols)

        # Prediction
        pred = model.predict(feats_df)[0]
        prob = model.predict_proba(feats_df)[0, 1]

        label = "🧠 OCD" if pred == 1 else "⚡ Anxiety"
        st.success(f"**Prediction:** {label}")
        st.write(f"**Probability OCD:** {prob:.2f}")

        # Show feature preview
        with st.expander("Show Extracted Features"):
            st.dataframe(feats_df.T)

    except Exception as e:
        st.error(f"Error processing file: {e}")
