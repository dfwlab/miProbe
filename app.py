import streamlit as st
import numpy as np
import joblib
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from miprobe import get_numpy_dataset


# Streamlit App Config
st.set_page_config(page_title="miProbe", layout="wide")
st.title("🔬 miProbe Toolkit Demo")


# ----------------------------
# 1. Retrieve embeddings via API
# ----------------------------
st.markdown("## 1. Retrieve peptide embedding via API")

st.code(
    '''from miprobe import get_numpy_dataset
peptide_ids = ['ORF050.00000001', 'ORF050.00000002', 'ORF050.00000003']
X = get_numpy_dataset(peptide_ids, embedding_model='prottrans')''',
    language="python"
)

if st.button("Fetch embeddings"):
    peptide_ids = ['ORF050.00000001', 'ORF050.00000002', 'ORF050.00000003']
    embeddings = get_numpy_dataset(peptide_ids)
    st.write(embeddings)

st.divider()


# ----------------------------
# 2. Train a model using peptide embeddings
# ----------------------------
st.markdown("## 2. Train scikit-learn classifier using peptide embeddings")

peptide_ids_input = st.text_area("🔢 Input peptide IDs (one per line):", "ORF050.00000001\nORF050.00000002\nORF050.00000003\nORF050.00000004\nORF050.00000005")
peptide_ids = [pid.strip() for pid in peptide_ids_input.strip().splitlines() if pid.strip()]

labels_input = st.text_input("🧬 Input binary labels (comma-separated):", "1,0,1,1,0")
try:
    labels = [int(x) for x in labels_input.strip().split(',')]
except ValueError:
    st.error("Label format error. Please use comma-separated integers like 1,0,1")
    st.stop()

if len(labels) != len(peptide_ids):
    st.error("Number of labels does not match number of peptide IDs.")
    st.stop()

MODEL_PATH = "peptide_model.pkl"
if st.button("🚀 Train Model"):
    with st.spinner("Downloading embeddings and training model..."):
        X = get_numpy_dataset(peptide_ids)
        y = np.array(labels)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        preds = clf.predict(X)
        report = classification_report(y, preds, output_dict=False)
        st.text("📈 Classification Report:")
        st.code(report)

        joblib.dump(clf, MODEL_PATH)


st.divider()


# ----------------------------
# 3. Predict peptide functionality
# ----------------------------
st.markdown("## 3. Predict function using trained model")

new_ids_input = st.text_area("🔢 Input new peptide IDs (one per line):", "ORF050.00000006\nORF050.00000007\nORF050.00000008")
new_ids = [pid.strip() for pid in new_ids_input.strip().splitlines() if pid.strip()]

if st.button("🚀 Predict"):
    try:
        clf = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error("Trained model not found. Please train a model first.")
        st.stop()

    with st.spinner("Predicting (1s delay per peptide for demo)..."):
        for pid in new_ids:
            X = get_numpy_dataset([pid])
            pred = clf.predict(X)[0]
            st.write(f"{pid} → Predicted Label: {pred}")
            time.sleep(1)
