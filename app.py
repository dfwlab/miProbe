import streamlit as st
import numpy as np
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from miprobe_toolkit import get_numpy_dataset

st.set_page_config(page_title="miProbe", layout="wide")

st.title("🔬 miProbe Toolkit Demo")

# ------------------------
# Section 1: API Usage
# ------------------------
st.markdown("## 1. Retrieve peptide sequence or embedding via API")
code = '''from miprobe_toolkit import get_numpy_dataset
peptide_ids = ['ORF050.00000001', 'ORF050.00000002', 'ORF050.00000003']
X = get_numpy_dataset(peptide_ids, embedding_type='prottrans')
'''
st.code(code, language="python")

if st.button("Fetch data via API"):
    peptide_ids = ['ORF050.00000001', 'ORF050.00000002', 'ORF050.00000003']
    st.write(get_numpy_dataset(peptide_ids, embedding_type='prottrans'))

st.divider()

# ------------------------
# Section 2: Train model
# ------------------------
st.markdown("## 2. Train a scikit-learn classifier using peptide embeddings")

# Input peptide IDs
peptide_ids_input = st.text_area("🔢 Enter peptide IDs (one per line):", "ORF050.00000001\nORF050.00000002\nORF050.00000003\nORF050.00000004\nORF050.00000005")
peptide_ids = [pid.strip() for pid in peptide_ids_input.strip().splitlines() if pid.strip()]

# Input labels
labels_input = st.text_input("🧬 Enter binary labels (comma-separated):", "1,0,1,1,0")
try:
    labels = [int(x) for x in labels_input.strip().split(',')]
except:
    st.error("Label format error. Please use comma-separated values like 1,0,1")
    st.stop()

# Validate input match
if len(labels) != len(peptide_ids):
    st.error("Number of labels does not match number of peptide IDs.")
    st.stop()

# Train and save model
MODEL_PATH = "peptide_model.pkl"
if st.button("🚀 Train Model"):
    with st.spinner("Fetching embeddings and training model..."):
        X = get_numpy_dataset(peptide_ids, embedding_type='prottrans')
        y = np.array(labels)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        preds = clf.predict(X)
        report = classification_report(y, preds, output_dict=False)
        st.text("📈 Classification Report:")
        st.code(report)

        joblib.dump(clf, MODEL_PATH)

st.divider()

# ------------------------
# Section 3: Predict new peptides
# ------------------------
st.markdown("## 3. Predict peptide function with saved model")

peptide_ids_input = st.text_area("🔢 Enter new peptide IDs (one per line):", "ORF050.00000006\nORF050.00000007\nORF050.00000008\nORF050.00000009\nORF050.00000010")
peptide_ids = [pid.strip() for pid in peptide_ids_input.strip().splitlines() if pid.strip()]

if st.button("🚀 Predict Function"):
    clf = joblib.load(MODEL_PATH)
    with st.spinner("Predicting (sleep 1s per peptide for demo)..."):
        for pid in peptide_ids:
            X = get_numpy_dataset([pid], embedding_type='prottrans')
            pred = clf.predict(X)[0]
            st.write(f"{pid} → predicted label: {pred}")
            time.sleep(1)
