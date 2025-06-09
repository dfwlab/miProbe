import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from peptide_embedding_toolkit import get_numpy_dataset
import joblib
import time

st.set_page_config(page_title="Demo", layout="centered")

st.title("🔬 Peptide Embedding Database Demo")

################
st.markdown("## 1. 通过API获取多肽序列或embedding数据")
code = '''from peptide_embedding_toolkit import get_numpy_dataset
peptide_ids = ['PEP0001', 'PEP0002', 'PEP0003']
X = get_numpy_dataset(peptide_ids, embedding_type='prottrans')
'''
st.code(code, language="python")
if st.button("获取信息"):
    peptide_ids = ['PEP0001', 'PEP0002', 'PEP0003']
    st.write(get_numpy_dataset(peptide_ids, embedding_type='prottrans'))

################
st.markdown("## 2. 使用预训练的 embedding 特征 + scikit-learn 分类器。")
# 输入多肽ID
peptide_ids_input = st.text_area("🔢 输入多肽ID（每行一个）：", "PEP0001\nPEP0002\nPEP0003\nPEP0004\nPEP0005")
peptide_ids = [pid.strip() for pid in peptide_ids_input.strip().splitlines() if pid.strip()]

# 输入标签
labels_input = st.text_input("🧬 输入对应的功能标签（如0/1，逗号分隔）", "1,0,1,1,0")
try:
    labels = [int(x) for x in labels_input.strip().split(',')]
except:
    st.error("标签格式错误，请用逗号分隔，如 1,0,1")
    st.stop()
    
# 验证输入匹配
if len(labels) != len(peptide_ids):
    st.error("多肽数量与标签数量不一致")
    st.stop()

# 按钮触发建模
MODEL_PATH = "peptide_model.pkl"
if st.button("🚀 开始训练模型"):
    with st.spinner("正在下载 embedding 并训练模型..."):
        X = get_numpy_dataset(peptide_ids, embedding_type='prottrans')
        y = np.array(labels)

        # 简单分类器
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        # 展示训练报告
        preds = clf.predict(X)
        report = classification_report(y, preds, output_dict=False)
        st.text("📈 分类报告：")
        st.code(report)

        # 保存模型
        joblib.dump(clf, MODEL_PATH)
        
        # 展示预测结果
        #st.text("🔍 预测结果：")
        #for pid, pred in zip(peptide_ids, preds):
        #    st.write(f"{pid} → 预测标签: {pred}")

################
st.markdown("## 3. 实时获取多肽embedding，通过模型批量预测多肽特征。")
# 输入多肽ID
peptide_ids_input = st.text_area("🔢 输入多肽ID（每行一个）：", "PEP0006\nPEP0007\nPEP0008\nPEP0009\nPEP0010")
peptide_ids = [pid.strip() for pid in peptide_ids_input.strip().splitlines() if pid.strip()]
# 按钮触发建模
if st.button("🚀 开始预测(sleep 1 second for each peptide)"):
    clf = joblib.load(MODEL_PATH)
    with st.spinner("正在下载 embedding 并预测结果..."):
        for pid in peptide_ids:
            X = get_numpy_dataset([pid], embedding_type='prottrans')
            pred = clf.predict(X)[0]
            st.write(f"{pid} → 预测标签: {pred}")
            time.sleep(1)

