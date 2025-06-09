import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from peptide_embedding_toolkit import get_numpy_dataset

st.set_page_config(page_title="Peptide Classifier Demo", layout="centered")

st.title("🔬 Peptide Function Prediction Demo")
st.markdown("使用预训练的 embedding 特征 + scikit-learn 分类器，演示多肽功能预测任务。")

# 输入多肽ID
peptide_ids_input = st.text_area("🔢 输入多肽ID（每行一个）：", "PEP0001\nPEP0002\nPEP0003")
peptide_ids = [pid.strip() for pid in peptide_ids_input.strip().splitlines() if pid.strip()]

# 输入标签
labels_input = st.text_input("🧬 输入对应的功能标签（如0/1，逗号分隔）", "1,0,1")
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

        # 展示预测结果
        st.text("🔍 预测结果：")
        for pid, pred in zip(peptide_ids, preds):
            st.write(f"{pid} → 预测标签: {pred}")
