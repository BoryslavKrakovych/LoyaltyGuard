from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def metric_row(col1, label1, value1, col2, label2, value2, col3, label3, value3):
    with col1:
        st.metric(label1, value1)
    with col2:
        st.metric(label2, value2)
    with col3:
        st.metric(label3, value3)


def plot_roc_curve(fpr, tpr, roc_auc: float):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve for churn model')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    st.pyplot(fig)


def download_dataframe_button(df: pd.DataFrame, file_name: str, label: str):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode('utf-8'),
        file_name=file_name,
        mime='text/csv',
    )
