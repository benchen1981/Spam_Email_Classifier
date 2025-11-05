import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, f1_score, precision_score, recall_score

def plot_confusion(yte, ypred):
    cm = confusion_matrix(yte, ypred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap="Blues")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)

def plot_roc_pr(yte, probas):
    fpr, tpr, _ = roc_curve(yte, probas)
    prec, rec, _ = precision_recall_curve(yte, probas)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(fpr, tpr)
    axs[0].set_title("ROC Curve")
    axs[1].plot(rec, prec)
    axs[1].set_title("PR Curve")
    st.pyplot(fig)

def plot_threshold_sweep(yte, probas):
    ths = np.round(np.linspace(0.3, 0.8, 11), 3)
    table = []
    for t in ths:
        ypred = (probas > t).astype(int)
        prec = precision_score(yte, ypred, zero_division=0)
        rec = recall_score(yte, ypred, zero_division=0)
        f1v = f1_score(yte, ypred, zero_division=0)
        table.append({"Threshold": t, "Precision": prec, "Recall": rec, "F1": f1v})
    st.table(pd.DataFrame(table))
