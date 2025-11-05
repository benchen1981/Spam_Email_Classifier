from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, f1_score, precision_score, recall_score
)
import numpy as np
import pandas as pd

def available_models():
    return ["Logistic Regression", "Naive Bayes", "Random Forest", "SVM"]

def train_and_predict(
    df, label_col, text_col,
    model_name, test_size, seed, threshold
):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df[text_col])
    y = (df[label_col].str.lower() == "spam").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    if model_name == "Naive Bayes":
        clf = MultinomialNB()
    elif model_name == "Logistic Regression":
        clf = LogisticRegression()
    elif model_name == "Random Forest":
        clf = RandomForestClassifier()
    elif model_name == "SVM":
        clf = SVC(probability=True)
    clf.fit(Xtr, ytr)
    probas = clf.predict_proba(Xte)[:, 1]
    ypred = (probas > threshold).astype(int)
    metrics = {
        "confusion": confusion_matrix(yte, ypred),
        "roc": roc_curve(yte, probas),
        "pr": precision_recall_curve(yte, probas),
        "auc": auc(*roc_curve(yte, probas)[:2]),
        "precision": precision_score(yte, ypred, zero_division=0),
        "recall": recall_score(yte, ypred, zero_division=0),
        "f1": f1_score(yte, ypred, zero_division=0),
        "yte": yte,
        "ypred": ypred,
        "probas": probas
    }
    return metrics, yte, ypred, probas

def get_model_metrics(metrics):
    return f"""
    精確度 Precision: {metrics['precision']:.3f}
    召回率 Recall:   {metrics['recall']:.3f}
    F1分數:         {metrics['f1']:.3f}
    AUC:            {metrics['auc']:.3f}
    """

def live_predict(
    text, model_name, threshold
):
    # DEMO用，正式用法應串接已訓練持久化模型
    dummy_prob = 0.6
    label = "Spam" if dummy_prob > threshold else "Ham"
    return label, dummy_prob
