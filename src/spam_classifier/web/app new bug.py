import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    f1_score, precision_score, recall_score, roc_auc_score
)
from collections import Counter

# -------- Utility Functions --------

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def infer_cols(df: pd.DataFrame):
    label = next((c for c in df.columns if 'label' in c.lower() or 'target' in c.lower()), df.columns[0])
    text = next((c for c in df.columns if 'text' in c.lower() or 'message' in c.lower()), df.columns[-1])
    return label, text

def token_topn(series: pd.Series, topn: int =20) -> pd.DataFrame:
    counter = Counter(" ".join(series.astype(str)).split())
    return pd.DataFrame(counter.most_common(topn), columns=['token', 'count'])

def available_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True)
    }

# -------- Classification Function --------

def classify_text(text, model, vectorizer, threshold):
    xtest = vectorizer.transform([text])
    proba = model.predict_proba(xtest)[0][1]
    pred = "Spam" if proba > threshold else "Ham"
    return pred, proba

# -------- Main App --------

def main():
    st.set_page_config(page_title="Spam Classifier Professional", layout="wide")
    st.title("📧 AI Spam/Ham 郵件分類系統（模組化專業版）")
    st.caption("資料路徑 example: datasets/processed/sms_spam_clean.csv")

    # Sidebar controls
    with st.sidebar:
        st.header("資料與模型選擇")
        data_path = st.text_input("資料集路徑", "datasets/processed/sms_spam_clean.csv")
        df = load_data(data_path)
        if not df.empty:
            label_col, text_col = infer_cols(df)
            st.info(f"Auto label col: `{label_col}` / text col: `{text_col}`")
            model_name = st.selectbox("模型", list(available_models().keys()))
            test_size = st.slider("測試集比例", 0.1, 0.4, 0.2, 0.01)
            threshold = st.slider("Spam 閾值", 0.1, 0.9, 0.5, 0.01)
            random_seed = st.number_input("隨機種子", value=42, step=1)
        else:
            label_col, text_col = "",""
            st.warning("No data loaded.")

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 分類體驗", "📊 資料與詞頻", "🧪 性能比較", "📜 歷史紀錄"])

    # --- Classification Demo Tab ---
    with tab1:
        st.subheader("即時郵件 Spam/Ham 分類")
        user_input = st.text_area("請輸入郵件內容", height=150)
        run_pred = st.button("進行分類")
        if run_pred:
            if not user_input.strip():
                st.error("請輸入郵件內容。")
            elif df.empty or label_col == "" or text_col == "":
                st.error("未載入資料。")
            else:
                try:
                    X = df[text_col]
                    y = (df[label_col].astype(str).str.lower()=="spam").astype(int)
                    tfidf = TfidfVectorizer()
                    X_vec = tfidf.fit_transform(X)
                    model = available_models()[model_name]
                    model.fit(X_vec, y)
                    pred, proba = classify_text(user_input, model, tfidf, threshold)
                    st.metric("預測結果", pred, f"Spam 機率：{proba:.3f} (閾值 {threshold})")
                    st.write({"機率(Ham)": round(1-proba,3), "機率(Spam)": round(proba,3)})
                    # Save to history
                    if "history" not in st.session_state:
                        st.session_state["history"] = []
                    st.session_state["history"].append({
                        "input": user_input,
                        "prediction": pred,
                        "probability_spam": proba,
                        "timestamp": pd.Timestamp.now()
                    })
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

    # --- Data Analysis Tab ---
    with tab2:
        st.subheader("資料狀態")
        if not df.empty:
            st.write(df.head())
            st.write("標籤分佈")
            st.bar_chart(df[label_col].value_counts())
            st.write("訊息長度分布")
            st.bar_chart(df[text_col].apply(len))
            st.write("高頻詞 (Ham)")
            st.table(token_topn(df[df[label_col].astype(str).str.lower()=="ham"][text_col], 20))
            st.write("高頻詞 (Spam)")
            st.table(token_topn(df[df[label_col].astype(str).str.lower()=="spam"][text_col], 20))

    # --- Model Performance Tab ---
    with tab3:
        st.subheader("多模型性能對比")
        if not df.empty:
            try:
                X = df[text_col]
                y = (df[label_col].astype(str).str.lower()=="spam").astype(int)
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_seed, stratify=y)
                tfidf = TfidfVectorizer()
                Xtrv = tfidf.fit_transform(Xtr)
                Xtev = tfidf.transform(Xte)
                perf = []
                for name, clf in available_models().items():
                    clf.fit(Xtrv, ytr)
                    probas = clf.predict_proba(Xtev)[:, 1]
                    ypred = (probas > threshold).astype(int)
                    metrics = {
                        "Model": name,
                        "Precision": precision_score(yte, ypred),
                        "Recall": recall_score(yte, ypred),
                        "F1": f1_score(yte, ypred),
                        "AUC": roc_auc_score(yte, probas)
                    }
                    perf.append(metrics)
                st.write(pd.DataFrame(perf).set_index("Model"))
                # Confusion Matrix & Curves
                chosen_model = available_models()[model_name]
                chosen_model.fit(Xtrv, ytr)
                cprob = chosen_model.predict_proba(Xtev)[:, 1]
                ypredc = (cprob > threshold).astype(int)
                cm = confusion_matrix(yte, ypredc)
                st.write("Confusion Matrix", pd.DataFrame(cm, index=["Ham", "Spam"], columns=["Pred Ham", "Pred Spam"]))
                fpr, tpr, _ = roc_curve(yte, cprob)
                precs, recs, _ = precision_recall_curve(yte, cprob)
                st.line_chart({"FPR":fpr, "TPR":tpr})
                st.line_chart({"Recall":recs, "Precision":precs})
            except Exception as e:
                st.error(f"性能比較錯誤: {str(e)}")

    # --- History Tab ---
    with tab4:
        st.subheader("分類歷史紀錄")
        history = st.session_state.get("history", [])
        if history:
            st.table(pd.DataFrame(history))
        else:
            st.write("目前尚無分類紀錄。")

if __name__ == "__main__":
    main()
