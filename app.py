import streamlit as st
import pandas as pd
import numpy as np

# ================= IMPORT MODELS =================
from models.knn import KNN
from models.svm import SVM
from models.naive_bayes import NaiveBayes
from models.decision_tree import DecisionTree
from models.perceptron_single import SingleLayerPerceptron
from models.perceptron_multi import MultiLayerPerceptron

from decomposition.pca import PCA
from decomposition.fourier import FourierTransform

from utils.metrics import accuracy

# ================= PAGE CONFIG =================
st.set_page_config(page_title="ML Models", layout="wide")

st.title("Machine Learning Models")
st.markdown(
    "Upload **any dataset** → automatic preprocessing → intelligent model execution."
)
st.divider()

# ================= CSV LOADER =================
def load_csv(file):
    for params in [
        {},
        {"encoding": "latin1"},
        {"sep": ";"},
        {"sep": ";", "encoding": "latin1"},
    ]:
        try:
            return pd.read_csv(file, **params)
        except Exception:
            pass
    st.error("❌ Unable to read CSV file.")
    st.stop()

# ================= PREPROCESSING =================
def preprocess_dataframe(df):
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df

# ================= LABEL DETECTION =================
def detect_label(df):
    common = ["label", "target", "class", "y", "output", "diagnosis", "species"]

    for col in df.columns:
        if col.lower() in common:
            return col

    for col in reversed(df.columns):
        unique_vals = df[col].nunique()
        if unique_vals <= max(10, int(0.05 * len(df))):
            return col

    return None

# ================= LABEL ENCODING =================
def encode_labels(y):
    """
    Encode non-numeric labels into integers.
    Handles mixed types and missing values safely.
    """
    # If already numeric, just return
    if np.issubdtype(y.dtype, np.number):
        return y, None

    # Convert everything to string (safe for mixed types)
    y_clean = pd.Series(y).astype(str).values

    unique = np.unique(y_clean)
    mapping = {label: idx for idx, label in enumerate(unique)}
    y_encoded = np.array([mapping[val] for val in y_clean])

    return y_encoded, mapping


# ================= DATA PREPARATION =================
def prepare_supervised_data(df, model_name):
    label = detect_label(df)

    if label is None:
        st.error("❌ No suitable label column detected.")
        st.stop()

    X = df.drop(columns=[label]).select_dtypes(include=np.number).values
    y_raw = df[label].values
    y, label_map = encode_labels(y_raw)

    if label_map:
        st.info(f"ℹ️ Label encoding applied: {label_map}")

    if X.shape[1] == 0:
        st.error("❌ No numeric features available.")
        st.stop()

    if model_name == "SLP" and len(np.unique(y)) != 2:
        st.warning("⚠️ Single Layer Perceptron supports binary classification only.")
        st.stop()

    return X, y

# ================= STRATIFIED SPLIT =================
def smart_split(X, y):
    if len(X) < 30:
        return X, X, y, y

    train_idx, test_idx = [], []
    classes = np.unique(y)

    for c in classes:
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        split = int(0.8 * len(idx))
        train_idx.extend(idx[:split])
        test_idx.extend(idx[split:])

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# ================= NORMALIZATION =================
def normalize(X):
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

# ================= DATASET PROFILING =================
def profile_dataset(df, label):
    return {
        "samples": len(df),
        "features": df.drop(columns=[label]).select_dtypes(include=np.number).shape[1],
        "classes": df[label].nunique()
    }

# ================= MODEL RECOMMENDATION =================
def recommend_models(p):
    rec = []

    if p["samples"] < 1000:
        rec += ["KNN", "Decision Tree"]

    if 1000 <= p["samples"] <= 10000:
        rec += ["SVM", "Naive Bayes"]

    if p["samples"] > 10000:
        rec += ["MLP"]

    if p["features"] > 50:
        rec.append("PCA")

    if p["classes"] == 2:
        rec.append("Single Layer Perceptron")

    return list(set(rec))

# ================= MODEL TILE =================
def model_tile(title, model_name, model_builder):
    st.subheader(title)
    file = st.file_uploader(f"Upload CSV ({model_name})", type=["csv"], key=model_name)

    if file:
        df = preprocess_dataframe(load_csv(file))
        label = detect_label(df)

        if label:
            prof = profile_dataset(df, label)
            recs = recommend_models(prof)

            if model_name in recs:
                st.success("✅ Recommended for this dataset")
            else:
                st.info(f"ℹ️ Other recommended models: {', '.join(recs)}")

        X, y = prepare_supervised_data(df, model_name)
        X = normalize(X)

        # KNN safety
        if model_name == "KNN" and X.shape[0] > 5000:
            st.warning(
                f"⚠️ Dataset has {X.shape[0]} samples.\n"
                "KNN scales poorly on large datasets.\n"
                "Please consider SVM or MLP."
            )
            st.stop()

        X_train, X_test, y_train, y_test = smart_split(X, y)

        if st.button(f"▶ Run {model_name}", key=f"run_{model_name}"):
            with st.spinner(f"Running {model_name}..."):
                model = model_builder()

                if model_name == "NB":
                    X_train += 1e-9
                    X_test += 1e-9

                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                st.success(f"Accuracy: {accuracy(y_test, preds):.3f}")

# ================= ROW 1 =================
c1, c2 = st.columns(2)
with c1:
    model_tile("🔹 KNN", "KNN", lambda: KNN(k=3))
with c2:
    model_tile("🔹 SVM", "SVM", lambda: SVM())

st.divider()

# ================= ROW 2 =================
c3, c4 = st.columns(2)
with c3:
    model_tile("🔹 Naive Bayes", "NB", lambda: NaiveBayes())
with c4:
    model_tile("🔹 Decision Tree", "DT", lambda: DecisionTree())

st.divider()

# ================= ROW 3 =================
c5, c6 = st.columns(2)
with c5:
    model_tile("🔹 Single Layer Perceptron", "SLP", lambda: SingleLayerPerceptron())
with c6:
    model_tile(
        "🔹 Multi Layer Perceptron",
        "MLP",
        lambda: MultiLayerPerceptron(hidden_size=16, lr=0.01, n_iters=3000),
    )

st.divider()

# ================= ROW 4 (UNSUPERVISED) =================
c7, c8 = st.columns(2)

with c7:
    st.subheader("🔹 PCA")
    file = st.file_uploader("Upload CSV (PCA)", type=["csv"], key="PCA")
    if file:
        df = preprocess_dataframe(load_csv(file))
        X = df.select_dtypes(include=np.number).values
        if st.button("▶ Run PCA"):
            X_pca = PCA(n_components=2).fit_transform(X)
            st.scatter_chart(X_pca)

with c8:
    st.subheader("🔹 Fourier Transform")
    file = st.file_uploader("Upload CSV (Fourier)", type=["csv"], key="FT")
    if file:
        df = preprocess_dataframe(load_csv(file))
        X = df.select_dtypes(include=np.number).values
        if st.button("▶ Run Fourier"):
            X_ft = FourierTransform().transform(X)
            st.line_chart(np.abs(X_ft[0]))

# ================= FOOTER =================
st.divider()
st.markdown(
    "<p style='text-align:center;color:gray;'>by <b>Parthivi Thakore</b></p>",
    unsafe_allow_html=True
)
