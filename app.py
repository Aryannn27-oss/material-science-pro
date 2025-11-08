import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix

# Page config and lightweight dark styling
st.set_page_config(page_title="Material Intelligence Pro — Research Dashboard", page_icon="🧪", layout="wide")
st.markdown(
    """
    <style>
    .main { background-color: #0f1724; color: #e6eef8; }
    .stButton>button { background-color: #0ea5a4; color: white; border-radius:8px; }
    .stDownloadButton>button { background-color:#0ea5a4;color:white;border-radius:8px; }
    .sidebar .stButton>button{ background-color:#0ea5a4; }
    .css-1d391kg { color: #e6eef8; } /* headings */
    .stMarkdown p { color: #cde7ff; }
    footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("Material Intelligence Pro")
st.sidebar.markdown("Research Dashboard — dark theme")
page = st.sidebar.radio("Navigate", [
    "Upload & Inspect",
    "Imputation",
    "Feature Engineering",
    "Modeling",
    "Visualizations",
    "Predict",
    "Download / About"
])

# Utilities: session state containers
if "df" not in st.session_state:
    st.session_state.df = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "regressors" not in st.session_state:
    st.session_state.regressors = {}
if "classifier" not in st.session_state:
    st.session_state.classifier = None
if "features" not in st.session_state:
    st.session_state.features = None
if "base_features" not in st.session_state:
    st.session_state.base_features = ["Su", "E", "G", "mu", "Ro"]
if "targets" not in st.session_state:
    st.session_state.targets = ["Bhn", "HV", "Sy"]

# Helper functions
def safe_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def impute_target_with_rf(df, base_features, target, n_estimators=250):
    data = df[base_features + [target]].dropna(subset=base_features)
    train_data = data.dropna(subset=[target])
    predict_data = data[data[target].isna()]

    if len(train_data) < 20:
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(train_data[base_features], train_data[target],
                                                        test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    if not predict_data.empty:
        df.loc[predict_data.index, target] = model.predict(predict_data[base_features])

    return model, r2, rmse

def real_life_category_row(row):
    Su, Sy, E, G, mu, Ro, Bhn, HV = row.get("Su", np.nan), row.get("Sy", np.nan), row.get("E", np.nan), row.get("G", np.nan), row.get("mu", np.nan), row.get("Ro", np.nan), row.get("Bhn", np.nan), row.get("HV", np.nan)
    if pd.notna(Bhn) and Bhn > 250 or pd.notna(HV) and HV > 250:
        return "Tool Material"
    if pd.notna(Su) and pd.notna(Sy) and pd.notna(Ro) and (Su > 600) and (Sy > 400) and (Ro < 5000):
        return "Aerospace Alloy"
    if pd.notna(Su) and pd.notna(Sy) and pd.notna(Ro) and (300 <= Su <= 800) and (200 <= Sy <= 600) and (6500 <= Ro <= 8000):
        return "Automotive Alloy"
    if pd.notna(Su) and pd.notna(Sy) and pd.notna(Ro) and pd.notna(E) and (Su > 400) and (Sy > 250) and (Ro > 7500) and (E > 180000):
        return "Structural Steel"
    if pd.notna(Ro) and pd.notna(Su) and (Ro < 4000) and (Su < 500):
        return "Lightweight Alloy"
    return "General Purpose"

def train_classifier(df, features, label_col, n_estimators=350, max_depth=12):
    X = df[features].dropna()
    y = df.loc[X.index, label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())>1 else None)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    return clf, acc, report, cm, clf.classes_

# Page: Upload & Inspect
if page == "Upload & Inspect":
    st.header("Upload & Inspect Data")
    st.write("Upload your CSV containing columns such as Su, Sy, E, G, mu, Ro, Bhn, HV (names are case-sensitive).")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df = df.rename(columns=lambda x: x.strip())
        st.session_state.df = df.copy()
        st.write("Columns detected:", list(df.columns))
        st.write("Preview:")
        st.dataframe(df.head(10))
        st.write("Missing values summary:")
        st.dataframe(df.isnull().sum().astype(int))
        st.write("Basic statistics (numerical columns):")
        st.dataframe(df.describe().T)

# Page: Imputation
if page == "Imputation":
    st.header("Imputation (fill missing Bhn, HV, Sy using Random Forest regression)")
    if st.session_state.df is None:
        st.warning("Please upload data first in 'Upload & Inspect'.")
    else:
        df = st.session_state.df.copy()
        base_features = st.session_state.base_features
        targets = st.session_state.targets
        df = safe_numeric(df, base_features + targets)
        st.write("Number of rows:", df.shape[0])
        st.write("Missing values before imputation:")
        st.dataframe(df[targets].isnull().sum().astype(int))

        if st.button("Run regression-based imputation"):
            scaler = StandardScaler()
            # scale in place only for columns present (handle missing cols gracefully)
            present_base = [c for c in base_features if c in df.columns]
            if len(present_base) < len(base_features):
                st.error(f"Required base features missing: {set(base_features)-set(present_base)}")
            else:
                df[present_base] = scaler.fit_transform(df[present_base])
                st.session_state.scaler = scaler
                regressors = {}
                results = {}
                for t in targets:
                    if t not in df.columns:
                        st.warning(f"{t} not in dataset, skipping.")
                        continue
                    model, r2, rmse = impute_target_with_rf(df, present_base, t)
                    if model is not None:
                        regressors[t] = model
                        results[t] = (r2, rmse)
                        st.write(f"Imputed {t}: R²={r2:.3f}, RMSE={rmse:.3f}")
                    else:
                        st.warning(f"Not enough labeled rows to train regressor for {t}.")
                st.session_state.regressors = regressors
                st.session_state.df = df
                st.success("Imputation finished. Check dataset preview and missing counts.")
                st.dataframe(df.head())

# Page: Feature Engineering
if page == "Feature Engineering":
    st.header("Feature Engineering")
    if st.session_state.df is None:
        st.warning("Upload and impute data first.")
    else:
        df = st.session_state.df.copy()
        # Ensure numeric
        df = safe_numeric(df, ["Su", "Sy", "E", "G", "mu", "Ro", "Bhn", "HV"])
        df["StrengthRatio"] = df["Su"] / (df["Sy"] + 1e-6)
        df["ElasticityIndex"] = df["E"] / (df["G"] + 1e-6)
        df["Density_Modulus"] = df["Ro"] / (df["E"] + 1e-6)
        st.session_state.df = df
        st.write("Added features: StrengthRatio, ElasticityIndex, Density_Modulus")
        st.dataframe(df[["Su", "Sy", "E", "G", "Ro", "StrengthRatio", "ElasticityIndex", "Density_Modulus"]].head())

        if st.button("Assign RealLife_Application (research rules)"):
            df["RealLife_Application"] = df.apply(real_life_category_row, axis=1)
            st.session_state.df = df
            st.success("Assigned RealLife_Application label with rule-based logic.")
            st.dataframe(df["RealLife_Application"].value_counts().rename_axis("label").reset_index(name="count"))

# Page: Modeling
if page == "Modeling":
    st.header("Model Training (Random Forest Classifier)")
    if st.session_state.df is None:
        st.warning("Prepare data first (upload → impute → feature engineer).")
    else:
        df = st.session_state.df.copy()
        required_cols = st.session_state.base_features + st.session_state.targets + ["StrengthRatio", "ElasticityIndex", "Density_Modulus", "RealLife_Application"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error("Missing columns required for modeling: " + ", ".join(missing))
        else:
            df = df.dropna(subset=required_cols)
            features = st.session_state.base_features + st.session_state.targets + ["StrengthRatio", "ElasticityIndex", "Density_Modulus"]
            st.session_state.features = features
            label = "RealLife_Application"

            if st.button("Train classifier"):
                clf, acc, report, cm, classes = train_classifier(df, features, label)
                st.session_state.classifier = clf
                st.write(f"Accuracy on held-out test set: {acc:.3f}")
                st.text(report)
                st.session_state.df = df
                st.success("Classifier trained and stored in session.")

# Page: Visualizations
if page == "Visualizations":
    st.header("Interactive Visualizations — Research View")
    if st.session_state.df is None:
        st.warning("Load and prepare data first.")
    else:
        df = st.session_state.df.copy()
        features = st.session_state.features or (st.session_state.base_features + st.session_state.targets + ["StrengthRatio", "ElasticityIndex", "Density_Modulus"])
        st.subheader("Correlation matrix")
        corr = df[features].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="burg")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature importance (classifier)")
        if st.session_state.classifier is not None:
            importances = pd.Series(st.session_state.classifier.feature_importances_, index=features).sort_values()
            fig2 = px.bar(x=importances.values, y=importances.index, orientation='h', title="Feature Importance")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Train the classifier first to see feature importances.")

        st.subheader("3D clustering — Su vs Sy vs Ro")
        if "RealLife_Application" in df.columns:
            fig3 = px.scatter_3d(df, x="Su", y="Sy", z="Ro", color="RealLife_Application", size="StrengthRatio",
                                 hover_data=["E", "G", "Ro"], title="3D: Su-Sy-Ro by RealLife_Application")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No RealLife_Application column available.")

        st.subheader("Distribution by class (violin plots)")
        if "RealLife_Application" in df.columns:
            selected_feature = st.selectbox("Select numeric feature", features, index=0)
            fig4 = px.violin(df, x="RealLife_Application", y=selected_feature, box=True, points="all", color="RealLife_Application")
            st.plotly_chart(fig4, use_container_width=True)

# Page: Predict
if page == "Predict":
    st.header("Interactive Prediction (single sample) — Research Mode")
    if st.session_state.classifier is None:
        st.warning("Train the classifier in 'Modeling' first.")
    else:
        with st.form("predict_form"):
            st.write("Enter material properties (real values — not scaled):")
            Su = st.number_input("Ultimate Tensile Strength (Su)", value=400.0)
            Sy = st.number_input("Yield Strength (Sy)", value=250.0)
            E = st.number_input("Elastic Modulus (E)", value=200000.0)
            G = st.number_input("Shear Modulus (G)", value=80000.0)
            mu = st.number_input("Poisson's Ratio (mu)", value=0.3, min_value=0.0, max_value=1.0)
            Ro = st.number_input("Density (Ro)", value=7850.0)
            Bhn = st.number_input("Brinell Hardness (Bhn)", value=200.0)
            HV = st.number_input("Vickers Hardness (HV)", value=190.0)
            submitted = st.form_submit_button("Predict application")

        if submitted:
            # Build DataFrame and scale base features the same way
            input_df = pd.DataFrame({
                "Su": [Su], "E": [E], "G": [G], "mu": [mu], "Ro": [Ro], "Bhn": [Bhn], "HV": [HV], "Sy": [Sy]
            })
            scaler = st.session_state.scaler
            if scaler is None:
                st.error("Scaler not found. Re-run imputation step where scaler is created.")
            else:
                input_df[st.session_state.base_features] = scaler.transform(input_df[st.session_state.base_features])
                input_df["StrengthRatio"] = input_df["Su"] / (input_df["Sy"] + 1e-6)
                input_df["ElasticityIndex"] = input_df["E"] / (input_df["G"] + 1e-6)
                input_df["Density_Modulus"] = input_df["Ro"] / (input_df["E"] + 1e-6)
                X_input = input_df[st.session_state.features]
                pred = st.session_state.classifier.predict(X_input)[0]
                proba = None
                if hasattr(st.session_state.classifier, "predict_proba"):
                    proba = st.session_state.classifier.predict_proba(X_input)[0]
                st.success(f"Predicted Real-Life Application: {pred}")
                if proba is not None:
                    classes = st.session_state.classifier.classes_
                    probs = pd.Series(proba, index=classes).sort_values(ascending=False)
                    st.write("Prediction probabilities:")
                    st.dataframe(probs.reset_index().rename(columns={"index": "class", 0: "probability"}))

# Page: Download / About
if page == "Download / About":
    st.header("Download & About")
    if st.session_state.df is None:
        st.warning("No data to download. Upload and process a dataset first.")
    else:
        df = st.session_state.df.copy()
        st.write("Processed dataset preview:")
        st.dataframe(df.head())
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download processed dataset (CSV)", csv, "Processed_Material_Data.csv", "text/csv")

    st.markdown("---")
    st.markdown("**Material Intelligence Pro — Research Dashboard**")
    st.markdown("Built by Aryan Verma — IIT BHU")
    st.markdown(
        "<small style='color:#9fb2c8'>Notes: this dashboard is intended for research and exploratory analysis. "
        "Rule-based labels (RealLife_Application) are heuristics and should be validated with domain experts before deployment.</small>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
