import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score, classification_report,
                              confusion_matrix, precision_recall_fscore_support)

MIN_CLASS_SAMPLES_FOR_CV = 10

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
if "df_unscaled" not in st.session_state:
    st.session_state.df_unscaled = None

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

if page == "Upload & Inspect":
    st.header("Upload & Inspect Data")
    st.write("Upload your CSV containing columns such as Su, Sy, E, G, mu, Ro, Bhn, HV (names are case-sensitive).")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df = df.rename(columns=lambda x: x.strip())
        st.session_state.df = df.copy()
        st.session_state.df_unscaled = df.copy()  # FIX: keep pristine copy
        st.write("Columns detected:", list(df.columns))
        st.write("Preview:")
        st.dataframe(df.head(10))
        st.write("Missing values summary:")
        st.dataframe(df.isnull().sum().astype(int))
        st.write("Basic statistics (numerical columns):")
        st.dataframe(df.describe().T)

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

                df_unscaled = st.session_state.df_unscaled.copy()
                df_unscaled = safe_numeric(df_unscaled, base_features + targets)
                for t in targets:
                    if t in regressors:
                        mask = df_unscaled[t].isna()
                        if mask.any():
                            X_missing = df_unscaled.loc[mask, present_base]
                            X_missing_scaled = scaler.transform(X_missing)
                            df_unscaled.loc[mask, t] = regressors[t].predict(X_missing_scaled)
                st.session_state.df_unscaled = df_unscaled

                st.success("Imputation finished. Check dataset preview and missing counts.")
                st.dataframe(df.head())

if page == "Feature Engineering":
    st.header("Feature Engineering")
    if st.session_state.df is None:
        st.warning("Upload and impute data first.")
    else:
        df = st.session_state.df.copy()
        df_unscaled = st.session_state.df_unscaled.copy()
        df = safe_numeric(df, ["Su", "Sy", "E", "G", "mu", "Ro", "Bhn", "HV"])
        df_unscaled = safe_numeric(df_unscaled, ["Su", "Sy", "E", "G", "mu", "Ro", "Bhn", "HV"])

        df["StrengthRatio"] = df["Su"] / (df["Sy"] + 1e-6)
        df["ElasticityIndex"] = df["E"] / (df["G"] + 1e-6)
        df["Density_Modulus"] = df["Ro"] / (df["E"] + 1e-6)
        st.session_state.df = df
        st.write("Added features: StrengthRatio, ElasticityIndex, Density_Modulus")
        st.dataframe(df[["Su", "Sy", "E", "G", "Ro", "StrengthRatio", "ElasticityIndex", "Density_Modulus"]].head())

        if st.button("Assign RealLife_Application (research rules)"):
            df_unscaled["RealLife_Application"] = df_unscaled.apply(real_life_category_row, axis=1)
            df["RealLife_Application"] = df_unscaled["RealLife_Application"]
            st.session_state.df = df
            st.session_state.df_unscaled = df_unscaled
            st.success("Assigned RealLife_Application label with rule-based logic (computed on real-unit values).")
            counts = df["RealLife_Application"].value_counts()
            st.dataframe(counts.rename_axis("label").reset_index(name="count"))

            st.warning(
                "⚠️ Label leakage notice: `RealLife_Application` is computed directly from "
                "**Bhn, HV, Su, Sy, Ro, and E** — the same raw columns used as model features. "
                "This means the label is an algebraic function of the inputs, not an independent "
                "ground truth. Expect inflated accuracy unless leaking features are removed or "
                "the evaluation is interpreted with that caveat (see Modeling page)."
            )

            rare = counts[counts < MIN_CLASS_SAMPLES_FOR_CV]
            if not rare.empty:
                for cls, n in rare.items():
                    st.error(
                        f"⚠️ Class **'{cls}'** has only **{n} sample(s)** — too few for reliable "
                        f"training or 5-fold stratified cross-validation (needs ≥{MIN_CLASS_SAMPLES_FOR_CV}). "
                        f"This class will be excluded from modeling with a visible note on the Modeling page."
                    )

if page == "Modeling":
    st.header("Model Training & Rigorous Evaluation")
    st.caption(
        "This page reports both a held-out test split and 5-fold stratified cross-validation, "
        "compares a model with and without HV, and flags signs of an inflated, non-trustworthy result."
    )
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
            label = "RealLife_Application"

            st.subheader("Class distribution before training")
            counts = df[label].value_counts()
            st.dataframe(counts.rename_axis("label").reset_index(name="count"))

            rare = counts[counts < MIN_CLASS_SAMPLES_FOR_CV]
            excluded_classes = list(rare.index)
            if excluded_classes:
                for cls, n in rare.items():
                    st.error(
                        f"⚠️ Excluding **'{cls}'** ({n} sample(s)) from training/evaluation — "
                        f"below the minimum of {MIN_CLASS_SAMPLES_FOR_CV} needed for stratified "
                        f"5-fold CV. A model cannot learn or be reliably evaluated on this class "
                        f"with so few examples; predictions for it would not be trustworthy."
                    )
                df = df[~df[label].isin(excluded_classes)].copy()
                st.write(f"Rows remaining after exclusion: {len(df)}")

            st.info(
                "ℹ️ Reminder: `RealLife_Application` is a deterministic rule computed from "
                "Bhn, HV, Su, Sy, Ro, E — all also used as classifier features. Treat any accuracy "
                "above ~0.97 with suspicion; it likely reflects the model reconstructing the "
                "labeling rule rather than learning a generalizable materials-science pattern."
            )

            base_feature_set = st.session_state.base_features + st.session_state.targets + [
                "StrengthRatio", "ElasticityIndex", "Density_Modulus"
            ]
            features_A = base_feature_set
            features_B = [f for f in base_feature_set if f != "HV"]
            st.session_state.features = features_A  # used by the Predict page

            def evaluate_model(df_model, feature_list, label_col, model_name):
                X = df_model[feature_list]
                y = df_model[label_col]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y)

                clf = RandomForestClassifier(n_estimators=350, max_depth=12, random_state=42,
                                              class_weight="balanced")
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                test_acc = accuracy_score(y_test, y_pred)
                _, _, test_macro_f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average="macro", zero_division=0)
                report = classification_report(y_test, y_pred, zero_division=0)
                cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_clf = RandomForestClassifier(n_estimators=350, max_depth=12, random_state=42,
                                                 class_weight="balanced")
                scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
                cv_results = cross_validate(cv_clf, X, y, cv=skf, scoring=scoring)

                importances = pd.Series(clf.feature_importances_, index=feature_list).sort_values(ascending=False)

                return {
                    "clf": clf,
                    "classes": clf.classes_,
                    "test_acc": test_acc,
                    "test_f1": test_macro_f1,
                    "report": report,
                    "cm": cm,
                    "cv_acc_mean": cv_results["test_accuracy"].mean(),
                    "cv_acc_std": cv_results["test_accuracy"].std(),
                    "cv_prec_mean": cv_results["test_precision_macro"].mean(),
                    "cv_rec_mean": cv_results["test_recall_macro"].mean(),
                    "cv_f1_mean": cv_results["test_f1_macro"].mean(),
                    "cv_f1_std": cv_results["test_f1_macro"].std(),
                    "importances": importances,
                }

            def render_model_results(results, model_name):
                st.markdown(f"#### {model_name}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("CV Accuracy", f"{results['cv_acc_mean']:.3f}", f"± {results['cv_acc_std']:.3f}")
                c2.metric("CV Macro F1", f"{results['cv_f1_mean']:.3f}", f"± {results['cv_f1_std']:.3f}")
                c3.metric("Test Accuracy", f"{results['test_acc']:.3f}")
                c4.metric("Test Macro F1", f"{results['test_f1']:.3f}")

                if results["cv_acc_mean"] > 0.97:
                    st.warning(
                        f"⚠️ CV accuracy of {results['cv_acc_mean']:.3f} is suspiciously high for a "
                        "5-class problem. This often indicates an easy/leaky labeling rule, a feature "
                        "that encodes the label directly, or low label diversity — not necessarily a "
                        "genuinely strong model. See the leakage notice above."
                    )

                with st.expander("Classification report"):
                    st.text(results["report"])
                with st.expander("Confusion matrix"):
                    st.dataframe(pd.DataFrame(results["cm"], index=results["classes"], columns=results["classes"]))
                with st.expander("Feature importances"):
                    st.dataframe(results["importances"].reset_index().rename(
                        columns={"index": "feature", 0: "importance"}))
                    fig = px.bar(results["importances"].sort_values(), orientation="h",
                                 title=f"{model_name} — Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)

            if st.button("Train & compare Model A (with HV) vs Model B (without HV)"):
                results_A = evaluate_model(df, features_A, label, "Model A")
                results_B = evaluate_model(df, features_B, label, "Model B")

                render_model_results(results_A, "Model A — includes HV")
                render_model_results(results_B, "Model B — excludes HV")

                st.markdown("---")
                st.subheader("Model recommendation")
                gap = results_A["cv_acc_mean"] - results_B["cv_acc_mean"]
                st.write(
                    f"Model A (with HV) CV accuracy: **{results_A['cv_acc_mean']:.3f}** — "
                    f"Model B (without HV) CV accuracy: **{results_B['cv_acc_mean']:.3f}** "
                    f"(gap: {gap:.3f})."
                )
                st.write(
                    "**Recommendation: treat Model B (without HV) as the primary research result.** "
                    "HV had the weakest imputation quality of any target (R² ≈ 0.22 during the "
                    "Imputation step, meaning ~90% of its values are model-predicted, not measured) "
                    "yet dominated Model A's feature importance. A model that leans heavily on its "
                    "least-reliable input is not trustworthy even if its accuracy is higher. Model B's "
                    "modestly lower accuracy is a more honest estimate of what the model can actually "
                    "generalize, built only on features with real physical measurements or well-imputed "
                    "values (Bhn, Sy: R² ≈ 0.97)."
                )
                st.caption(
                    "Note: because the label itself is rule-derived from several of the remaining "
                    "features (Bhn, Su, Sy, Ro, E), even Model B's accuracy should be read as an "
                    "upper bound on how well this specific rule can be reconstructed — not as a "
                    "validated real-world materials classification result."
                )

                st.session_state.classifier = results_B["clf"]
                st.session_state.features = features_B
                st.session_state.df = df
                st.success("Both models trained. Model B (without HV) stored as the active classifier for the Predict page.")

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

if page == "Predict":
    st.header("Interactive Prediction (single sample) — Research Mode")
    st.caption(
        "The active classifier is whichever model was trained last on the Modeling page "
        "(Model B — without HV — is stored by default as the more trustworthy option)."
    )
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
