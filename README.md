# Material Intelligence Pro — Research Dashboard

Material Intelligence Pro is a research-oriented Streamlit dashboard for systematic analysis of materials science datasets. It integrates data inspection, ML-based imputation, feature engineering, classification, visualization, and prediction within a single, reproducible workflow — with an explicit focus on **methodological transparency**: every modeling decision (leakage sources, evaluation protocol, class imbalance handling) is disclosed and quantified rather than presented only as a headline accuracy number.

This project is developed with an emphasis on **academic research use**, aligned with practices common in materials science and mechanical engineering laboratories, including those in Japan, where clarity, methodological rigor, and reproducibility are prioritized over optimistic point estimates.

---

## Dataset

- **1,552 samples**, 15 columns (material standard, ID, material name, heat treatment, and 8 numeric physical properties)
- Missing values prior to imputation: `Bhn` — 1,089 (70.2%), `HV` — 1,387 (89.4%), `Sy` — 8 (0.5%)

---

## Functional Modules

### Data Upload and Inspection
CSV upload with automatic column validation, missing-value summary, and descriptive statistics.

### Machine Learning–Based Imputation
Missing `Sy`, `Bhn`, and `HV` values are imputed with a `RandomForestRegressor` (n_estimators=250) trained on standardized base features (`Su`, `E`, `G`, `mu`, `Ro`), evaluated by held-out R² and RMSE:

| Target | R² | RMSE |
|---|---|---|
| Bhn | 0.966 | 17.95 |
| Sy | 0.966 | 49.19 |
| HV | 0.219 | 194.65 |

`HV`'s imputation quality is substantially weaker than the other two targets — with ~89% of its values model-predicted rather than measured, it is treated as a lower-confidence feature downstream (see *Model Comparison* below) rather than assumed reliable by default.

### Feature Engineering
Physically interpretable derived features:
- `StrengthRatio = Su / Sy`
- `ElasticityIndex = E / G`
- `Density_Modulus = Ro / E`

### Rule-Based Application Labeling
A heuristic rule assigns each sample a `RealLife_Application` category (Tool Material, Aerospace Alloy, Automotive Alloy, Structural Steel, Lightweight Alloy, General Purpose) from established property thresholds. Resulting class distribution:

| Class | Count |
|---|---|
| Tool Material | 805 |
| Automotive Alloy | 354 |
| Lightweight Alloy | 270 |
| General Purpose | 104 |
| Structural Steel | 17 |
| Aerospace Alloy | 2 |

**Labels are computed on unscaled, real-unit values** — an earlier iteration applied the same thresholds to standardized (post-`StandardScaler`) values, silently collapsing 3 of 5 categories to zero occurrences. This is corrected and documented in the pipeline as a methodological safeguard, not just a bug fix.

**Aerospace Alloy (n=2) is automatically excluded** from training and evaluation, since 5-fold stratified cross-validation requires at least 5 samples per class; the dashboard raises an explicit warning rather than silently dropping or misrepresenting this class.

### Classification: Methodology and Leakage Audit
Because `RealLife_Application` is a deterministic function of `Bhn`, `HV`, `Su`, `Sy`, `Ro`, and `E` — the same raw columns available as model inputs — naively training a classifier on all of them produces an inflated result (the model partially reconstructs the labeling rule rather than learning an independent pattern). The pipeline surfaces this explicitly and evaluates two models under a stricter protocol: `class_weight="balanced"`, a held-out test split, and **5-fold stratified cross-validation** reporting mean ± std accuracy and macro precision/recall/F1 (macro-averaging chosen specifically because the label distribution is imbalanced, ~52% Tool Material).

| Metric | Model A (includes HV) | Model B (excludes HV) |
|---|---|---|
| CV Accuracy | 0.990 ± 0.007 | 0.957 ± 0.008 |
| CV Macro F1 | 0.960 | 0.924 |
| Held-out Test Accuracy | 0.990 | 0.945 |
| Held-out Test Macro F1 | 0.982 | 0.917 |

**Model B (excludes HV) is adopted as the primary reported result.** `HV` combines the weakest imputation quality of any feature (R²=0.22) with the highest feature importance in Model A — a model leaning hardest on its least-reliable input is not a trustworthy result even though its accuracy is nominally higher. The dashboard also raises an automatic warning whenever cross-validated accuracy exceeds 0.97, since this range is disproportionately likely to reflect rule-reconstruction rather than genuine generalization.

### Visualization and Analysis
Correlation matrices, classifier feature importance, 3D Su–Sy–Ro clustering, and class-wise violin plots.

### Interactive Prediction
Single-sample prediction with class probabilities, using whichever model was most recently trained (Model B stored by default).

### Data Export
Full processed dataset exportable as CSV for reproducibility across sessions.

---

## Expected Data Format

**Base properties:** `Su` (Ultimate Tensile Strength), `E` (Elastic Modulus), `G` (Shear Modulus), `mu` (Poisson's Ratio), `Ro` (Density)
**Additional properties:** `Sy` (Yield Strength), `Bhn` (Brinell Hardness), `HV` (Vickers Hardness)

Column names are case-sensitive. Missing values are handled internally by the imputation pipeline.

---

## Implementation Details

- **Framework:** Streamlit
- **Data processing:** Pandas, NumPy
- **Machine learning:** scikit-learn (`RandomForestRegressor`, `RandomForestClassifier`, `StratifiedKFold`, `cross_validate`)
- **Visualization:** Plotly, Matplotlib, Seaborn

---

## Execution Instructions

```bash
git clone https://github.com/Aryannn27-oss/material-intelligence-pro.git
cd material-intelligence-pro
pip install -r requirements.txt
streamlit run app.py
```

---

## Research Considerations

- Imputation models are trained exclusively on samples with complete base properties.
- Feature scaling is applied prior to imputation and reused consistently at prediction time.
- `RealLife_Application` labels are rule-derived and heuristic; because the rule shares input columns with the classifier, **even the leakage-mitigated Model B accuracy should be read as an upper bound on how well the model reconstructs the labeling rule, not as a validated real-world materials classification.**
- The 99%-range accuracy an unaudited version of this pipeline would report is explicitly diagnosed, quantified, and reduced to a more defensible 0.957 ± 0.008 (5-fold CV) after removing the least-reliable, most leakage-prone feature.
- This system is designed for research and educational use; validation by domain experts and experimental verification are required before any practical or industrial application.

---

## Intended Research Applications

- Materials informatics and data-driven materials research
- Preliminary analysis of experimental or compiled materials datasets
- Demonstration of rigorous ML evaluation methodology (leakage auditing, cross-validation, imbalance handling) in engineering education
- Feature importance and interpretability studies

---

## Author

Aryan Verma
B.Tech Undergraduate, IIT (BHU) Varanasi

---

## Disclaimer

This software is provided for research and educational purposes only. The authors make no guarantees regarding the correctness of predictions or labels. Validation by domain experts and experimental verification are required before any practical or industrial use.
