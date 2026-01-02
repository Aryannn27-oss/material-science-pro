# Material Intelligence Pro — Research Dashboard

Material Intelligence Pro is a research-oriented Streamlit-based dashboard designed for systematic analysis of materials science datasets. The tool integrates data inspection, machine-learning-based imputation, feature engineering, classification, visualization, and prediction within a single, reproducible workflow.

This project is developed with an emphasis on **academic research use**, particularly aligned with practices common in **materials science and mechanical engineering laboratories**, including those in Japan, where clarity, methodological transparency, and reproducibility are prioritized.

---

## Overview

The objective of this project is to provide a unified platform for exploratory materials informatics studies. The dashboard enables researchers to preprocess incomplete experimental datasets, derive physically meaningful features, apply interpretable machine learning models, and visualize trends relevant to real-world material applications.

---

## Functional Modules

### Data Upload and Inspection

* Upload material property datasets in CSV format
* Automatic detection and validation of column names
* Summary of missing values and basic statistical descriptors

### Machine Learning–Based Imputation

* Imputation of missing material properties:

  * Yield Strength (Sy)
  * Brinell Hardness (BHN)
  * Vickers Hardness (HV)
* Random Forest Regression trained on physically motivated base features
* Quantitative evaluation using R² score and RMSE
* Consistent feature scaling using StandardScaler

### Feature Engineering

The following physically interpretable features are derived:

* StrengthRatio = Su / Sy
* ElasticityIndex = E / G
* Density_Modulus = Ro / E

These features are used in downstream classification and visualization tasks.

### Rule-Based Application Labeling

A heuristic labeling scheme assigns a RealLife_Application category based on established material property ranges:

* Tool Material
* Aerospace Alloy
* Automotive Alloy
* Structural Steel
* Lightweight Alloy
* General Purpose

These labels are intended for exploratory research and hypothesis generation and should not be interpreted as experimentally validated classifications.

### Classification Model

* Random Forest Classifier trained on base, target, and engineered features
* Prediction of material application category
* Performance evaluation using held-out test accuracy and classification report

### Visualization and Analysis

* Correlation matrices for feature relationships
* Feature importance analysis from trained classifier
* Three-dimensional clustering in Su–Sy–Ro space
* Distribution analysis using class-wise violin plots

### Interactive Prediction

* Single-sample prediction using user-defined material properties
* Output includes predicted application category and class probabilities when available

### Data Export

* Export of the fully processed dataset in CSV format
* Ensures reproducibility across analysis sessions

---

## Expected Data Format

Input datasets must be provided as CSV files with the following case-sensitive columns.

### Base Properties

* Su: Ultimate Tensile Strength
* E: Elastic Modulus
* G: Shear Modulus
* mu: Poisson’s Ratio
* Ro: Density

### Additional Properties

* Sy: Yield Strength
* Bhn: Brinell Hardness
* HV: Vickers Hardness

Datasets may contain missing values; these are handled internally by the imputation pipeline.

---

## Implementation Details

* Application Framework: Streamlit
* Data Processing: Pandas, NumPy
* Machine Learning: scikit-learn
* Visualization: Plotly, Matplotlib, Seaborn

### Models

* RandomForestRegressor for property imputation
* RandomForestClassifier for application classification

---

## Execution Instructions

```bash
# Clone repository
git clone https://github.com/your-username/material-intelligence-pro.git
cd material-intelligence-pro

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py
```

---

## Research Considerations

* Imputation models are trained exclusively on samples with complete base properties
* Feature scaling is applied prior to imputation and reused during prediction
* Application labels are heuristic and intended for exploratory analysis
* The system is designed for research and educational use, not for industrial deployment

---

## Intended Research Applications

* Materials informatics and data-driven materials research
* Preliminary analysis of experimental or compiled materials datasets
* Demonstration of machine learning methodologies in engineering education
* Feature importance and interpretability studies

---

## Author

Aryan Verma
B.Tech Undergraduate, IIT (BHU) Varanasi

---

## Disclaimer

This software is provided for research and educational purposes only. The authors make no guarantees regarding the correctness of predictions or labels. Validation by domain experts and experimental verification are required before any practical or industrial use.
