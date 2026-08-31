# Heart-Disease-Predictor
Machine learning classification models predicting cardiovascular risk with up to 88.04% accuracy using non-invasive clinical attributes and feature importance evaluation.
# Heart Disease Prediction Using Machine Learning Classification Techniques

A comparative analysis of supervised machine learning classification algorithms to predict cardiovascular disease risk using non-invasive clinical attributes from Kaggle's Heart Failure Prediction Dataset.

## 📌 Project Overview
Early detection of cardiovascular disease (CVD) is critical for timely clinical intervention[cite: 1]. This project evaluates five supervised machine learning models—Logistic Regression, K-Nearest Neighbors, Support Vector Classification, Decision Tree, and Random Forest—to determine the most accurate and interpretable model for predicting heart disease[cite: 1].

The dataset contains 918 patient records across 11 clinical features sourced from five combined heart disease datasets[cite: 1].

## 📊 Key Results

Random Forest and Support Vector Classification (SVC) achieved the highest overall test accuracy at **88.04%**[cite: 1]. Random Forest provided the highest 10-fold cross-validation performance (**87.60%**) and clearest feature interpretability[cite: 1].

| Model | Training Acc. (%) | Testing Acc. (%) | F1-Score | 10-Fold CV Acc. (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 87.06 | 86.41 | 0.87 | 86.11 |
| **K-Nearest Neighbors** | 91.69 | 86.41 | 0.87 | 86.38 |
| **Support Vector Classification** | 91.28 | 88.04 | 0.89 | 86.78 |
| **Decision Tree Classification** | 88.15 | 85.33 | 0.86 | 83.24 |
| **Random Forest Classification** | 100.00 | 88.04 | 0.89 | 87.60 |

## 🔑 Key Insights & Feature Importance
Based on the Random Forest feature importance analysis, the top four predictors accounting for over 50% of the model's decision-making weight are[cite: 1]:
1. **ST_Slope_Up** – Slope of peak exercise ST segment[cite: 1]
2. **Oldpeak** – Exercise-induced ST depression[cite: 1]
3. **Cholesterol** – Serum cholesterol levels[cite: 1]
4. **MaxHR** – Maximum heart rate achieved[cite: 1]

## 🛠 Tech Stack & Workflow
* **Language & Libraries:** Python, NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn[cite: 1]
* **Preprocessing:** One-Hot Encoding, Label Encoding, `StandardScaler` (z-score normalization), ANOVA F-test statistical validation[cite: 1]
* **Evaluation:** 80:20 Stratified Train-Test Split, 10-Fold Cross-Validation, Confusion Matrices, Precision/Recall/F1-Score[cite: 1]

## 📁 Repository Structure
```text
├── data/                  # Heart failure prediction dataset
├── notebooks/             # Jupyter notebook with complete code and visualizations
├── models/                # Saved model files (scaler.pk, Random_Forest_Classifier.pk)
├── report/                # Industrial training report document
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
