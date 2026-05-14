# 🏦 Loan Default Predictor

An end-to-end Machine Learning project that predicts whether a loan applicant will default, built on 307,511 real loan applications from Home Credit Group.

## 🎯 Project Overview
Banks lose billions every year from loan defaults. This project builds an AI system that predicts default risk before a loan is approved — and explains exactly why using SHAP explainability.

## 📊 Results
| Model | AUC Score |
|---|---|
| Logistic Regression (baseline) | 0.6372 |
| XGBoost (final) | 0.7271 |

## 🛠️ Tech Stack
| Layer | Tools |
|---|---|
| Data processing | pandas, numpy, scikit-learn |
| Imbalance fix | SMOTE (imbalanced-learn) |
| ML Model | XGBoost |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Deployment | Docker |

## 🚀 How to run locally
bash
git clone https://github.com/YOUR_USERNAME/loan-default-predictor.git
cd loan-default-predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py


## 📁 Project Structure
loan-default-predictor/
├── data/
│   ├── raw/              ← Kaggle dataset (not in repo — download separately)
│   └── processed/        ← Cleaned data (not in repo — run notebooks to generate)
├── notebooks/
│   ├── 01_eda.ipynb              ← Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb   ← Cleaning, SMOTE, feature engineering
│   ├── 03_modeling.ipynb         ← XGBoost training and evaluation
│   └── 04_explainability.ipynb  ← SHAP explainability
├── model/
│   └── feature_names.pkl ← Saved feature names
├── app/
│   └── main.py           ← Streamlit dashboard
├── Dockerfile
└── requirements.txt

## 📓 How to reproduce
1. Download `application_train.csv` from [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)
2. Place it in `data/raw/`
3. Run notebooks in order: 01 → 02 → 03 → 04
4. Run `streamlit run app/main.py`

## 👤 Author
Built by [Your Name] — Computer Science (AI) student