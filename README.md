# 🚀 AttritionIQ – Employee Attrition Prediction & Retention Analytics

An interactive Machine Learning application that predicts employee attrition risk and provides HR analytics to support data-driven employee retention strategies.

🌐 **Live Demo:** https://attritioniq-qev6ks5fbhti3ctfpaeryr.streamlit.app/  
📂 **GitHub Repository:** https://github.com/manishhyaadavv/AttritionIQ

---

## 📌 Project Overview

Employee attrition is a major challenge for organizations. Losing employees can increase recruitment costs, reduce productivity, and affect business performance.

**AttritionIQ** uses Machine Learning to analyze employee data, estimate attrition risk, identify important factors, and explore potential retention improvements.

The application provides an interactive dashboard designed to help HR teams understand employee attrition patterns.

## 🎯 Project Objectives

- Predict employee attrition using Machine Learning.
- Identify important factors associated with employee attrition.
- Compare multiple classification algorithms.
- Segment employees based on predicted attrition risk.
- Provide interactive HR analytics and visualizations.
- Explore hypothetical employee retention scenarios.

## 📊 Dataset

**IBM HR Analytics Employee Attrition & Performance Dataset**

- Total Records: 1,470
- Total Features: 35
- Target Variable: Attrition
- Missing Values: 0
- Duplicate Records: 0

The dataset contains employee information such as age, department, job role, income, job satisfaction, overtime, and work-life balance.

## 🤖 Machine Learning Algorithms

Three classification algorithms were implemented:

1. Random Forest Classifier
2. Extra Trees Classifier
3. Gradient Boosting Classifier

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 85.03% | 71.43% | 10.64% | 18.52% | 0.7842 |
| Extra Trees | 85.37% | 66.67% | 17.02% | 27.12% | 0.8032 |
| Gradient Boosting | 79.93% | 39.60% | 48.94% | 43.81% | 0.7694 |

Gradient Boosting achieved the highest recall and F1-score among the three models, making it useful for identifying more potentially at-risk employees.

Extra Trees achieved the highest ROC-AUC score.

## ✨ Application Features

### 1. Interactive Dashboard
- Employee statistics and KPIs
- Attrition distribution
- Department-wise analytics
- Job role analysis
- Age and gender analysis
- Interactive filters

### 2. Employee Prediction
- Enter employee details.
- Predict attrition probability.
- View predicted employee risk.

### 3. Model Comparison
- Compare three ML algorithms.
- Visualize model performance metrics.

### 4. Feature Importance
- Identify influential model features.
- Visualize feature importance rankings.

### 5. SHAP Explainability
- Understand global feature contributions.
- Explore individual prediction explanations.

### 6. Advanced Evaluation
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Classification metrics

### 7. Employee Risk Segmentation
- High Risk
- Medium Risk
- Low Risk
- Download employee risk reports.

### 8. Employee Risk Comparison
Compare predicted attrition risks between employees.

### 9. Employee Retention Simulator
Explore how hypothetical changes in overtime, job satisfaction, and work-life balance affect model predictions.

*Simulation results are model estimates and do not establish causal relationships.*

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- SHAP
- Joblib

## 📁 Project Structure

```text
AttritionIQ/
│
├── app.py
├── attrition_models.pkl
├── preprocessor.pkl
├── requirements.txt
├── runtime.txt
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── README.md
└── .gitignore
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/manishhyaadavv/AttritionIQ.git
```

### 2. Navigate to the Project Directory

```bash
cd AttritionIQ
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 👥 Team Members

**Group 3 – Lazy Developers**

- Manish Yadav
- Tanvi Rao
- Vineet Kashyap
- Chetan Sharma

**Track:** Track 3 – Project Showcase  
**Project:** Applied Machine Learning (AML)

## 🎓 Project Purpose

This project was developed as part of an Applied Machine Learning academic project to demonstrate practical applications of classification algorithms, model evaluation, explainable AI, and interactive data analytics.

---

## ⭐ Support

If you find this project interesting, consider giving the repository a star!

**Built with Python, Machine Learning & Streamlit.**
