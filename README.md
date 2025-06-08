# Project-RUL-SOH-Prediction
Rest useful life / State of Health of  a Battery in EV

# 🔋 Battery Health Prediction using Classical Machine Learning

This project aims to predict the **health status of a battery** based on various sensor readings and usage parameters using classical machine learning models like Logistic Regression, Random Forest, and Support Vector Machine.

---

## 📌 Problem Statement

Battery degradation is a key concern in electric vehicles (EVs), smartphones, and other portable devices. This project focuses on:

- Predicting **battery health levels** (e.g., Good, Moderate, Poor)
- Using available parameters such as temperature, voltage, charge/discharge cycles, internal resistance, etc.

---

## 🎯 Goals

✅ Preprocess raw battery data  
✅ Train classical ML models to classify battery health  
✅ Evaluate and compare model performance  
✅ Save the best model for real-time prediction

---

## 🧠 Machine Learning Models Used

- Random Forest Classifier
- Support Vector Machine (SVM)
- Decision Tree
- XGBoost (optional)

---

## ⚙️ Tech Stack

| Component     | Tool/Library         |
|---------------|----------------------|
| Language      | Python 3.8+          |
| ML Libraries  | Scikit-learn, XGBoost |
| EDA & Plots   | Pandas, Matplotlib, Seaborn |
| Deployment (optional) | Streamlit / Flask |

---

##########################################

**Workflow**-
- Data Cleaning: Handle missing values, outliers

- EDA: Explore variable relationships

- Feature Engineering: Encoding, scaling, selection

- Model Training: Train and compare multiple models

- Hyperparameter Tuning: GridSearchCV/RandomizedSearch

- Evaluation: On test data

- Model Saving: Using joblib/pickle

- (Optional): Web App using Streamlit

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/battery-health-prediction.git
cd battery-health-prediction




