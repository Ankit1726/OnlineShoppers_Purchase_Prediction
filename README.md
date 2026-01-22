# 🛒 Online Shoppers Purchase Prediction

A complete end-to-end **Machine Learning deployment project** that predicts whether an online visitor is likely to make a purchase based on browsing behavior, session metrics, and user attributes.

This project demonstrates **real-world ML engineering practices**, including data preprocessing pipelines, model training, serialization, and deployment using Streamlit.

---

## 🚀 Project Overview

E-commerce platforms generate massive behavioral data from users. Understanding whether a visitor will convert into a customer is crucial for:

- marketing optimization
- user targeting
- recommendation systems
- revenue forecasting

This project uses the **Online Shoppers Purchasing Intention Dataset** to build a predictive model that classifies visitors as:

- ✅ Likely to Purchase
- ❌ Not Likely to Purchase

The trained model is deployed as an interactive web application using **Streamlit**.

---

## 🧠 Machine Learning Workflow

### 1. Data Understanding

Dataset contains information related to:

- page visits
- session durations
- bounce and exit rates
- special day impact
- visitor type
- traffic source
- technical environment

Target variable:

```
Revenue (0 = No Purchase, 1 = Purchase)
```

---

### 2. Data Preprocessing

Implemented using **Scikit-learn Pipeline** and **ColumnTransformer**.

#### Numerical Features

- Administrative
- Administrative_Duration
- Informational
- Informational_Duration
- ProductRelated
- ProductRelated_Duration
- BounceRates
- ExitRates
- PageValues
- SpecialDay
- Weekend

Preprocessing:
- StandardScaler

#### Categorical Features

- OperatingSystems
- Browser
- Region
- TrafficType
- Month
- VisitorType

Preprocessing:
- OneHotEncoder (handle_unknown = "ignore")

---

### 3. Model Training

Algorithms evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Final selected model:

✅ **Decision Tree Classifier**

Reason:
- strong performance
- interpretability
- fast inference

The full preprocessing + model was saved as a **single pipeline**.

---

### 4. Model Serialization

The trained pipeline was stored using pickle:

```python
pickle.dump(model, open("model.pkl", "wb"))
```

This ensures:
- same preprocessing during training and prediction
- no feature mismatch issues
- production-safe deployment

---

### 5. Web Application (Streamlit)

The Streamlit app allows users to:

- input customer behavior metrics
- select technical and visit attributes
- instantly predict purchase intention
- view prediction confidence

Key features:

- interactive UI
- clean CSS styling
- real-time prediction
- probability-based confidence score

---

## 🖥️ Application Demo

### Example High-Purchase Visitor

- High product-related page views
- Long session duration
- Low bounce rate
- Returning visitor
- High page value
- Month: November

Prediction Output:

```
✅ Likely to Purchase
Confidence: 85%+
```

---

## 🧩 Tech Stack

- Python 3.10+
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle

---

## 📁 Project Structure

```
Project/
│
├── app.py                # Streamlit web application
├── model.pkl             # Trained ML pipeline
├── requirements.txt      # Dependencies
├── README.md
```

---

## ▶️ How to Run Locally

### 1. Clone repository

```bash
git clone <your-repo-url>
cd Project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit app

```bash
streamlit run app.py
```

---

## 📊 Model Prediction Output

- **1 → Likely to Purchase**
- **0 → Not Likely to Purchase**

Probability score is also displayed to show confidence in prediction.

---

## 🔐 Production-Level Design Highlights

- End-to-end ML pipeline
- No manual dummy encoding
- ColumnTransformer-based preprocessing
- Safe handling of unseen categories
- Feature mismatch prevention
- Clean separation of training and inference

---

## 🎯 Key Learning Outcomes

- Built production-ready ML pipeline
- Implemented ColumnTransformer and OneHotEncoder
- Prevented common deployment feature mismatch errors
- Deployed ML model using Streamlit
- Designed user-friendly ML web interface
- Applied real-world ML engineering best practices

---

## 📌 Future Enhancements

- Model performance comparison dashboard
- Feature importance visualization
- SHAP explainability
- User session clustering
- Cloud deployment (Streamlit Cloud / AWS)

---
#### Live Demo: https://lnkd.in/gmYXyUHd
---
### 👤 Author
**Ankit Gupta**  
AIML & Data Science Enthusitic
