# 🛒 E-Commerce Purchase Prediction

> Predict whether a customer will make a purchase on an e-commerce website using Machine Learning models, with 89% accuracy achieved via Random Forest.

---

## 📌 Overview

With the shift from physical stores to online shopping, understanding customer behaviour has become critical for improving conversions.  
This project builds and compares multiple ML models to predict purchase intent using clickstream and customer session data.  

By focusing on the most relevant features from user browsing behaviour, the model achieves high accuracy without relying on personal information — addressing privacy concerns while maintaining performance.

---

## 🎯 Objectives

- Classify website visitors as **Buy** or **Not Buy**.
- Compare the performance of multiple ML models on the same dataset.
- Identify which features and algorithms yield the best predictive accuracy.
- Deploy the best model in a **Streamlit web app** for real-time prediction.

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository – Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)
- **Instances:** 12,330 sessions (each session belongs to a unique user over a year)
- **Attributes:** 18 total (10 numerical, 8 categorical)
- **Target Variable:** `Revenue` (1 if purchase made, 0 otherwise)
- **No missing values**

Key Features include:
- `PageValues` – Average value of a page viewed before a transaction
- `ExitRates` – % of users who exit after visiting a specific page
- `ProductRelated` & `ProductRelated_Duration` – Number and time spent on product pages
- `BounceRates`, `Informational`, `Month`, `VisitorType`, `Weekend`, etc.

---

## 🛠 Methodology

1. **Exploratory Data Analysis (EDA)** – Check distributions, detect outliers, understand feature relationships.
2. **Data Preprocessing**
   - Convert categorical to numeric
   - Replace irrelevant values (e.g., `'Other'` → `'Returning_Visitor'`)
   - Drop non-numeric columns (`Month`)
3. **Feature Selection**
   - Used `ExtraTreesClassifier` to rank feature importance
   - Selected top 4 features: `PageValues`, `ExitRates`, `ProductRelated_Duration`, `ProductRelated`
4. **Feature Scaling** – Standardized selected features.
5. **Model Building**
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Naïve Bayes
   - Decision Tree
   - Random Forest
6. **Model Evaluation**
   - Metrics: Accuracy, Classification Report
   - Best Model: **Random Forest** (Accuracy: 89.2% after tuning)
7. **Hyperparameter Tuning**
   - Used GridSearchCV to optimize `max_depth`, `min_samples_leaf`, `min_samples_split`, `n_estimators`
8. **Deployment**
   - Streamlit app with sliders for inputting feature values
   - Real-time predictions using the trained model

---

## 📈 Results

| Model                   | Accuracy   |
|-------------------------|------------|
| Logistic Regression     | 86.9%      |
| Decision Tree           | 85.1%      |
| Random Forest           | **89.2%**  |
| KNN                     | 87.6%      |
| Naïve Bayes             | 86.6%      |

- Random Forest provided the **highest accuracy and robustness**.
- Only session-based behavioural data was needed for strong predictions.

---

## 🚀 Deployment

**Streamlit App:** [Click to View](https://ridit07-ml-project-app-f3zc6r.streamlit.app/)  
**GitHub Code:** [Repository Link](https://github.com/Ridit07/ml-project)

---

## 📂 Project Structure

e-commerce-purchase-prediction/
├── app.py # Streamlit app for predictions
├── project.pkl # Trained Random Forest model
├── ML_Notebook.ipynb # Full EDA + model building notebook
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## ⚙️ Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Ridit07/ml-project.git
cd ml-project
```

2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

3️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

## 📌 How to Use the App

1. Use sliders to input:
   - `PageValues`
   - `ExitRates`
   - `ProductRelated_Duration`
   - `ProductRelated`
2. Click **Predict**
3. The app will output whether the customer is **likely to purchase**.

---

## 🧠 Key Insights

- Session clickstream data is **more predictive** than static customer data.
- `PageValues` was the **most important feature** for prediction.
- Combining behavioural features yields better accuracy **without breaching privacy**.
