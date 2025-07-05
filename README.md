# 💧 Water Quality Classification Project

## 📌 Overview

This project implements a machine learning pipeline to classify water quality based on chemical and physical properties. It uses a dataset containing features like **Chloride**, **Organic Carbon**, **Solids**, **Sulphate**, **Turbidity**, and **pH** to predict a **binary label** indicating water quality.

It trains and evaluates four models:
- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  

Evaluation is based on metrics such as **accuracy**, **precision**, **recall**, **F1-score**, **sensitivity**, and **specificity**.

---

## 🛠️ Features

- **Data Preprocessing**:  
  - Handles missing values  
  - Scales features using `StandardScaler`

- **Model Training**:  
  - Trains Logistic Regression, Random Forest, SVM, and KNN

- **Model Evaluation**:  
  - Calculates and compares performance metrics  
  - Visualizes confusion matrices and metric bar plots

- **Feature Importance**:  
  - Analyzes features using the Random Forest model

- **Model Persistence**:  
  - Saves the best model and scaler using `joblib`

---

## 📁 Installation

### 1. Clone the Repository

git clone https://github.com/TheFallenGuru/Water-Prediction
cd water-quality-classification

### 2. Install Requirements

Make sure you have **Python 3.8+** installed.

Install the dependencies:

pip install -r requirements.txt

#### `requirements.txt` Includes:

- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  
- joblib

---

## 📊 Dataset

- **File Name**: `Data.csv`  
- **Features**:
  - `Chloride`
  - `Organic_Carbon`
  - `Solids`
  - `Sulphate`
  - `Turbidity`
  - `pH`

- **Target Column**: `Label` (0 or 1)  
- Place the `Data.csv` file in the **project root directory**.  
- Missing values are **dropped** during preprocessing.

---

## 🚀 Usage

### Run the main script:

python water_quality_classification.py

The script will:
- Load and preprocess the dataset  
- Train and evaluate four ML models  
- Print performance metrics to the console  
- Save the following files:
  - `confusion_matrices.png`
  - `model_comparison.png`
  - `feature_importance.png`
  - `scaler.pkl`
  - `best_water_quality_model.pkl`

---

## 📂 Output Files

| File                          | Description                                     |
|-------------------------------|-------------------------------------------------|
| `confusion_matrices.png`      | Confusion matrices for all models               |
| `model_comparison.png`        | Bar plot comparing model performance metrics    |
| `feature_importance.png`      | Feature importance plot for Random Forest       |
| `scaler.pkl`                  | Saved StandardScaler object                     |
| `best_water_quality_model.pkl`| Best model saved based on F1-score              |

---

## 📈 Model Details

| Model              | Parameters                                          |
|-------------------|------------------------------------------------------|
| Logistic Regression | `random_state=42`                                 |
| Random Forest      | `n_estimators=100, random_state=42`               |
| SVM                | `kernel='rbf', probability=True, random_state=42`  |
| KNN                | `n_neighbors=5`                                    |

### 📌 Metrics Evaluated

- Precision  
- Recall (Sensitivity)  
- Accuracy  
- F1-Score  
- Specificity  
- Confusion Matrix

---

## 🏆 Results

The script outputs:

- **Performance metrics** for each model  
- **Visualizations**:
  - Confusion matrices (2x2 grid)  
  - Metric comparison (bar chart)  
  - Feature importance (Random Forest)

The **best-performing model** (based on F1-score) is saved for future use.




## 📧 Contact

For questions, issues, or suggestions:  
Open an issue on GitHub or contact:  
📩 archit.23bec10003@vitbhopal.ac.in

