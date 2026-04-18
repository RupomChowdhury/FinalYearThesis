# FinalYearThesis

# 🧠 Mental Fatigue Detection using EEG Signals

## 📌 Overview

This project focuses on **detecting mental fatigue from EEG signals** using a regression-based machine learning pipeline.

Instead of treating fatigue as a binary classification problem, this approach models it as a **continuous score**, enabling more fine-grained analysis of mental states.

---

## 📂 Dataset

* EEG data stored as multiple CSV files
* Each file contains EEG signal recordings
* All files are merged into a single dataset for processing

---

## ⚙️ Methodology

### 1. Data Loading

* CSV files are loaded from Google Drive
* All data is concatenated into one DataFrame

---

### 2. Feature Engineering

From raw EEG signals, statistical features are extracted:

* Mean
* Standard Deviation
* Variance
* Minimum / Maximum
* Range
* Energy (signal power)
* Activity Ratio
* Entropy Proxy

These features summarize EEG signal behavior efficiently.

---

### 3. Fatigue Score (Regression Target)

A continuous fatigue score (`y_reg`) is constructed using:

* Signal Energy
* Signal Variability (Standard Deviation)

#### Steps:

1. Compute energy and standard deviation from raw signals
2. Calculate ratio:

   ```
   fatigue_score = energy / std
   ```
3. Clip extreme values (1st–99th percentile)
4. Normalize to range `[0, 1]`

This produces a **robust and normalized fatigue metric**.

---

### 4. Preprocessing

* Missing values handled using **mean imputation**
* Features scaled using **standardization** (mean = 0, std = 1)

---

### 5. Feature Selection

* Method: Mutual Information (regression-based)
* Top features selected

---

## 🤖 Regression Models

The following models are trained and evaluated:

* **Random Forest Regressor**
* **Extra Trees Regressor**
* **XGBoost Regressor**
* **Support Vector Regressor (RBF kernel)**

---

## 📊 Evaluation Metrics

Model performance is evaluated using:

* **MAE (Mean Absolute Error)**  
  → Average prediction error  
  * `0.0` → Perfect prediction  
  * `< 0.05` → Excellent  
  * `0.05 – 0.10` → Good  
  * `> 0.10` → Needs improvement  

* **RMSE (Root Mean Squared Error)**  
  → Penalizes larger errors more heavily  
  * `0.0` → Perfect prediction  
  * `< 0.05` → Excellent (very low variance in errors)  
  * `0.05 – 0.10` → Good  
  * `> 0.10` → High error variance  

* **R² Score (Coefficient of Determination)**  
  → Measures how well the model explains variance  

  * `1.0` → Perfect prediction  
  * `0.90 – 1.0` → Excellent  
  * `0.75 – 0.90` → Good  
  * `0.50 – 0.75` → Moderate  
  * `0.0 – 0.50` → Weak  
  * `< 0` → Worse than baseline  
---

## 📈 Visualizations

### 1. Fatigue Score Distribution

* Histogram showing the distribution of normalized fatigue values

### 2. Prediction vs True Values

* Scatter plots comparing predicted vs actual fatigue scores
* Ideal alignment follows a diagonal line

### 3. Error Distribution

* Histogram of residuals (prediction errors)
* Helps analyze model bias and variance

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone <your-repo-link>
cd <repo-name>
```

### 2. Open in Google Colab

### 3. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Set Dataset Path

```python
folder_path = '/content/drive/MyDrive/EEG_Thesis/'
```

### 5. Run All Cells

---

## 🧪 Output

Each model produces results in the format:

```
Model Name
MAE  : 0.xxxxx
RMSE : 0.xxxxx
R²   : 0.xxxxx
```

---

## 🛠 Tech Stack

* Python
* NumPy, Pandas
* Scikit-learn
* XGBoost
* Matplotlib, Seaborn
* Google Colab
* Visual Studio Code

---
