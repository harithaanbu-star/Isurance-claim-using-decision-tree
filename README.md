# Insurance Claims Fraud Detection
# 🚀 Day 4 — AI Internship Journey

## 📌 Project Overview
This repository contains my Day 4 learning progress during my AI Internship.
I worked on a real-world **Insurance Claims Fraud Detection** project using a dataset
of 1000 insurance claim records with 40 features, covering:
- Data Cleaning & Feature Selection
- Missing Value & Noise Handling
- Label Encoding for Categorical Features
- Outlier Treatment using Winsorization
- Decision Tree Classification Model
- Model Evaluation & Visualization

---

## 🛠 Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Jupyter Notebook

---

## 📊 Topics Practiced

### ✅ Dataset Overview
- Loaded Insurance Claims dataset with 1000 rows and 40 columns
- Explored structure using head, tail, info, and isnull
- Identified data types — 17 integer, 2 float, 21 object columns
- Found one completely empty column — `_c39` (1000 null values)
- Found 91 missing values in `authorities_contacted` column

### ✅ Data Cleaning & Feature Removal

**Dropped completely null column:**
- `_c39` — 100% null, no useful information

**Dropped irrelevant identifier columns:**
- `policy_number` — unique ID, not useful for prediction
- `policy_bind_date` — date field, not directly useful
- `incident_date` — date field, not directly useful
- `incident_location` — high cardinality text field
- `insured_zip` — zip code, not meaningful for model

**Dropped high missing value column:**
- `authorities_contacted` — 91 missing values, dropped entirely

**Final feature selection — kept 17 most relevant columns:**
- `months_as_customer`, `insured_occupation`, `capital-gains`
- `incident_type`, `collision_type`, `incident_severity`
- `incident_hour_of_the_day`, `number_of_vehicles_involved`
- `property_damage`, `bodily_injuries`, `witnesses`
- `police_report_available`, `total_claim_amount`
- `injury_claim`, `property_claim`, `vehicle_claim`
- `fraud_reported` (Target Variable)

### ✅ Noise Value Handling
- Dataset contained `?` values representing unknown or missing entries
- Replaced all `?`, `NaN`, and `nan` string values with proper `np.nan`
- After replacement, identified actual missing values:

| Column | Missing Values |
|--------|---------------|
| collision_type | 178 |
| property_damage | 360 |
| police_report_available | 343 |

### ✅ Label Encoding
Applied Label Encoding to convert all categorical columns to numeric:

| Column | Type |
|--------|------|
| insured_occupation | Object → Integer |
| incident_type | Object → Integer |
| incident_severity | Object → Integer |
| fraud_reported | Object → Integer |
| property_damage | Object → Integer |
| police_report_available | Object → Integer |
| collision_type | Object → Integer |

After encoding, all 17 columns became integer type,
making the dataset fully numeric and ready for ML model

### ✅ Outlier Treatment
- Applied **Winsorization** on `vehicle_claim` column
- Used 5% limits on both lower and upper tails
- Extreme claim values replaced with boundary values
- Visualized the result using a bar chart

### ✅ Visualization Attempts
- Attempted histogram subplots for all features
- Encountered subplot grid size error (17 features exceeded 3x3=9 grid)
- Attempted boxplot subplots for all features
- Encountered subplot grid size error (17 features exceeded 4x4=16 grid)
- Successfully visualized vehicle_claim outlier treatment using bar chart
- Plotted Actual vs Predicted line chart for model results

### ✅ Machine Learning Model — Decision Tree Classifier

**Model Configuration:**
- Algorithm: Decision Tree Classifier
- Criterion: Entropy (Information Gain)
- Train-Test Split: 70% Train / 30% Test
- Random State: 42

**Target Variable:**
- `fraud_reported` — Y (Fraud) / N (Not Fraud)
- Encoded as 1 (Fraud) and 0 (Not Fraud)

**Model Training:**
- Features (X): All 16 columns except fraud_reported
- Target (Y): fraud_reported column
- Decision Tree trained on 700 training samples
- Tree visualized with max_depth=3 and filled color nodes

---

## 📈 Model Performance

### Accuracy Score
- **Overall Accuracy: 73%**

### Confusion Matrix

| | Predicted Not Fraud | Predicted Fraud |
|--|--|--|
| **Actual Not Fraud** | 182 | 38 |
| **Actual Fraud** | 43 | 37 |

### Confusion Matrix Interpretation

| Metric | Value |
|--------|-------|
| True Negatives (TN) | 182 — Correctly predicted Not Fraud |
| False Positives (FP) | 38 — Not Fraud predicted as Fraud |
| False Negatives (FN) | 43 — Fraud predicted as Not Fraud |
| True Positives (TP) | 37 — Correctly predicted Fraud |

### Visualization
- Confusion Matrix plotted as heatmap using Seaborn with Blues colormap
- Actual vs Predicted values plotted as line chart
- Decision Tree structure visualized with figsize 150x100 and max_depth=3

---

## 📁 Dataset Description

| Property | Detail |
|----------|--------|
| Total Records | 1000 rows |
| Original Features | 40 columns |
| Features After Cleaning | 17 columns |
| Target Variable | fraud_reported |
| Target Classes | Y (Fraud) / N (Not Fraud) |
| Numeric Features | 16 after encoding |

**Key Features Used:**

| Feature | Description |
|---------|-------------|
| months_as_customer | How long the customer has been insured |
| insured_occupation | Customer's job type |
| capital-gains | Capital gains amount |
| incident_type | Type of incident — theft, collision, etc |
| collision_type | Side, Front, or Rear collision |
| incident_severity | Minor, Major, or Total Loss |
| incident_hour_of_the_day | Hour when incident occurred |
| number_of_vehicles_involved | Vehicles in the incident |
| property_damage | Whether property was damaged |
| bodily_injuries | Number of bodily injuries |
| witnesses | Number of witnesses present |
| police_report_available | Whether police filed a report |
| total_claim_amount | Total insurance claim amount |
| injury_claim | Claim amount for injuries |
| property_claim | Claim amount for property |
| vehicle_claim | Claim amount for vehicle |

---

## 🔍 Key Observations

| Observation | Detail |
|-------------|--------|
| Dataset Size | 1000 rows × 40 columns |
| After Cleaning | 1000 rows × 17 columns |
| Noise Values | ? replaced with NaN |
| Largest Missing Group | property_damage — 360 missing |
| Outlier Treated | vehicle_claim — Winsorized at 5% |
| Model Used | Decision Tree with Entropy |
| Model Accuracy | 73% |
| Fraud Detection Rate | 37 out of 80 fraud cases detected |
| Errors Learned | Subplot grid must match total feature count |

---

## ⚠️ Errors Encountered & Lessons Learned

**Error 1 — Histogram Subplot Grid Mismatch**
- Used 3x3 grid for 17 features — maximum is 9 plots
- Solution: Increase grid size to match number of features

**Error 2 — Boxplot Subplot Grid Mismatch**
- Used 4x4 grid for 17 features — maximum is 16 plots
- Solution: Use 5x4 or higher grid for 17+ features

**Lesson:** Always calculate rows × cols before setting subplot grid size

---

## 📈 Learning Outcome

Through this project, I learned:
- How to handle real-world messy datasets with noise values like `?`
- How to strategically select features by removing irrelevant columns
- How to replace noise values with proper NaN before imputation
- How to apply Label Encoding to multiple categorical columns at once
- How to treat outliers in claim amount data using Winsorization
- How to build a Decision Tree Classifier with Entropy criterion
- How to evaluate a classification model using Confusion Matrix
- How to interpret True Positive, False Positive, True Negative, False Negative
- How to visualize Decision Tree structure for explainability
- How to match subplot grid size correctly to avoid ValueError

---

## 🎯 Internship Progress

This is **Day 4** of my AI and Machine Learning journey.
I will continue uploading my daily learning progress,
projects, and experiments throughout the internship.

---

## ⭐ Connect With Me

Feel free to explore the repository and follow my learning journey.
