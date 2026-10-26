# DataCo Supply Chain — Late Delivery Prediction & Shipment Prioritization

## 📌 Project Overview

This project develops a machine learning-based solution to predict late deliveries and prioritize shipments using the **DataCo Supply Chain Dataset**.

The system uses a two-stage approach:

- **Lens 1 — Late Delivery Prediction:** Predicts whether an order is likely to arrive late using a five-model machine learning ensemble.
- **Lens 2 — Shipment Prioritization:** Combines predicted late-delivery risk with order monetary value to classify shipments into operational priority levels.

The main objective is to help supply chain teams identify high-risk deliveries, prioritize valuable orders, and support proactive decision-making.

## 🎯 Project Objectives

- Analyze supply chain data to identify patterns associated with late deliveries.
- Prepare data and select relevant features for machine learning.
- Develop and evaluate multiple classification algorithms.
- Improve prediction robustness through ensemble learning.
- Estimate late-delivery risk at the order level.
- Prioritize shipments based on delivery risk and monetary value.
- Provide actionable insights for supply chain management.

## 📊 Dataset Description

The project uses the DataCo Supply Chain Dataset, containing information about orders, customers, products, shipping methods, sales, and delivery performance.

| Dataset Information | Value |
|---|---|
| Total Order-Item Records | 180,519 |
| Unique Orders | 65,752 |
| Unique Products | 118 |
| Unique Customers | 20,652 |
| Target Variable | `Late_delivery_risk` |

### Target Variable Distribution

| Class | Description | Records |
|---|---|---:|
| 0 | No Late Delivery Risk | 81,542 |
| 1 | Late Delivery Risk | 98,977 |

The analysis uses the following files:

- `DataCoSupplyChainDataset.csv`
- `DescriptionDataCoSupplyChain.csv`
- `tokenized_access_logs.csv`

## 🔄 Project Methodology

The project follows an end-to-end machine learning workflow.

### 1. Exploratory Data Analysis (EDA)

- Examined dataset structure and data types.
- Analyzed missing values and duplicate records.
- Studied late-delivery risk distribution.
- Investigated shipping modes, regions, and product categories.
- Explored relationships between supply chain features and delivery performance.

### 2. Data Preprocessing and Feature Engineering

- Prepared numerical and categorical variables.
- Encoded categorical features.
- Performed feature selection.
- Prepared training and testing data for classification.
- Selected 100 encoded features for the final ensemble.

### 3. Machine Learning Model Development

Five machine learning algorithms were used in the final ensemble:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. LightGBM
5. Linear Support Vector Machine (SVM)

The final approach uses an **equal-weight soft-voting ensemble**, where each model contributes 20% to the predicted probability.

### 4. Late Delivery Prediction — Lens 1

The ensemble predicts the probability of late delivery for each order line.

A classification threshold of **0.50** is used to convert predicted probabilities into binary predictions.

### 5. Shipment Prioritization — Lens 2

Order-line predictions are aggregated by `Order Id` to produce an order-level risk score.

Each order is evaluated using:

- Predicted late-delivery probability
- Total order monetary value
- Risk category
- Monetary impact category

The combined assessment produces three operational priorities: Low, Medium, and Urgent.

## 🤖 Machine Learning Results

### Final Model Performance

| Evaluation Metric | Order-Line Level | Order Level |
|---|---:|---:|
| Accuracy | 71.69% | 72.00% |
| Precision | 84.81% | 85.25% |
| Recall | 58.95% | 59.40% |
| F1-Score | 69.56% | 70.01% |
| ROC-AUC | 0.7675 | 0.7734 |
| Average Precision | 0.8303 | 0.8344 |
| Brier Score | 0.1854 | 0.1838 |
| Log Loss | 0.5407 | 0.5371 |

### Key Findings

- The ensemble achieved **71.69% accuracy** at the order-line level.
- Order-level prediction accuracy reached **72.00%**.
- Order-level precision reached **85.25%**, indicating relatively reliable positive predictions.
- Recall remained moderate, meaning some late deliveries were not identified.

## 🚦 Shipment Risk Classification

Predicted delivery probabilities are divided into three risk categories.

| Risk Level | Predicted Probability |
|---|---|
| Low Risk | Below 0.30 |
| Medium Risk | 0.30 – below 0.65 |
| High Risk | 0.65 and above |

### Observed Risk Classification Results

| Risk Category | Orders | Actual Late Delivery Rate |
|---|---:|---:|
| Low | 228 | 9.65% |
| Medium | 2,816 | 38.42% |
| High | 1,888 | 85.28% |

The high-risk category demonstrated a substantially higher observed late-delivery rate.

## 💰 Business Impact Analysis

The project uses total order value as a proxy for monetary exposure.

| Monetary Impact | Order Value |
|---|---|
| Low | Below 314.55 |
| Medium | 314.55 – below 625.71 |
| High | 625.71 and above |

These thresholds were derived from training-order values. The dataset analysis does not establish a currency for these values.

### Priority Assignment Matrix

| Delivery Risk / Monetary Impact | Low | Medium | High |
|---|---|---|---|
| Low Risk | Low | Low | Medium |
| Medium Risk | Low | Medium | Medium |
| High Risk | Medium | Medium | Urgent |

### Recommended Operational Actions

| Priority Level | Recommended Action |
|---|---|
| Low | Routine shipment tracking |
| Medium | Enhanced monitoring and review |
| Urgent | Immediate operational escalation |

## 📈 Shipment Prioritization Results

| Metric | Low Priority | Medium Priority | Urgent Priority |
|---|---:|---:|---:|
| Total Orders | 1,114 | 3,243 | 575 |
| Actually Late Orders | 380 | 1,852 | 482 |
| Late Delivery Rate | 34.11% | 57.11% | 83.83% |
| Share of Operational Workload | 22.59% | 65.75% | 11.66% |
| Share of Total Order Value | 8.67% | 70.70% | 20.63% |

### Business Insights

- Only **11.66%** of evaluated orders were classified as Urgent.
- Urgent orders accounted for **20.63%** of total order value.
- **83.83%** of Urgent orders were actually delivered late.
- The Medium priority category contained the largest number of late orders.

These results demonstrate how predictive analytics can support operational prioritization. They do not establish that the proposed interventions have already reduced delivery delays.

## 🛠️ Technologies Used

**Programming Language**
- Python

**Development Environment**
- Jupyter Notebook

**Data Analysis and Visualization**
- pandas
- NumPy
- Matplotlib
- Seaborn

**Machine Learning**
- scikit-learn
- LightGBM
- SciPy

## ⚙️ Installation and Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

### Step 2 — Install Required Libraries

```bash
pip install notebook pandas numpy scipy matplotlib seaborn scikit-learn lightgbm
```

### Step 3 — Prepare the Dataset

Place the required DataCo CSV files in the working directory expected by the notebook.

### Step 4 — Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
04_final_feature selection (19).ipynb
```

Execute the notebook cells in order.

**Note:** The notebook is a research prototype. Some local file paths and intermediate variables may require adjustments before execution in a different environment.

## 📁 Suggested Project Structure

```text
DataCo-Supply-Chain-Analytics/
│
├── README.md
│
├── 04_final_feature selection (19).ipynb
│
├── data/
│   └── Dataset files
│
├── results/
│   └── Generated analysis outputs
│
└── .gitignore
```

This is a suggested repository structure rather than a verified existing directory structure.

## ⚠️ Limitations

- The model has moderate recall and may miss some late deliveries.
- Shipping mode influences prediction performance.
- Order-level probabilities are calculated by averaging order-line probabilities.
- Priority thresholds have not been optimized using actual intervention costs.
- Order monetary value does not represent the actual financial loss caused by a late delivery.
- The solution has not been validated in a real-time production environment.

## 🚀 Future Improvements

- Improve recall through classification threshold optimization.
- Explore additional feature engineering and feature selection techniques.
- Investigate performance across different shipping modes and regions.
- Improve order-level probability calibration.
- Integrate operational costs into shipment prioritization.
- Develop an interactive supply chain monitoring dashboard.
- Create a reproducible training and prediction pipeline.
- Evaluate the solution using new operational data.

## 🏁 Conclusion

This project demonstrates the application of machine learning and business analytics to supply chain delivery management.

The proposed approach combines a five-model ensemble for late-delivery prediction with a risk-based shipment prioritization framework.

The final model achieved **71.69% order-line accuracy** and **85.25% order-level precision**. The prioritization framework identified a small group of urgent orders with high observed late-delivery rates and substantial monetary exposure.

Overall, the project provides a proof of concept for using predictive analytics to support more informed and proactive supply chain decision-making.

---

**Project Status:** Research Prototype / Proof of Concept

**Dataset:** DataCo Supply Chain Dataset

**Domain:** Machine Learning | Predictive Analytics | Supply Chain Management
