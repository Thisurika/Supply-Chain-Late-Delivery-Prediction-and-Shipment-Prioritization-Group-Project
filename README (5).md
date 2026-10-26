# DataCo Supply Chain: Late Delivery Prediction & Shipment Prioritization

An end-to-end machine learning **proof of concept** for predicting late deliveries and prioritizing orders for operational review. The project analyzes the DataCo Supply Chain dataset, trains and compares classification models, and connects predicted late-delivery risk with order monetary value to create actionable priority tiers.

## Project overview

The solution contains two connected components:

- **Lens 1 — Late-delivery prediction:** Predicts the probability of late delivery for each order line using an equal-weight, five-model soft-voting ensemble.
- **Lens 2 — Business prioritization:** Aggregates order-line probabilities to unique orders, combines risk with total order value, and assigns **Low**, **Medium**, or **Urgent** operational priorities.

The final approach documented in the notebook is **Experiment 2**.

## Dataset

The analysis reads these local CSV files:

| File | Purpose |
|---|---|
| `DataCoSupplyChainDataset.csv` | Main order, shipment, product, customer, and sales data |
| `DescriptionDataCoSupplyChain.csv` | Dataset field descriptions |
| `tokenized_access_logs.csv` | Tokenized access-log data |

The main dataset contains **180,519 order-item records**, **65,752 unique orders**, **118 products**, and **20,652 customers**. The prediction target is `Late_delivery_risk`:

| Class | Meaning | Records | Share |
|---|---|---:|---:|
| `0` | No late-delivery risk | 81,542 | 45.17% |
| `1` | Late-delivery risk | 98,977 | 54.83% |

The main dataset is not included in this README. Obtain the relevant DataCo CSV files separately and follow their applicable usage terms.

## Workflow

1. **Data exploration and auditing:** Inspect types, missing values, duplicates, target distribution, and unique identifiers; examine delivery risk across shipping modes, regions, and categories.
2. **Preparation and feature engineering:** Prepare order data, encode predictors, and perform feature selection. The final ensemble uses **100 encoded features**.
3. **Model development:** Tune and evaluate individual classifiers, then combine five models using soft voting.
4. **Lens 1 evaluation:** Evaluate order-line predictions with classification and probability-quality metrics.
5. **Lens 2 aggregation:** Average order-line risk probabilities by `Order Id` to obtain one risk score per unique order.
6. **Business prioritization:** Combine predicted risk bands with training-derived order-value bands and assign operational actions.
7. **Final evaluation:** Assess both prediction levels and priority-tier workload, late-order capture, and monetary-value coverage.

## Models and configuration

The final **five-model soft-voting ensemble** combines:

- Logistic Regression
- Decision Tree
- Random Forest
- LightGBM
- Linear Support Vector Machine (SVM), calibrated to output probabilities

Each model contributes **20%** to the final probability:

\[
P(\text{late}) = \frac{1}{5}\sum_{i=1}^{5} P_i(\text{late})
\]

The final binary classification threshold is **0.50**. This threshold prioritizes comparatively high precision over maximizing recall.

## Final test results

### Lens 1 vs. Lens 2

| Metric | Order-line predictions (Lens 1) | Unique-order predictions (Lens 2) |
|---|---:|---:|
| Test records | 13,378 | 4,932 |
| Accuracy | **71.69%** | **72.00%** |
| Precision | **84.81%** | **85.25%** |
| Recall | 58.95% | 59.40% |
| F1-score | 69.56% | 70.01% |
| ROC-AUC | 0.7675 | 0.7734 |
| Average precision | 0.8303 | 0.8344 |
| Brier score | 0.1854 | 0.1838 |
| Log loss | 0.5407 | 0.5371 |

**Order-line confusion matrix** (test set):

| Actual / Predicted | On time | Late |
|---|---:|---:|
| On time | 5,265 | 775 |
| Late | 3,012 | 4,326 |

**Unique-order confusion matrix** (test set):

| Actual / Predicted | On time | Late |
|---|---:|---:|
| On time | 1,939 | 279 |
| Late | 1,102 | 1,612 |

The ensemble is relatively precise when it flags a late delivery, but its recall is moderate: approximately **41% of late deliveries are missed** at the selected threshold.

## Lens 2: Risk profiling

Order-line probabilities are averaged within each unique order. The resulting order-level score is grouped as follows:

| Risk band | Predicted late-delivery probability |
|---|---|
| Low | Below 0.30 |
| Medium | 0.30 to below 0.65 |
| High | 0.65 and above |

### Observed test performance by risk band

| Risk band | Orders | Actually late | Observed late rate | Share of orders | Share of all late orders captured |
|---|---:|---:|---:|---:|---:|
| Low | 228 | 22 | 9.65% | 4.62% | 0.81% |
| Medium | 2,816 | 1,082 | 38.42% | 57.10% | 39.87% |
| High | 1,888 | 1,610 | **85.28%** | 38.28% | 59.32% |

The risk bands separate orders into groups with substantially different observed late-delivery rates.

## Business impact and operational prioritization

Total order value serves as a proxy for **monetary exposure**, not the actual cost of a delay. Value boundaries were derived from training-order values:

| Impact band | Total order value |
|---|---|
| Low | Below 314.55 |
| Medium | 314.55 to below 625.71 |
| High | 625.71 and above |

The notebook does not establish a currency for these thresholds, so values are shown without a currency symbol.

### Priority assignment matrix

| Delivery risk \ Monetary impact | Low | Medium | High |
|---|---|---|---|
| Low | Low | Low | Medium |
| Medium | Low | Medium | Medium |
| High | Medium | Medium | **Urgent** |

| Priority | Recommended action |
|---|---|
| Low | Routine tracking |
| Medium | Enhanced monitoring and review |
| Urgent | Immediate operational escalation |

### Priority-tier performance on the test set

| Metric | Low | Medium | Urgent |
|---|---:|---:|---:|
| Orders | 1,114 | 3,243 | 575 |
| Actually late orders | 380 | 1,852 | 482 |
| Observed late rate | 34.11% | 57.11% | **83.83%** |
| Share of operational workload | 22.59% | 65.75% | **11.66%** |
| Share of late orders captured | 14.00% | 68.24% | 17.76% |
| Share of total order value | 8.67% | 70.70% | **20.63%** |

**Business takeaway:** The **575 Urgent orders** represent **11.66%** of the evaluated orders but account for **20.63%** of their total value. **83.83%** of those urgent orders were actually late. However, the Medium tier still contains most of the late orders, so it cannot be ignored.

## Tools and libraries

- Python and Jupyter Notebook
- pandas and NumPy for data processing
- Matplotlib and Seaborn for visualization
- scikit-learn for preprocessing, model selection, training, calibration, and evaluation
- LightGBM for gradient-boosted classification
- SciPy for numerical and sparse-matrix operations

## Getting started

1. Clone your repository and enter its directory:

   ```bash
   git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPOSITORY>.git
   cd <YOUR-REPOSITORY>
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. Install the core packages:

   ```bash
   pip install notebook pandas numpy scipy matplotlib seaborn scikit-learn lightgbm
   ```

4. Place the three dataset CSVs listed above in the notebook's working directory. Create any expected output directories (such as `data/` and `results/`) before running cells that save files.

5. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

6. Open **`04_final_feature selection (19).ipynb`** and run its cells in order.

> **Reproducibility note:** This is a research notebook, not a packaged application. Its cells depend on intermediate variables created earlier in the notebook. Package versions, input data, execution order, and local paths may need adjustment. The installation command above is a starting point, not a verified lockfile. The notebook was analyzed, not re-executed as part of preparing this README.

## Suggested repository layout

```text
.
├── README.md
├── 04_final_feature selection (19).ipynb
├── data/                     # Input/audit files as appropriate
├── results/                  # Generated tables and analysis outputs
└── .gitignore
```

The layout above is a suggested organization, **not** a claim that these folders and files have already been committed. Avoid committing sensitive access logs or other restricted data.

## Limitations

- **Moderate recall:** Many actual late deliveries are not detected at the 0.50 classification threshold.
- **Shipping-mode dependence:** Feature analysis identifies Shipping Mode as influential; earlier validation error analysis found weak detection for Standard Class shipments.
- **Order-level aggregation:** Averaging order-line probabilities is not the same as fitting and calibrating a dedicated order-level model.
- **Business rules:** Priority bands have not been optimized against real intervention costs or proven to reduce delays.
- **Value proxy:** Order value is not the financial loss attributable to late delivery.
- **Generalization:** Similar validation and test scores do not alone rule out data leakage, distribution shift, or performance degradation in live operations.

## Future improvements

- Explore recall-focused thresholds and precision–recall trade-offs.
- Audit feature-selection and preprocessing steps for leakage and assess time-based generalization.
- Investigate model performance within each shipping mode and region.
- Calibrate and validate order-level risk estimates separately.
- Incorporate intervention cost, potential loss, and operational capacity into priority rules.
- Refactor the notebook into reproducible training/inference pipelines with pinned dependencies and automated tests.
- Validate performance prospectively before any operational deployment.

## Project status

**Research prototype / proof of concept.** The notebook demonstrates offline predictive and business-prioritization performance; it does not establish a deployed, real-time production system.

---

*Metrics and project details are based on the final project summary and code in the supplied Jupyter notebook.*
