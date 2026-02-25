# 💳 FICO Score Bucketing & Optimization

A sophisticated credit risk modeling tool designed to segment FICO scores into discrete buckets using advanced statistical optimization techniques. This project has been productionized into a modular Python package, demonstrating expertise in feature engineering, credit risk validation, and professional software development.

## 📌 Project Overview

In the banking and credit industry, raw scores like FICO are often "bucketed" or "binned" into discrete ranges. This process is critical for:
- **Improving model transparency** and stability.
- **Handling non-linear relationships** between predictors and default risk.
- **Creating standardized risk ratings** for regulatory compliance.

This repository implements a robust **FICO Quantizer** that evaluates multiple bucketing strategies (Equal Width, Equal Frequency, and Information Value Optimization) to identify the most predictive and stable segmentation.

---

## ✨ Features

- **Modular Architecture**: core logic extracted into a reusable `fico_bucketing` package.
- **Advanced Optimization**: iterative Information Value (IV) refinement for maximum discriminatory power.
- **Production-Ready**: comprehensive logging, type hinting, and robust error handling for edge cases.
- **Validation Suite**: built-in metrics for Monotonicity, KS Statistic, Gini Coefficient, and Chi-Square significance.
- **Automated Testing**: a full unit test suite ensuring reliability and stability.

---

## 📂 Repository Structure

```text
├── fico_bucketing/         # Core Python package
│   ├── __init__.py         # Package entry point
│   └── processor.py        # FICOQuantizer & StrategyOptimizer classes
├── tests/                  # Unit tests for the package
│   └── test_processor.py   # Test suite for bucketing logic
├── bucketFICOscore.ipynb   # Orchestration notebook & visual analysis
├── requirements.txt        # Project dependencies
└── Task 3 and 4_Loan_Data.csv # Sample credit dataset
```

---

## 📊 Methodology: The FICO Quantizer

### 1. Bucketing Strategies
- **Equal Width**: segments scores into ranges of identical size. Good for uniform distributions.
- **Equal Frequency**: ensures each bucket has an approximately equal number of observations (quantiles).
- **IV Optimization**: a dynamic refinement algorithm that adjusts boundaries to maximize the separation between "Good" and "Bad" customers.

### 2. Statistical Framework (IV & WoE)
The quantizer utilizes **Weight of Evidence (WoE)** and **Information Value (IV)** to measure predictive power:

$$WoE_i = \ln\left(\frac{\% \text{Good}_i}{\% \text{Bad}_i}\right)$$

$$IV = \sum_{i=1}^{n} (\% \text{Good}_i - \% \text{Bad}_i) \cdot WoE_i$$

---

## 🛠️ Performance & Validation Metrics

To ensure the buckets are production-ready, we validate them against:

| Metric | Purpose |
| :--- | :--- |
| **Monotonicity** | Ensures default rates increase/decrease consistently across buckets. |
| **KS Statistic** | Measures the maximum separation between the cumulative distribution of good and bad customers. |
| **Gini Coefficient** | Gauges the discriminatory power (rank-orderability) of the buckets. |
| **Chi-Square ($p$-value)** | Tests the statistical independence between buckets and defaults. |

---

## 🚀 Results & Recommendation

The **Bucketing Strategy Optimizer** evaluated dozens of combinations. Based on the analysis in the orchestration notebook:

> [!IMPORTANT]
> **Recommended Strategy**: Equal Width with 10 Buckets.
> - **Gini Index**: 0.8524 (Excellent discriminatory power).
> - **Monotonicity**: 100% consistent default rate progression.
> - **Population Stability**: Evenly distributed population across the FICO spectrum.

---

## 💻 Installation & Usage

### 1. Setup
Install requirements:
```bash
pip install -r requirements.txt
```

### 2. Basic Usage
```python
from fico_bucketing.processor import FICOQuantizer
import pandas as pd

# Load your data
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')

# Initialize Quantizer
quantizer = FICOQuantizer(df)

# Get optimal buckets (Equal Width, 10 buckets)
results = quantizer.get_optimal_buckets(n_buckets=10, method="equal_width")

# View detailed report
print(results['analysis'])
```

### 3. Running Unit Tests
To verify the installation and logic:
```bash
python -m unittest tests/test_processor.py
```

---

## 👨‍💻 Author
**Arko Saha**  
*Project designed to showcase expertise in Financial Risk Modeling and Production-Scale Data Science.*
