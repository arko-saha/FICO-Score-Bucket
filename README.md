# 💳 FICO Score Bucketing & Optimization

A sophisticated credit risk modeling tool designed to segment FICO scores into discrete buckets using advanced statistical optimization techniques. This project demonstrates expertise in feature engineering, credit risk validation, and mathematical optimization.

## 📌 Project Overview

In the banking and credit industry, raw scores like FICO are often "bucketed" or "binned" into discrete ranges. This process is critical for:
- Improving model transparency and stability.
- Handling non-linear relationships between predictors and default risk.
- Creating standardized risk ratings for regulatory compliance.

This repository implements a robust **FICO Quantizer** that evaluates multiple bucketing strategies (Equal Width, Equal Frequency, and Information Value Optimization) to identify the most predictive and stable segmentation.

---

## 📊 Dataset: Loan Performance Analysis
The project utilizes `Task 3 and 4_Loan_Data.csv`, consisting of 10,000 customer records with the following features:
- **`fico_score`**: The primary predictor for bucketing.
- **`default`**: Binary target (1 = Default, 0 = No Default).
- **Control Features**: `income`, `years_employed`, `credit_lines_outstanding`, `total_debt_outstanding`.

---

## ⚙️ Core Algorithm: The FICO Quantizer

### 1. Bucketing Strategies
- **Equal Width**: Segments scores into ranges of identical size. Good for uniform distributions.
- **Equal Frequency**: Ensures each bucket has an approximately equal number of observations. Useful for handling data density variations.
- **Information Value (IV) Optimization**: A dynamic refinement algorithm that adjust boundaries to maximize the separation between "Good" and "Bad" customers.

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
| **Chi-Square ($p$-value)** | Tests the statistical independence between the buckets and the default outcome. |

---

## 🚀 Results & Recommendation

The **Bucketing Strategy Optimizer** evaluated dozens of combinations. Based on the analysis in `bucketFICOscore.ipynb`:

> [!IMPORTANT]
> **Recommended Strategy**: Equal Width with 10 Buckets.
> - **KS Statistic**: High discriminatory power.
> - **Monotonicity**: 100% consistent default rate progression.
> - **Population Stability**: Evenly distributed across the FICO spectrum.

---

## 💻 How to Use
1. Ensure you have `pandas`, `numpy`, `matplotlib`, and `seaborn` installed.
2. Open `bucketFICOscore.ipynb` in Jupyter or Google Colab.
3. Run the cells to see the full optimization comparison and visual analysis.

---

## 👨‍💻 Author
**Arko Saha**  
*Project designed to showcase expertise in Financial Risk Modeling and Data Science.*
