# Commercial & Retail Credit Risk Modeling: Probability of Default (PD) & Scorecard Engine

## Overview
This repository implements an end-to-end **Credit Risk Appraisal and Probability of Default (PD) Modeling Pipeline** in Python. Designed around regulatory banking frameworks (Basel III / IFRS 9) and credit bureau scoring architectures (300–900 scale), this project bridges econometric interpretability with machine learning performance.

## Key Features
- **Synthetic Bureau & Financial Data Generator:** Simulates realistic borrower profiles including Debt-to-Income (DTI) ratios, credit utilization, delinquency history, and macroeconomic stress indicators.
- **Econometric Feature Screening:** Implements **Weight of Evidence (WoE)** monotonic binning and **Information Value (IV)** calculation to filter predictive credit variables.
- **Dual-Model Architecture:** 
  - **Baseline Econometric Logit:** For regulatory transparency and odds-ratio interpretation.
  - **Gradient Boosting Classifier:** Captures non-linear interactions across borrower financial ratios.
- **Banking Evaluation Metrics:** Evaluates models using **ROC-AUC**, **Gini Coefficient**, and the **Kolmogorov-Smirnov (KS) Statistic**—the industry standard for separating defaulting vs. non-defaulting populations.
- **Scorecard Scaling (300–900):** Converts raw default probabilities into an intuitive, CIBIL-style credit score using Points-to-Double-the-Odds (PDO) log-odds scaling.

## Project Structure
```text
├── credit_risk_pipeline.py   # End-to-end data generation, WoE/IV, modeling & scorecard script
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation