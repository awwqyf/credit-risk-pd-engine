"""
Credit Risk Probability of Default (PD) & Scorecard Engine
Author: Aaqif Hussain Mattoo
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)


def generate_credit_bureau_data(n_samples: int = 10000) -> pd.DataFrame:
    age = np.random.randint(21, 65, size=n_samples)
    annual_income_inr = np.random.lognormal(mean=13.3, sigma=0.6, size=n_samples)
    credit_utilization = np.random.beta(a=2, b=5, size=n_samples)
    dti_ratio = np.clip(np.random.normal(loc=0.35, scale=0.15, size=n_samples), 0.05, 0.95)
    past_delinquencies_24m = np.random.poisson(lam=0.4, size=n_samples)
    credit_vintage_months = np.random.randint(6, 180, size=n_samples)
    loan_amount_inr = annual_income_inr * np.random.uniform(0.2, 1.8, size=n_samples)

    logit_latent = (
        -3.4
        + 2.6 * credit_utilization
        + 3.1 * dti_ratio
        + 0.75 * past_delinquencies_24m
        - 0.012 * (age - 21)
        - 0.008 * credit_vintage_months
        + 0.45 * (loan_amount_inr / annual_income_inr)
        + np.random.normal(0, 0.5, size=n_samples)
    )

    prob_default = 1 / (1 + np.exp(-logit_latent))
    default_flag = (np.random.uniform(0, 1, size=n_samples) < prob_default).astype(int)

    return pd.DataFrame({
        "borrower_age": age,
        "annual_income_inr": np.round(annual_income_inr, 2),
        "loan_amount_inr": np.round(loan_amount_inr, 2),
        "dti_ratio": np.round(dti_ratio, 4),
        "credit_utilization": np.round(credit_utilization, 4),
        "past_delinquencies_24m": past_delinquencies_24m,
        "credit_vintage_months": credit_vintage_months,
        "default_12m": default_flag
    })


def calculate_woe_iv(df: pd.DataFrame, feature: str, target: str, bins: int = 5) -> float:
    temp = pd.DataFrame({"feature": df[feature], "target": df[target]})
    if temp["feature"].nunique() > bins:
        temp["bin"] = pd.qcut(temp["feature"], q=bins, duplicates="drop")
    else:
        temp["bin"] = temp["feature"]

    grouped = temp.groupby("bin", observed=False)["target"].agg(["count", "sum"])
    grouped.columns = ["total", "bads"]
    grouped["goods"] = grouped["total"] - grouped["bads"]

    dist_good = (grouped["goods"] + 0.5) / (grouped["goods"].sum() + 0.5)
    dist_bad = (grouped["bads"] + 0.5) / (grouped["bads"].sum() + 0.5)

    grouped["woe"] = np.log(dist_good / dist_bad)
    grouped["iv"] = (dist_good - dist_bad) * grouped["woe"]
    return float(grouped["iv"].sum())


def compute_banking_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1
    ks_stat, _ = ks_2samp(y_prob[y_true == 0], y_prob[y_true == 1])
    return {"ROC_AUC": auc, "Gini": gini, "KS_Statistic": ks_stat}


def scale_to_credit_score(pd_prob: np.ndarray, base_score: int = 600, base_odds: float = 19.0, pdo: int = 50) -> np.ndarray:
    pd_clipped = np.clip(pd_prob, 1e-6, 1 - 1e-6)
    odds = (1 - pd_clipped) / pd_clipped
    factor = pdo / np.log(2)
    offset = base_score - (factor * np.log(base_odds))
    raw_score = offset + factor * np.log(odds)
    return np.clip(np.round(raw_score), 300, 900).astype(int)


def plot_diagnostics(y_test: np.ndarray, logit_probs: np.ndarray, gbm_probs: np.ndarray, scores: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    fpr_l, tpr_l, _ = roc_curve(y_test, logit_probs)
    fpr_g, tpr_g, _ = roc_curve(y_test, gbm_probs)
    axes[0].plot(fpr_l, tpr_l, label=f"Econometric Logit (AUC = {roc_auc_score(y_test, logit_probs):.3f})", lw=2)
    axes[0].plot(fpr_g, tpr_g, label=f"Gradient Boosting (AUC = {roc_auc_score(y_test, gbm_probs):.3f})", lw=2, linestyle="--")
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle=":")
    axes[0].set_title("ROC Curve: PD Model Discrimination")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Sensitivity)")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    score_df = pd.DataFrame({
        "Credit_Score": scores,
        "Default_Status": np.where(y_test == 1, "Default (Bad)", "Performing (Good)")
    })
    sns.kdeplot(
        data=score_df, x="Credit_Score", hue="Default_Status",
        fill=True, common_norm=False, palette=["#2b5c8f", "#d95f02"], ax=axes[1]
    )
    axes[1].set_title("Calibrated Credit Score Distribution (300 - 900 Scale)")
    axes[1].set_xlabel("Credit Bureau Score")
    axes[1].set_ylabel("Density")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credit_risk_diagnostics.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n[+] SUCCESS! Saved diagnostic chart to: {output_path}")


def main():
    print("=" * 72)
    print("COMMERCIAL & RETAIL CREDIT RISK PD ENGINE")
    print("=" * 72)

    df = generate_credit_bureau_data(n_samples=10000)
    print(f"[+] Generated {len(df):,} borrower profiles | Baseline Default Rate: {df['default_12m'].mean():.2%}\n")

    features = [c for c in df.columns if c != "default_12m"]
    print("--- Feature Information Value (IV) Screening ---")
    iv_records = []
    for feat in features:
        iv_val = calculate_woe_iv(df, feat, "default_12m")
        strength = "Strong" if iv_val >= 0.3 else ("Medium" if iv_val >= 0.1 else "Weak")
        iv_records.append({"Feature": feat, "IV": round(iv_val, 4), "Predictive_Power": strength})
    print(pd.DataFrame(iv_records).sort_values(by="IV", ascending=False).to_string(index=False))

    X = df[features]
    y = df["default_12m"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logit_model = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)
    logit_probs = logit_model.predict_proba(X_test_scaled)[:, 1]

    gbm_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42).fit(X_train, y_train)
    gbm_probs = gbm_model.predict_proba(X_test)[:, 1]

    print("\n--- Out-of-Sample Model Validation (Test Set) ---")
    eval_df = pd.DataFrame(
        [compute_banking_metrics(y_test, logit_probs), compute_banking_metrics(y_test, gbm_probs)],
        index=["Econometric Logit", "Gradient Boosting"]
    ).round(4)
    print(eval_df)

    credit_scores = scale_to_credit_score(logit_probs)
    results_df = X_test.copy()
    results_df["Actual_Default"] = y_test
    results_df["Predicted_PD"] = np.round(logit_probs, 4)
    results_df["Credit_Score"] = credit_scores
    results_df["Risk_Grade"] = pd.cut(
        results_df["Credit_Score"],
        bins=[299, 550, 650, 750, 900],
        labels=["Subprime (High Risk)", "Near-Prime (Moderate)", "Prime (Low Risk)", "Super-Prime (Minimal)"]
    )

    print("\n--- Scorecard Risk Grade Calibration Table ---")
    grade_summary = results_df.groupby("Risk_Grade", observed=False).agg(
        Borrowers=("Actual_Default", "count"),
        Observed_Default_Rate=("Actual_Default", "mean"),
        Mean_Predicted_PD=("Predicted_PD", "mean"),
        Avg_Score=("Credit_Score", "mean")
    ).fillna(0)
    grade_summary["Observed_Default_Rate"] = (grade_summary["Observed_Default_Rate"] * 100).round(2).astype(str) + "%"
    grade_summary["Mean_Predicted_PD"] = (grade_summary["Mean_Predicted_PD"] * 100).round(2).astype(str) + "%"
    grade_summary["Avg_Score"] = grade_summary["Avg_Score"].round(0).astype(int)
    print(grade_summary)

    plot_diagnostics(y_test, logit_probs, gbm_probs, credit_scores)


if __name__ == "__main__":
    main()