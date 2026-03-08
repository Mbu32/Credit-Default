# Applied Data Analysis & Statistical Modeling Projects

This repsitory contains an applied (end-to-end) project demonstrating SQL data pulling/cleaning, statistical reasoning, feature engineering, inference modelling, model diagnostics and interpreation of a real world dataset.
Drawing on my background in mathematics, statistics, and finance, these end-to-end projects will apply Python and machine learning techniques with an emphasis on interpratibilty.

---
## 1. Credit Card Default Risk Modeling

### Objective
Develop and compare classification models for credit default prediction, with emphasis on:
- Proper handling of class imbalance
- Feature diagnostics and multicollinearity control
- Threshold optimization for decision-making
- Out-of-sample validation
- Interpretability in a risk management context

### Dataset
- [Lending Club Loan Data](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv) 
- Despite it's decription has been updated to be data only from 2020.
- This company is a peer-to-peer lending corp based in the US that is still currently active.

### Initial Data Engineering & Feature Pruning (SQL Pipeline)

Preprocessing was conducted in SQL prior to modeling.

- Reduced dataset from **144 columns → 91 columns**
- Removed redundant and post loan outcome leakage variables  
- Documented rationale for each decision

Full SQL workflow can be checked here with details on every step: [SQL](https://github.com/Mbu32/Data-Analysis/tree/a5d0f93a8ba6a3a7d62020c47d289a1080efbb36/SQL)

## Logistic Regression &  Baseline

Before moving to complex non-linear models, I established a baseline using Logistic Regression. This phase focused on: handling extreme outliers (Winsorization at the 1st and 99th percentiles), addressing multicollinearity, and optimizing the solver with ElasticNet regularization.

### Model Performance with 5 Fold Cross-Validation

The model was tuned using Optuna to find the optimal balance of L1 and L2 penalties ($l1\_ratio \approx 0.69$). I had a significantly low standard deviation, suggesting the model has generalized decently well.

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| :--- | :--- | :--- | :--- | :--- |
| 0.6995 | 0.7066 | 0.7008 | 0.7039 | 0.7025 |

> **Mean AUC:** `0.7026`  
> **Standard Deviation:** `0.0024`

---

### Key Risk Drivers:

By combining **Cohen’s d** (effect size) and **Mean Difference**, we show the typical profile of a defaulter vs. a successful borrower.

* **Loan Term:** On average, borrowers who default select terms **4.4 months longer** than those who pay back. Longer exposure directly correlates with higher default probability.
* **Debt-to-Income (DTI) Ratio:** Defaulters carry a DTI that is **2.27 points higher**. This tells us that they are financially stretched thin before the loan even begins.
* **Total High Credit Limit:** Borrowers who successfully repay have on average **~$30,150** MORE in total high credit limits. 

---

###  Effect Plot

I've added the Effect plot to show priority our model put on features rather than the univariate metrics like above. By looking at the distribution below, we can see model placing more emphasis on features like:

![Effect Plot](Images_log/Logreg_effectplot.png)

* `all_util`,  `acc_open_past_24mths`, `loan_amnt`, and `dti` show broad whiskers. This indicates they are the primary drivers of individual risk scores which shifts a borrower significantly toward or away from default based on their specific values.
* While `term` had a massive Cohen's d and weight, it's actual impact in the effect plot is shown by a tight and discrete line this is because it is categorical and its effect is more all or nothing rather than a continuous line.





---

For a more detailed breakdown of all steps taken and all logic behind steps, **[Read Here](/LogisticRegression/ReadMe.md)**
