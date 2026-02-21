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
- [Lending Club Loan Data](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv) (2020)
- Despites it's decription has been updated to be data only from 2020.
- This company is a peer-to-peer lending corp based in the US that is still currently active.

### Initial Data Engineering & Feature Pruning (SQL Pipeline)

Preprocessing was conducted in SQL prior to modeling.

- Reduced dataset from **144 columns → 91 columns**
- Removed redundant and post-outcome leakage variables  
- Standardized feature formats  
- Documented rationale for each pruning decision

Full SQL workflow can be checked here with details on every step: [SQL](https://github.com/Mbu32/Data-Analysis/tree/a5d0f93a8ba6a3a7d62020c47d289a1080efbb36/SQL)


