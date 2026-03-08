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

### Base model and Results

After running through a full EDA, outlier, winsorizing, feature and model significance, Model hypertuning and CV I ended up with this result for our logistic model:

|Fold 1| Fold 2|Fold 3| Fold 4| Fold 5|
| :--- | :--- |:---|:---|:---
|0.69950638|  0.70659691|0.70082593| 0.70390687|0.70252757|

|Mean|Standard Deviation|
| :--- | :--- 
|0.7026|  0.0024|

All details and results can be seen in more detail [Here](/LogisticRegression/ReadMe.md)
