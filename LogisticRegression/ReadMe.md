#  Logistic Regression & Statistical Baseline

This part of the project focuses on establishing a good baseline. We move through EDA, statistical significance testing, and hyperparameter optimization to create a benchmark for non-linear models.

---

### Target & One Hot Encoding
For high cardinality data like `addr_state` (with 49 categories), One-Hot Encoding would have introduced excessive sparsity. Instead, I used Target Encoding, replacing each state with its default rate. 

For lower cardinality features like **Home Ownership**, **Verification Status**, and **Application Type** I used One Hot Encoding, dropping one column per feature to avoid perfect multicollinearity.

### Imputation & Missing Value Flags
* **Median Imputation:** I applied to features with minimal NaN's. KNN-Imputation was considered but using Median was chosen due to computational efficiency and stability across our 100,000+ records.
* **Zero Filling:** Applied to account specific features (ex. specific credit lines). A `NaN` here tells us the borrower doesn't have that account type, making `0` a logical value for it.
* **Indicator Flags:** For features like `mths_since_last_delinq`, I added a binary flag. A `1` indicates the data was missing because the borrower has never been delinquent.

### Univariate Analysis (t-tests)
I performed individual t-tests to identify significant differences between the "Default" ($y=1$) and "Paid" ($y=0$). 

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

Note: With a dataset of this size, nearly every feature can appear statistically significant. To bypass this I prioritized Effect Size (Cohen's d) over p-values alone.

---

## Feature Engineering
To capture complex borrower behavior without adding unnecessary complexity to our model, I created ratios for features that showed high Variance Inflation Factors (VIF):

* **Free Cash Flow:** `Monthly_Income * (1 - DTI/100)`. This shows the actual disposable income available to service the loan after all other debt obligations.
* **Activity Ratio:** `num_actv_rev_tl / num_op_rev_tl`. This measures a borrower's current credit usage relative to their total available lines.

---

## Multicollinearity & VIF Diagnosis
I used **Mutual Information**, **Linear Correlation**, and **Variance Inflation Factor (VIF)** to justify column drops:

Checking columns that had a high VIF. This tells us whether or not columns were adding anything or simply repeating the information already present in other columns.
I used a combination of Mutual Information, linear correlation with target, and lastly a Correlation Heatmap.
When I find a column with high VIF I check our heatmap to see what column most correlates with the problematic one then I move on to correlation with target/ MI on deiciding which column to drop.


**Initial VIF sample top 3:**
| Feature | VIF | Corr with Target | Mutual Information |
| :--- | :--- | :--- | :--- |
| **missingindicator_mths_since_last_record** | 824.65 | 0.0297 | 0.0140 |
| **pct_tl_nvr_dlq** | 320.689288 |0.014923| 0.003169|
| **num_sats** | 256.72 | 0.0269 | 0.0028 |



**Final VIF Results sample top 3:**
| Feature | VIF | Corr with Target | Mutual Information |
| :--- | :--- | :--- | :--- |
| **bc_util** | 25.31 | 0.0713 | 0.0043 |
| **term** | 19.21 | 0.1746 | 0.0212 |
| **activity_ratio** | 17.79 | 0.0607 | 0.0036 |

![Final Heatmap](/Images_log/corr_heatmap2.png)

---

## Outlier Analysis & Winsorization
Visual analysis of scaled features revealed extreme outliers with some reaching >40 standard deviations from the mean. While these can be valuable tailrisk information, they could pull our model's decision boundary. I **Winsorize** these features, capping them at the 1st and 99th percentiles to retain the data points and maintain our tailrisk.

![Feature Distribution](/Images_log/boxplot_features.png)

---

## Effect Size: Cohen's d & CI
To identify the strongest historical signals, I calculated Cohen's $d$ and 95% Confidence Intervals:

| Feature | Cohen's d | CI (Lower, Upper) | Mean Diff (Default-Paid) |
| :--- | :--- | :--- | :--- |
| **term** | 0.44 | 4.30, 4.64 | 4.47 |
| **dti** | 0.27 | 2.13, 2.39 | 2.26 |
| **tot_hi_cred_lim** | 0.19 | -32305.8, -28001.1 | -30153.45 |

* **Term:** Defaulters take out loans that are, on average, **4.4 months longer**.
* **DTI:** Defaulters carry a DTI **2.27 points higher**, telling that they are financially over leveraged.
* **Total Credit Limit:** Successful payers have **~$30,150 more** in total high credit limits.

---

## Model Performance & Optimization

### 5-Fold Cross Validation 

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.6995 | 0.7066 | 0.7008 | 0.7039 | 0.7025 | **0.7026** | **0.0024** |

### Optuna Hyperparameter Tuning
AUC using ElasticNet (a blend of L1 and L2 penalties). 

| Parameter | Final Value |
| :--- | :--- |
| **AUC** | 0.7026 |
| **C (Inverse Regularization)** | **90.31** |
| **l1_ratio** | 0.6871 |

> **Key Insight:** Our final $C$ value is significantly higher than our initial model ($C=90.31$ vs. $C=0.009$). This is a major success: it shows that by cleaning the noise (outliers/VIF pruning), the model now trusts the features more and needs far less regularization to maintain stability.

---

## Final Model Interpretation

### Learning Curve
![Learning Curve](/Images_log/LearningCurve_Log.png)
The convergence of training and validation scores suggests the model has reached its linear capacity. While more data could enhance the result, the incremental gain would be trivial and minimal.

### Impact Analysis: Weights vs. Effects
While the Weight Plot shows the importance of features, the Effect Plot shows the realized impact ($Weight \times Value$) across our the borrowers.

![Weight Plot](/Images_log/LogisticRegression_feature_weight.png)
![Effect Plot](/Images_log/Logreg_effectplot.png)

*  `all_util` and `acc_open_past_24mths` show wide whiskers in the effect plot, meaning they are the most powerful individual "risk shifters" in our dataset.
*  A final multivariate analysis via `statsmodels` confirmed the model's overall significance (LLR p-value: 0.0000).

**Conclusion:** This baseline proves that linear patterns can capture ~70% of the risk. We now move to **Tree-Based Models** to capture the non-linear interactions.