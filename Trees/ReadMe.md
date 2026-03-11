#  Non-Linear models

This part of the project attempts to maximize accuracy by finding patterns and interactions that can be found in non linear methods like tree ensemble unlike our logistic regression that was our baseline.

---
### Preprocessing

For this phase, I adjusted my feature selection. In the linear baseline, we used VIF and univariate methods to drop highly collinear or linearly insignificant features. We now shift because tree based models natively handle collinearity and discover non-linear relationships, I chose to instead to model a wider feature set. I'll let the algorithm do the heavy lifting, and then use SHAP (Shapley Additive Explanations) post-training to prune unhelpful features.

---

### Base models + hypertune and Learning Curve

To establish our new baseline, I placed three models XGboost, LGBM, and lastly Catboost (tree based models) and cross validate their scored with 3 stratified folds. CatBoost had the highest average AUC with 0.703.

After selecting CatBoost, I optimized its hyperparameters using Optuna and plotted a Learning Curve to check for data saturation and overfitting:

![Learning Curve](Images_trees/LearningCurve_CatBoost.png)

Observations: The model exhibits slight overfitting, with the training curve falling around ~0.76 AUC and the cross-validation curve at ~0.72 AUC. Both curves have flattened out, telling us that simply adding more training data of the same type will not significantly improve performance. Given the high degree of noise that can undoubtedly be found in real world data, our performance aligns closely with other benchmarks for this dataset without resorting to more complex methods (ANN's).

Comparable tries with this dataset can be found on [here](https://www.kaggle.com/code/faressayah/lending-club-loan-defaulters-prediction)

---

### Finding Optimal Threshold

![ROC plus RecallxPrecision](Images_trees/roc_prc.png)

A model's raw probability is useful if we know exactly where to draw the line between our decisions, that being to approve or deny applicants. I mapped the model's performance across these thresholds (as well as use our argmax function) to find optial cutoff:

| Threshold| Precision| Recall| F1| Amount Flagged| Percentage Flagged
| :--- | :--- | :--- | :--- | :--- | :---|
|0.190|        0.318|        0.684|        0.434|45,153|43.2% of applicants  |     
|0.206|        0.332 |       0.637  |      0.437|40,267|38.5% of applicants |
|0.250|        0.369  |      0.513    |    0.429|29,169|27.9% of applicants|
|0.300|        0.415   |     0.392      |  0.403|19,839|19.0% of applicants |
|0.350 |       0.457 |       0.289 |       0.354|13,281|12.7% of applicants  |   
|0.400  |      0.499   |     0.207    |    0.293|8,720|8.3% of applicants   |
|0.500  |      0.572    |    0.091      |  0.157|3,333|3.2% of applicants  |

While **0.206** is the mathematical optimum score for F1, metric don't tell us the whole story so I translated that to dollar values and show their financial impact:

|Threshold |   Expected Loss |       Expected Gain   |     Net Value |          
| :--- | :--- | :--- | :--- |
|0.190 |       $115,222,627  |       $223,082,557  |       $107,859,930|        
|0.206 |       $116,313,212  |       $221,991,973  |       $105,678,761|        
|0.250 |       $122,902,276  |       $215,402,909  |       $92,500,634 |        
|0.300 |       $132,020,591  |       $206,284,593  |       $74,264,002 |        
|0.350 |       $142,279,452  |       $196,025,733  |       $53,746,281 |        
|0.400 |       $151,831,294  |       $186,473,891  |       $34,642,597 |        
|0.500 |       $167,362,652  |       $170,942,532  |       $3,579,880  |

While 0.206 maximizes F1 and net value, it flags 38.5% of applicants. Depending on actual capacity, 0.300 (19% flagged, $74M net value) might be more realistic.


> **Assumptions**  
- Loss Given Default: 60% of loan amount
- Lost interest revenue: average loan x average interest rate
- These are estimates, actual Losses will vary of course 

---


### Interpretations of our Model using SHAP


After running our SHAP and finding Mean Absolute SHAP values, we've pruned out 28 features that we're not contributing, after rerunning our model (with cross validation) we saw a ~0.001 drop in AUC, not a bad deal!


Our top 10 features being:

|Feature | Mean Absolute SHAP|
| :--- | :--- |
|term          |  0.303660|
|loan_to_income|            0.185780|
|acc_open_past_24mths    |        0.124888|
|dti        |    0.115023|
|bc_open_to_buy   |         0.089766|
|percent_bc_gt_75 |           0.072230|
|all_util          |  0.069582|
|tot_hi_cred_lim   |         0.068780 |
|is_consolidation       |     0.062240|
|mths_since_recent_inq  |          0.060376|