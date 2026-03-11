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

To determine which features drove the model's predictions on a global scale, I calculated the Mean Absolute SHAP values across the dataset. The SHAP values represent the impact of a feature on the model's output in log-odds. I also pruned out 28 features that we're not contributing, after rerunning our model (with cross validation) we saw a ~0.001 drop in AUC, not a bad deal!

Bar Plot of mean Absolute SHAP Values
![Mean Absolute SHAP value](Images_trees/SHAP_bar.png)
Key Drivers of Default Risk:

Loan Term (remainder__term): This is by far the strongest feature. On average, the length of the loan swung the model's prediction by 0.3 log-odds. In credit risk, a 60-month term inherently carries much more uncertainty than a 36-month term, and the model heavily relied on it!

Debt-to-Income Dynamics (FE_loan_to_income & remainder__dti): The custom feature I engineered (FE_loan_to_income) outperformed the base DTI metric, this is a good proof that our feature engineering was fruitful! Together, a borrower's raw capacity to take on more debt was definitely  a strong predictor.

Recent Credit Seeking (median__acc_open_past_24mths): The model assigns significant weight to how many new accounts a borrower opened in the last two years. A sudden spike in new credit lines is an indicator of financial distress.

An extra plot:
![alt text](Images_trees/Global_featureImpact.png)

It shows three things at once: Importance (Y-axis), magnitude (X-axis), and Directionality (Color). It proves that High (red) Terms lead to Positive (risky) SHAP values.

## A quick check of correlation

To make sure our model wasn't splitting up significance between features, we can check here

![Correlation](Images_trees/dendrogram.png)

The majority of our (strongest) features we're very well independent of each other, while some definitely showed to be closer to 75% correlation.

---

### Interaction Terms & non-linear patterns

![Interactions](Images_trees/interaction_features.png)

A quick glance at how features are interacting shows that there are very weak interactions for our top 10 features.


However, we can now show why we chose non linear methods and its for this reason:
![non](Images_trees/nonlinear.png)

As you can see, a pure linear model would NOT catch a pattern like this. We can see how Bank Card utility is being assessed by our model, those with high (85%>) card utility (median__bc_util) had a significantly higher risk to them than those below that region and we can again see higher risk with those with no utility as well.