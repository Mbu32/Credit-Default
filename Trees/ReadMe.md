#  Non-Linear models

This part of the project attempts to maximize accuracy by finding patterns and interactions that can be found in non linear methods like tree ensemble unlike our logistic regression that was our baseline.

---
### Preprocessing

I kept the same preprocessing except when it came to dropping columns. We used VIF and univariate methods to find significance/non-signficance with features but these were simply linear methods. I chose to keep all columns and let our tree model do the work and then plotting SHAP values and other methods listed later to ascertain whether or not we will drop features or not!


---

### Base models + hypertune and Learning Curve

To begin, we placed three models XGboost, LGBM, and lastly Catboost (tree based models) and cross validate their scored with 3 stratified folds. CatBoost had the highest average AUC with 0.703.

We then moved on to Hypertune our base model via Optuna and then finished off with a Learning Curve

![Learning Curve](Images_trees\LearningCurve_CatBoost.png)

What we can see is that our model definitely has some slight overfitting where the gap of our training curve is sitting at ~0.76 and CV sitting at ~0.72, both curves have also flattened at the end so that tells us that more data won't be helping out.

The model still performs generally well considering human patterns can be extremely hard to predict.

As a sidenote, comparing to other models on Kaggle, I was concerned maybe I was missing out on something but nope, my models pretty close to top models without using ANN's.


