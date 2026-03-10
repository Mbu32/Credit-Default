#  Non-Linear models

This part of the project attempts to maximize accuracy by finding patterns and interactions that can be found in non linear methods like tree ensemble unlike our logistic regression that was our baseline.

---
### Preprocessing

I kept the same preprocessing except when it came to dropping columns. We used VIF and univariate methods to find significance/non-signficance with features but these were simply linear methods. I chose to keep all columns and let our tree model do the work and then plotting SHAP values and other methods listed later to ascertain whether or not we will drop features or not!


---

### SHAP plots

