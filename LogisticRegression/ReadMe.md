# Logistic Regression Analysis

We'll go through EDA, cv, model tuning, etc.


---

### 1. Target Encoding

For highly cardinal categorical data (ex. like addr_state for our dataset) where we have 49 different categories, applying One Hot Encoding would not be the ideal method, I instead chose to to Target Encode add_state. We did this by calculating percentage of default for each state and replacing that state with that value. 

For the three other categorical columns such as: Home Ownership, Verification Status (of income), and Application type. These were OneHotEncoded with one of the columns dropped to avoid perfect collinearity.


### 2. Pipeline for missing values / Flags

For most columns they were missing 1 or 2 values, I opted to impute with median. If computer capabilities were possible I would've used a KNN neighbours imputation method to find a value for each feature that would make most sense for each instance (Individual) through a cluster method.

The second set of columns were imputed with Zero's, this was because most individuals who didn't qualify/open some specific account, their details would be filled with NaN's instead of any value. 0's tell us instead that they were not in use.

Last column, we added a flag indiator. I opted for this method simpy because most instances lacked data for these columns. For example, we have mths_since_last_delinq, a column that told us how many months since person was in delinquency stage (overdue payment), for customer who never went into delinquency, we would simply have NA's. Customer's that have gone into arrears or delinquency get a value of zero, customers with NaN's get a value of 1.


### 3. Univariate Analysis

we did individual t-test to analyze if there was a significanct difference between groups in default (predictor = 1) and paid off (predictor = 0)

Note: Keep in mind that we are working with many samples and so  many of our features could appear to be significant but actually not be simply due to too many instances.

formula for reference

t = (x_1 - x_2) / sqrt( (s_1/n) + (s_2/n))


We still had many columns to be dropped, specifics can be seen in ['Logistic Regression Notebook'](endtoend_log_upd.ipynb)


## 4. Feature Engineering

Since we already had many columns, I didn't want to add to the complexity. Instead, I chose a couple of columns that had a high VIF and chose to create a ratio of them where it made sense such as:

Free cash flow = Monthly Income * (1 -  DTI/100)

- Represents actual disposible income after debt obligations, where higher values indicate stronger ability to repay
- Monthly Income we got from (Annual Income / 12)
- dti is debt to income ratio.


Activity Ratio = num_actv_rev_tl / num_op_rev_tl

where num_actv_rev_tl is number of currently active revolving trades
and num_op_rev_tl is number of open revolving accounts




## 5. VIF check

Now I began to diagnose, checking columns that had a high VIF. This tells us whether or not columns were adding anything or simply repeating the information already present in other columns.

I used a combination of Mutual Information, linear correlation with target, and lastly a Correlation Heatmap.

When I find a column with high VIF I check our heatmap to see what column most correlates with the problematic one then I move on to correlation with target. 

Sample from our initial VIF:

| Feature | VIF | Corr with Target | Mutual Information
| :--- | :--- |:---|:---
| **missingindicator_mths_since_last_record** | 824.647813|0.029706| 0.014045|
| **pct_tl_nvr_dlq** | 320.689288 |0.014923| 0.003169|
| **num_sats** |256.721724 | 0.026935| 0.002837|

Heatmap too large to add, but for our final VIF check:

| Feature | VIF | Corr with Target | Mutual Information
| :--- | :--- |:---|:---
|**bc_util**|  25.310456|0.071276|0.004328|
|**has_il_history**|  20.112435|0.016097  |0.012967|
|**term** | 19.207303|0.174618|0.021245 |
|**activity_ratio**  |17.793271|0.060687|0.003632|

With the respective heatmap (much smaller now)

!['Heatmap'](..\Images_log\corr_heatmap2.png)

[For detailed feature engineering methodology, see the **Feature Engineering Details** section below](..\docs\feature_engineering_details.md)


## 6 Outlier Analysis

We plotted our (normalized) features to verify if we had any really big outliers. We found almost a tenth of features did display features that were >40 standard deviations away. While instances that contain these outliers could be the key for our model predictions, at a certain point, they simply distract our model. Instead of removing outliers, we winsorized them to our upper / lower bounds of 0.99 & 0.01 respectively.


we plotted 10 features in each plot
Sample plot:

!['Feature Distribution'](..\Images_log\boxplot_features.png)

## 7. Cohen's D & Confidence Intervals

Looking at our top 4 features:

|Feature | Cohens d |  CI, (lower,Upper) | mean difference (default-paid)|
| :--- | :--- |:---|:---
|**term** |0.44  |   4.3, 4.64 |4.47|
|**dti**|0.27| 2.13, 2.39|2.26|
|**acc_open_past_24mths**| 0.25| 0.69, 0.79|0.74|
|**tot_hi_cred_lim** |0.19|-32305.8, -28001.1| -30153.45|

      0.22   -3080.43, -2737.1                        -2908.77
Term: On average, borrowers who default are taking out loans that are 4.4 months LONGER than borrowers who pay back.

DTI: defaulters have a debt to income ratio that is 2.27 higher, this tells us they are already financially stretched thin before said loan.

tot_hi_cred_lim: Because it's negative, it means the Paid group is higher. Borrowers who successfully pay off their loans have, on average, ~$30,150 MORE in total high credit limits than people who default.
