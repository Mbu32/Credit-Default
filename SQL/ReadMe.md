# Consumer Loan Risk: Data Preprocessing & Feature Engineering

This repository contains the SQL implementation for cleaning and transforming raw lending data into a model-ready dataset. The objective of this project is to identify key risk drivers and prepare a robust feature set for a binary classification model (e.g., Logistic Regression) to predict loan defaults.

---

## 1. Target Variable Definition
To allow the model to learn the difference between "Good" and "Bad" loans, I created a binary `predictor` column based on the historical `loan_status`.

* **Target = 1 (Bad):** Loans with status 'Charged Off', 'Default', or 'Does not meet credit policy (Charged Off)'.
* **Target = 0 (Good):** Loans with status 'Fully Paid' or 'Does not meet credit policy (Fully Paid)'.

---

## 2. Feature Selection and Engineering

### 2.1 - 2.3 Dimensionality Reduction & Credit Experience
* **Identity:** Dropped empty/unnecessary identifiers (`id`, `member_id`, `url`, `desc`) and created a sequential `Loan_ID` for unique record.
* **Cardinality:** Dropped `zip_code`, `emp_title`, and `policy_code` to prevent model overfitting, relying instead on broader geographic regions.
* **Credit:** turned `earliest_cr_line` and `issue_d` into a numeric feature, `months_sincefrst_credit`, to capture the borrower's total credit history.

### 2.4 Public Records (`pub_rec`)
Historical data showed that having even one public record significantly increases default probability. I converted this into a **Binary Flag** (1 = Has Record, 0 = Clean) to simplify the model's input and increase signaling of risk.

<img src="/Images/pub_rec.jpg" width="50%" alt="Public Records Default Rate" />

### 2.5 Open Credit Lines (`open_acc` & `total_acc`)
I analyzed the number of open credit lines to determine the relationship between account volume and default risk. To stabilize the variance in the extreme tail of the distribution, I capped the total number of accounts at 50.

<img src="/Images/open_acc.jpg" width="50%" alt="Open Accounts Default Rate" />

### 2.6 Credit Inquiries (`inq_last_6mths` & `inq_fi`)
Excessive recent inquiries are a strong indicator of "credit hunger" or financial distress. Values above 6 are extreme outliers with low sample sizes, so `inq_last_6mths` was capped at 6, and personal finance inquiries (`inq_fi`) were capped at 3. NULLs were imputed as 0.

<img src="/Images/inq_last_6mths.jpg" width="50%" alt="Inquiries Default Rate" />

### 2.7 & 2.10 Handling Missing Values (Imputation Strategy)
* **Imputing 999:** For time based delinquency variables (e.g., `mths_since_last_delinq`, `mths_since_last_record`, `mths_since_last_major_derog`), NULL values show a lack of negative history. These were filled with `999` to separate "clean" borrowers from those with more recent repayment issues.
* **Debt-to-Income (DTI):** Imputed missing or invalid DTI ratios with the median (19%) and capped extreme outliers at 40% to avoid harmful skews.
* **Recent Delinquency:** Capped `delinq_2yrs` at 5, focusing the model on recent instability without letting extreme cases distort fit.

### 2.11 & 2.14 Purpose Categorization
* **Consolidation Flag:** Created a binary `is_consolidation` flag from a keyword search in the loan `title`. Debt consolidation carried an elevated risk profile compared to other loan types which we confirmed.
* **Purpose Tiers:** Grouped the 14+ `purpose` categories into three tiers (1 = Low Risk, 2 = Medium Risk, 3 = High Risk) based on observed historical default rates, to reduce noise.

<img src="/Images/purpose.jpg" width="70%" alt="Loan Purpose Tiers Analysis" />

### 2.12 & 2.13 Credit Grading & Employment
* **Numeric Conversion:** removed text from `term` and `emp_length` to convert them into numerical formats (e.g., '10+ years' -> 10.0, '< 1 year' -> 0.5).
* **Sub-Grade:** Converted `sub_grade` into a continuous `sub_grade_num` (1-35) and `grade` to (1-7). This allows the model to interpret credit rank as a  progression of risk. As seen below, the default rate increases as the grade drops.

<img src="/Images/grade.jpg" width="50%" alt="Sub-Grade Linear Default Progression" />

### 2.8 & 2.16 Data Leakage Prevention
To ensure the model does not "cheat" by looking at future events, I strictly removed all post loan features. 
* Dropped payment records (`last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d`).
* Dropped hardship metrics (`hardship_flag`, `settlement_amount`, `recoveries`, `collection_recovery_fee`, etc.).

### 2.17 - 2.19 Final Polish & Geography
* **Verification Logic:** combined 'Source Verified' into 'Verified' to create a cleaner binary signal representing whether a borrower's income was verified/audited.
* **Regional Grouping:** Addressed the high cardinality of the 50 US states (`addr_state`) by turning them into 4 Regions (Northeast, Midwest, South, West).

---

## How to Run
The full SQL transformation script, can be found here: [analysis_queries.sql](/SQL/analysis_queries.sql)
