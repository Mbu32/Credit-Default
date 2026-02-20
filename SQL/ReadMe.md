# Consumer Loan Risk: Data Preprocessing & Feature Engineering

This repository contains the SQL implementation for cleaning and transforming raw lending data into a model-ready dataset. The goal was to identify key risk drivers and prepare the features for a binary classification model.

---

## 1. Target Variable Definition
To allow the model to learn the difference between "Good" and "Bad" loans, I created a binary `predictor` column.

* **Target = 1 (Bad):** Loans with status 'Charged Off', 'Default', or 'Does not meet credit policy (Charged Off)'.
* **Target = 0 (Good):** Loans with status 'Fully Paid' or 'Does not meet credit policy (Fully Paid)'.

---

## 2. Feature Selection and Engineering

### 2.1 - 2.3 Data Reduction
* **High Cardinality & Noise:** Dropped `emp_title` and `zip_code`. With hundreds of thousands of unique entries, these columns risk overfitting the model. I opted to rely on income and state-level data instead.
* **Irrelevant Information:** Dropped `url` (all nulls) and `desc`. While `desc` contains text reasons for loans, these are largely redundant as they are already categorized in the `purpose` column.
* **Derived Features:** Created `months_sincefrst_credit` by calculating the difference between the first credit line (`earliest_cr_line`) and the loan issue date (`issue_d`). This captures the borrower's total experience with credit.

### 2.4 Public Records (`pub_rec`)
The data showed that having even one public record significantly increases default probability. I converted this into a **Binary Flag** (0/1) to simplify the model's input.

![Public Record Default Analysis](images/pub_rec.jpg)

### 2.5 Total Accounts (`total_acc`)
Analysis revealed that default rates stagnate around 20% once a borrower has more than 50 accounts. To handle outliers and focus on the most dense part of the distribution, I capped this value at 50.

![Total Accounts Outlier Analysis](images/total_acc.jpg)

### 2.6 Credit Inquiries (`inq_last_6mths`)
Excessive inquiries are a sign of credit hunger. Values above 6 are extreme outliers with low sample sizes, so I capped this at 6 to reduce "noise" in the tail end of the distribution.

![Inquiry Frequency Analysis](images/inq_last_6mths.jpg)

### 2.7 Handling Missing Values (Imputation)
* **Delinquency & Records:** For `mths_since_last_delinq` and `mths_since_last_record`, NULL values were replaced with `999`. This tells the model these borrowers have **no history** of delinquency.
* **DTI:** Missing Debt-to-Income ratios were imputed with the median (19) and capped at 40 to ensure outliers do not skew the regression weights.

### 2.11 Purpose Categorization
I created a high-level flag `is_consolidation` by searching for keywords like "consol" or "card" in the loan title. Debt consolidation carries a specific risk profile compared to other loan types.

---

## How to Run
The full SQL transformation script can be found here: [cleaning_logic.sql](./cleaning_logic.sql)