# Feature Selection & Engineering Log

Elimination of redundant and non-predictive features using VIF, Mutual Information, and Correlation analysis.

---

## Methodology

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| VIF | > 10 | Detect multicollinearity |
| MI | < 0.001 | Identify noise features |
| Correlation | < 0.01 | Detect weak linear relationships |

---

##  Initial Deletions

| Dropped Feature | VIF | MI | Corr | Reason |
|----------------|-----|-----|------|---------|
| `remainder__home_ownership_RENT` | 501.672883 | 0.008664 | 0.062095 | Dummy variable trap (kept MORTGAGE) |
| `remainder__home_ownership_NONE` | - | 0.000000 | 0.000 | Pure noise |
| `median__chargeoff_within_12_mths` | - | 0.000000 | 0.000 | Pure noise |
| `median__num_sats` | 256.0 | 0.002837 | 0.026935 | Perfect twin of `open_acc` |
| `flags__missingindicator_mths_since_last_record` | 824.6 | - | - | Perfect overlap with imputed column |
| `median__pct_tl_nvr_dlq` | 320.6 | 0.003 | 0.014 | Redundant + weak predictor |
| `remainder__public_record` | 175.3 | 0.000000 | - | Zero information |
| `median__total_acc` | 101.3 | 0.0004 | 0.014 | High VIF, negligible MI |
| `median__num_rev_accts` | 74.9 | 0.000000 | 0.005 | Complete clutter |

---

## Engineered Features Validation

**Income vs. Free Cash Flow** (99% correlated)

| Feature | Correlation | MI | Verdict |
|---------|-------------|-----|---------|
| `median__annual_inc` | 0.0456 | 0.0016 | Dropped |
| `FE_free_cash_flow` | **0.0504** | **0.0049** | **Kept** |

**Engineered Feature Performance**

| Feature | Correlation | MI | VIF | Verdict |
|---------|-------------|-----|-----|---------|
| `FE_free_cash_flow` | 0.0504 | 0.0049 | 548.5 | Pretty good - replaces raw income |
| `FE_activity_ratio` | 0.0600 | - | 45.3 | Definitely keeping |
| `FE_loan_to_income` | 0.0018 | 0.009 | 1.01 | Dropped wasn't adding significant signal (linearly) |

---

## Third round


| Dropped Feature | VIF | Kept Feature | Rationale |
|-----------------|-----|--------------|-----------|
| `median__months_sincefrst_credit` | 49.7 | `median__mo_sin_old_rev_tl_op` | Better predictor (0.050 vs 0.046) |
| `median__revol_util` | 22.8 | `median__bc_util` | Stronger correlation (0.071 vs 0.065) |
| `median__num_actv_bc_tl` | 25.6 | `median__num_actv_rev_tl` | Active rev is stronger feature |
| `flags__missingindicator_mths_since_rcnt_il` | 21.9 | `zeros__all_util` | Better correlation (0.069 vs 0.045) |
| `median__total_bc_limit` | 21.0 | `median__bc_open_to_buy` | Available credit more predictive |
| `median__open_acc` | 29.1 | `median__num_actv_rev_tl` | Low MI (0.00065) vs active accounts |
| `median__num_bc_tl` | 10.3 | `median__num_bc_sats` | Double the MI (0.0041 vs 0.0016) |
| `median__num_tl_op_past_12m` | 10.8 | `median__acc_open_past_24mths` | 24-month version stronger (0.097 vs 0.084) |
| `flags__missingindicator_mths_since_recent_revol_delinq` | 11.4 | `flags__missingindicator_mths_since_recent_bc_dlq` | Slightly better correlation (0.0146 vs 0.0123) |

---



**Final count:** Started with 82 features & Reduced to 57 high-quality, non-redundant predictors.