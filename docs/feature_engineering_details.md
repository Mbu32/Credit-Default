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

## Round 1: Initial Purge

| Dropped Feature | VIF | MI | Corr | Reason |
|----------------|-----|-----|------|---------|
| `remainder__home_ownership_RENT` | - | - | - | Dummy variable trap (kept MORTGAGE) |
| `remainder__home_ownership_NONE` | - | 0.000000 | 0.000 | Pure noise |
| `median__chargeoff_within_12_mths` | - | 0.000000 | 0.000 | Pure noise |
| `median__num_sats` | 256.0 | - | - | Perfect twin of `open_acc` |
| `flags__missingindicator_mths_since_last_record` | 824.6 | - | - | Perfect overlap with imputed column |
| `median__pct_tl_nvr_dlq` | 320.6 | 0.003 | 0.014 | Redundant + weak predictor |
| `remainder__public_record` | 175.3 | 0.000000 | - | Zero information |
| `median__total_acc` | 101.3 | 0.0004 | 0.014 | High VIF, negligible MI |
| `median__num_rev_accts` | 74.9 | 0.000000 | 0.005 | Complete clutter |

---

## Round 2: Engineered Feature Validation

**Income vs. Free Cash Flow** (99% correlated)

| Feature | Correlation | MI | Verdict |
|---------|-------------|-----|---------|
| `median__annual_inc` | 0.0456 | 0.0016 | Dropped |
| `FE_free_cash_flow` | **0.0504** | **0.0049** | **Kept** |

**Engineered Feature Performance**

| Feature | Correlation | MI | VIF | Verdict |
|---------|-------------|-----|-----|---------|
| `FE_free_cash_flow` | 0.0504 | 0.0049 | 548.5 | Success - replaces raw income |
| `FE_activity_ratio` | 0.0600 | - | 45.3 | Solid performer |
| `FE_loan_to_income` | 0.0018 | 0.009 | 1.01 | Non-linear (keep for trees) |

---

## Round 3: Twin Feature Resolution

| Dropped Feature | Kept Feature | Correlation | Rationale |
|-----------------|--------------|-------------|-----------|
| `median__num_rev_tl_bal_gt_0` | `median__num_actv_rev_tl` | 0.98 | Kept higher correlation (0.071 vs 0.069) |
| `remainder__tot_cur_bal` | `median__tot_hi_cred_lim` | 0.97 | Kept better predictor (0.074 vs 0.062) |
| `median__num_op_rev_tl` | `FE_activity_ratio` | - | Ratio > raw denominator |
| `median__annual_inc` | `FE_free_cash_flow` | 0.99 | Engineered feature outperforms |

---

## Round 4: The Clean Sweep

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

## Final Feature Set

**Top Performers by Correlation**

| Feature | Correlation | Type |
|---------|-------------|------|
| `median__acc_open_past_24mths` | 0.097 | Raw |
| `median__bc_open_to_buy` | 0.082 | Raw |
| `median__tot_hi_cred_lim` | 0.074 | Raw |
| `median__num_actv_rev_tl` | 0.071 | Raw |
| `median__bc_util` | 0.071 | Raw |
| `zeros__all_util` | 0.069 | Raw |
| `FE_activity_ratio` | 0.060 | Engineered |
| `FE_free_cash_flow` | 0.050 | Engineered |

**Features Removed by Category**

| Category | Count | Examples |
|----------|-------|----------|
| Zero-signal features | 5 | `home_ownership_NONE`, `chargeoff_within_12_mths`, `public_record` |
| Redundant twins | 12 | `num_sats`, `num_rev_tl_bal_gt_0`, `tot_cur_bal` |
| Denominator features | 3 | `num_op_rev_tl`, `annual_inc` |
| Weaker predictors | 8 | `months_sincefrst_credit`, `revol_util`, `num_actv_bc_tl` |
| Missing indicators | 2 | `mths_since_rcnt_il`, `mths_since_recent_revol_delinq` |

---

## Key Takeaways

1. **Engineered features outperformed raw data** - `FE_free_cash_flow` beat raw income despite 99% correlation
2. **Zero-signal features were common** - 5 features had literally 0 Mutual Information
3. **Redundancy was widespread** - Multiple feature pairs showed >0.90 correlation
4. **Longer lookback periods win** - 24-month metrics consistently outperformed 12-month versions
5. **Available credit > total credit** - `bc_open_to_buy` was more predictive than `total_bc_limit`

**Final count:** Started with ~50 features → Reduced to ~25 high-quality, non-redundant predictors.