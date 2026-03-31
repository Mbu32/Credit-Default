import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def load_fitted_preprocessor(path):
    global preprocessor
    import joblib
    preprocessor = joblib.load(path)


zero_cols = [
    'max_bal_bc', 'all_util', 'il_util', 'open_acc_6m',
    'open_il_12m', 'open_il_24m', 'open_rv_12m', 'open_rv_24m', 'inq_last_12m',
    'open_act_il', 'total_bal_il', 'total_il_high_credit_limit', 'is_consolidation'
]
flag_cols = [
    'mths_since_last_delinq', 'mths_since_last_record',
    'mths_since_recent_bc_dlq', 'mths_since_recent_revol_delinq',
    'mths_since_recent_inq', 'mths_since_rcnt_il',
    'mths_since_last_major_derog'
]
median_cols = [
    'months_sincefrst_credit', 'annual_inc', 'inq_last_6mths',
    'revol_util', 'total_acc', 'pub_rec', 'open_acc',
    'mo_sin_old_rev_tl_op', 'num_rev_accts', 'tot_hi_cred_lim',
    'acc_open_past_24mths', 'num_bc_sats', 'num_sats', 'mort_acc',
    'mths_since_recent_bc', 'total_bc_limit', 'pub_rec_bankruptcies',
    'total_rev_hi_lim', 'inq_fi', 'avg_cur_bal', 'bc_open_to_buy',
    'bc_util', 'mo_sin_old_il_acct', 'mo_sin_rcnt_rev_tl_op',
    'mo_sin_rcnt_tl', 'num_accts_ever_120_pd', 'num_actv_bc_tl',
    'num_actv_rev_tl', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
    'num_rev_tl_bal_gt_0', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m',
    'pct_tl_nvr_dlq', 'percent_bc_gt_75',
    'total_cu_tl', 'total_bal_ex_mort', 'num_tl_30dpd',
    'num_tl_120dpd_2m', 'chargeoff_within_12_mths'
]

features_todrop = [
    'open_rv_12m', 'mths_since_last_delinq',
    'num_sats', 'open_act_il', 'il_util',
    'mths_since_rcnt_il',
    'mths_since_last_record', 'tax_liens',
    'total_bal_il', 'mths_since_recent_revol_delinq',
    'num_op_rev_tl', 'pub_rec', 'num_bc_tl',
    'mths_since_recent_inq', 'inq_fi',
    'public_record',
    'mths_since_recent_revol_delinq',
    'application_type_Joint App', 'pub_rec_bankruptcies',
    'num_accts_ever_120_pd',
    'mths_since_recent_bc_dlq',
    'home_ownership_OWN', 'has_il_history',
    'num_tl_90g_dpd_24m', 'chargeoff_within_12_mths',
    'num_tl_30dpd', 'num_tl_120dpd_2m',
    'is_currently_delinq', 'home_ownership_NONE',
    'home_ownership_OTHER'
]


numerical_features = ['loan_amnt', 'term', 'emp_length',
       'annual_inc', 'purpose', 'dti', 'delinq_2yrs',
       'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',
       'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
       'initial_list_status', 'mths_since_last_major_derog',
       'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m',
       'open_act_il', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il',
       'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc',
       'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
       'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util',
       'chargeoff_within_12_mths', 'mo_sin_old_il_acct',
       'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
       'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
       'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
       'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
       'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
       'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m',
       'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m',
       'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec_bankruptcies',
       'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort', 'total_bc_limit',
       'total_il_high_credit_limit',
       'months_sincefrst_credit', 'public_record', 'is_consolidation',
       'is_currently_delinq', 'has_il_history']


def winsorize_fn(X):
    return np.array(winsorize(np.array(X), limits=[0.01, 0.01], axis=0))

def make_winsorizer():
    return FunctionTransformer(winsorize_fn, feature_names_out='one-to-one')

def make_ratio(X):
    eps = 0.001
    return (X[:, [0]] / (X[:, [1]] + eps))

def monthlycash(X):
    return ((X[:, [0]] / 12) * (1 - (X[:, [1]] / 100)))

def ratio_name(function_transformer, feature_names_in):
    return ['custom_ratio']




from sklearn import set_config
set_config(transform_output="pandas")

preprocessor = ColumnTransformer([
    ('zeros',  SimpleImputer(strategy='constant', fill_value=0), zero_cols),
    ('flags',  SimpleImputer(strategy='median', add_indicator=True), flag_cols),
    ('median', SimpleImputer(strategy='median'), median_cols)
], remainder='passthrough')



def fit_state_encoding(X_train, y_train, m=10):
    X_temp = X_train.copy()
    X_temp['predictor'] = y_train.values

    global_default_mean = y_train.mean()
    state_means  = X_temp.groupby('addr_state')['predictor'].mean()
    state_counts = X_temp['addr_state'].value_counts()

    means_smoothed = ((state_counts * state_means) + (m * global_default_mean)) / (state_counts + m)

    return means_smoothed, global_default_mean


def apply_state_encoding(X, means_smoothed, global_default_mean):
    X = X.copy()
    X['state_enc'] = X['addr_state'].map(means_smoothed).fillna(global_default_mean)
    return X



categorical_features = ['home_ownership', 'verification_status', 'application_type']

def apply_ohe(X):
    return pd.get_dummies(X[categorical_features], drop_first=True, dtype=int)


def add_features(X):
    eps = 0.001
    X = X.copy()

    X['FE_loan_to_income'] = X['remainder__loan_amnt'] / (X['median__annual_inc'] + eps)
    X['FE_free_cash_flow'] = (X['median__annual_inc'] / 12) * (1 - (X['remainder__dti'] / 100))
    X['FE_activity_ratio'] = X['median__num_actv_rev_tl'] / (X['median__num_op_rev_tl'] + eps)

    return X

def preprocess_train(X_train, y_train, numerical_features):

    means_smoothed, global_default_mean = fit_state_encoding(X_train, y_train)
    X_train = apply_state_encoding(X_train, means_smoothed, global_default_mean)

    X_encoded = apply_ohe(X_train)

    X_train_full = pd.concat([X_train[numerical_features], X_encoded], axis=1)

    train_columns_before_impute = X_train_full.columns.tolist()

    X_train_processed = preprocessor.fit_transform(X_train_full)

    X_train_processed = add_features(X_train_processed)

    #Only clipping numerical Columns
    all_numeric_cols = X_train_processed.select_dtypes(include = [np.number]).columns
    exclude_keywords = ['purpose','home_ownership', 'verification_status', 'application_type']
    cols_to_clip = [col for col in all_numeric_cols if not any(kw in col for kw in exclude_keywords)]

    upperbounds = X_train_processed[cols_to_clip].quantile(0.99)
    lowerbounds = X_train_processed[cols_to_clip].quantile(0.01)
    X_train_processed[cols_to_clip] = X_train_processed[cols_to_clip].clip(lower=lowerbounds, upper=upperbounds, axis=1)

    X_train_processed = X_train_processed.drop(
        columns=[c for c in features_todrop if c in X_train_processed.columns]
    )

    return (
        X_train_processed,
        means_smoothed,
        global_default_mean,
        upperbounds,
        lowerbounds,
        train_columns_before_impute
    )



def preprocess_test(X_test,means_smoothed,global_default_mean,
    upperbounds,lowerbounds,numerical_features,train_columns_before_impute):

    X_test = apply_state_encoding(X_test, means_smoothed, global_default_mean)
    X_encoded_test = apply_ohe(X_test)
    X_test_full = pd.concat([X_test[numerical_features], X_encoded_test], axis=1)
    X_test_full = X_test_full.reindex(columns=train_columns_before_impute, fill_value=0)
    X_test_processed = preprocessor.transform(X_test_full)

    X_test_processed = pd.DataFrame(X_test_processed,columns = preprocessor.get_feature_names_out())

    X_test_processed = add_features(X_test_processed)

    cols_to_clip = upperbounds.index
    X_test_processed[cols_to_clip] = X_test_processed[cols_to_clip].clip(lower=lowerbounds, upper=upperbounds, axis=1)
    X_test_processed = X_test_processed.drop(columns=[c for c in features_todrop if c in X_test_processed.columns])

    return X_test_processed