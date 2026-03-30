from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
from catboost import CatBoostClassifier
import utils
from typing import Optional



app = FastAPI(title='Credit Default Risk')

model = CatBoostClassifier()
model.load_model('models/Trees/catboost_tuned.cbm')
utils.preprocessor= joblib.load('models/Trees/preprocessor.pkl')
means_smoothed= joblib.load('models/Trees/state_means.pkl')
global_default_mean = joblib.load('models/Trees/global_default_mean.pkl')
upperbounds= joblib.load('models/Trees/upperbounds.pkl')
lowerbounds= joblib.load('models/Trees/lowerbounds.pkl')
train_columns = joblib.load('models/Trees/train_columns.pkl')


with open('models/config.json') as f:
    config = json.load(f)

threshold = config['threshold']



class LoanApplication(BaseModel):
    # Core loan info — these should always be provided
    loan_amnt: float
    term: float
    annual_inc: float
    dti: float
    addr_state: str
    home_ownership: str = 'RENT'
    verification_status: str = 'Verified'
    purpose: int = 1
    application_type: str = "Individual"  # sensible default
    initial_list_status: str = "w"        # sensible default

    # Zero-imputed — default to 0 since that's what pipeline does
    max_bal_bc: float = 0
    all_util: float = 0
    il_util: float = 0
    open_acc_6m: float = 0
    open_il_12m: float = 0
    open_il_24m: float = 0
    open_rv_12m: float = 0
    open_rv_24m: float = 0
    inq_last_12m: float = 0
    open_act_il: float = 0
    total_bal_il: float = 0
    total_il_high_credit_limit: float = 0
    is_consolidation: float = 0


    #THe rest is optional because our pipeline handles them (whether its flags/NULL its imputed so not strictly needed)
    mths_since_last_delinq: Optional[float] = None
    mths_since_last_record: Optional[float] = None
    mths_since_recent_bc_dlq: Optional[float] = None
    mths_since_recent_revol_delinq: Optional[float] = None
    mths_since_recent_inq: Optional[float] = None
    mths_since_rcnt_il: Optional[float] = None
    mths_since_last_major_derog: Optional[float] = None
    emp_length: Optional[float] = None
    delinq_2yrs: Optional[float] = None
    inq_last_6mths: Optional[float] = None
    revol_bal: Optional[float] = None
    revol_util: Optional[float] = None
    total_acc: Optional[float] = None
    tot_coll_amt: Optional[float] = None
    tot_cur_bal: Optional[float] = None
    avg_cur_bal: Optional[float] = None
    bc_open_to_buy: Optional[float] = None
    bc_util: Optional[float] = None
    mo_sin_old_il_acct: Optional[float] = None
    mo_sin_old_rev_tl_op: Optional[float] = None
    mo_sin_rcnt_rev_tl_op: Optional[float] = None
    mo_sin_rcnt_tl: Optional[float] = None
    mort_acc: Optional[float] = None
    mths_since_recent_bc: Optional[float] = None
    num_accts_ever_120_pd: Optional[float] = None
    num_actv_bc_tl: Optional[float] = None
    num_actv_rev_tl: Optional[float] = None
    num_bc_sats: Optional[float] = None
    num_bc_tl: Optional[float] = None
    num_il_tl: Optional[float] = None
    num_op_rev_tl: Optional[float] = None
    num_rev_accts: Optional[float] = None
    num_rev_tl_bal_gt_0: Optional[float] = None
    num_sats: Optional[float] = None
    num_tl_120dpd_2m: Optional[float] = None
    num_tl_30dpd: Optional[float] = None
    num_tl_90g_dpd_24m: Optional[float] = None
    num_tl_op_past_12m: Optional[float] = None
    pct_tl_nvr_dlq: Optional[float] = None
    percent_bc_gt_75: Optional[float] = None
    pub_rec_bankruptcies: Optional[float] = None
    tax_liens: Optional[float] = None
    tot_hi_cred_lim: Optional[float] = None
    total_bal_ex_mort: Optional[float] = None
    total_bc_limit: Optional[float] = None
    total_rev_hi_lim: Optional[float] = None
    inq_fi: Optional[float] = None
    total_cu_tl: Optional[float] = None
    acc_open_past_24mths: Optional[float] = None
    chargeoff_within_12_mths: Optional[float] = None
    mo_sin_old_il_acct: Optional[float] = None
    months_sincefrst_credit: Optional[float] = None
    public_record: Optional[float] = None
    is_currently_delinq: Optional[float] = None
    has_il_history: Optional[float] = None
    open_acc: Optional[float] = None
    pub_rec: Optional[float] = None
    num_rev_tl_bal_gt_0: Optional[float] = None
    mths_since_recent_inq: Optional[float] = None

@app.get("/")
def root():
    return {'message': 'Credit Default Risk', 'threshold': threshold}

@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(application: LoanApplication):

    X = pd.DataFrame([application.dict()])


    #now for preproc
    X_pre = utils.preprocess_test(
        X,means_smoothed,global_default_mean,
        upperbounds,lowerbounds,utils.numerical_features,train_columns
    )

    X_pre= X_pre[model.feature_names_]

    #now to predict

    proba = model.predict_proba(X_pre)[0,1]
    decision = 'FLAG - Needs further investigation' if proba >= threshold else 'APPROVE'


    return {
        'Default Probability': round(float(proba),4),
        'Decision': decision,
        'Threshold Used': threshold,
        "Risk Tier": get_risk_tier(proba)
    }

def get_risk_tier(proba):
    if proba < 0.1:   return "LOW"
    elif proba < 0.2: return "MEDIUM"
    elif proba < 0.3: return "HIGH"
    else:             return "VERY HIGH"



# Batch endpoint
@app.post("/predict_batch")
def predict_batch(applications: list[LoanApplication]):
    X= pd.DataFrame([a.dict() for a in applications])

    X_pre = utils.preprocess_test(
        X, means_smoothed, global_default_mean,
        upperbounds, lowerbounds,
        utils.numerical_features, train_columns)
    
    X_pre = X_pre[model.feature_names_]
    
    probas = model.predict_proba(X_pre)[:, 1]
    
    return [
        {
            "default_probability": round(float(p), 4),
            "decision": "FLAG" if p >= threshold else "APPROVE",
            "risk_tier": get_risk_tier(p)
        }
        for p in probas
    ]