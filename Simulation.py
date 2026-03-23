import pandas as pd
import numpy as np
import joblib
import json
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix


#Loading model
model = CatBoostClassifier()
model.load_model('models\catboost_tuned.cbm')
preprocessor  = joblib.load('models\preprocessor.pkl')

with open('models/features.json') as f:
    features = json.load(f)

with open('models/config.json') as f:
    config = json.load(f)

threshold = config['threshold']
lgd = config['lgd']


