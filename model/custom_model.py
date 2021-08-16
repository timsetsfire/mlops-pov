import numpy as np
import pandas as pd
import yaml
from joblib import dump, load
from category_encoders import OrdinalEncoder
import lightgbm as lgb
from sklearn.preprocessing import label_binarize
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from create_data import process_data
import os
import json
from copy import deepcopy
# import shap

class CustomModel(object):
    def __init__(self, code_dir):
        self.code_dir = code_dir
        self.model = pd.read_pickle( os.path.join(code_dir, "artifact.pkl"))
        self.schema = json.load(open( os.path.join(code_dir, "schema.json"), "r"))
        with open(os.path.join(code_dir, "feature_detail.yaml"), "r") as f:
            self.feature_type_dict = yaml.load(f, Loader=yaml.FullLoader)
#         self.shap_explainer = shap.TreeExplainer(self.model.steps[-1][1])
        self.transformer = deepcopy(self.model)
        del(self.transformer.steps[-1])

    def predict(self, X):
        pred = self.model.predict_proba(X)
        return pred

    def preprocess(self, X):
        numeric_features = self.feature_type_dict["Numeric"]
        categorical_features = self.feature_type_dict["Categorical"]
        cols = X.columns
        drop_these = list(set(cols).difference(set(numeric_features)).difference(set(categorical_features)))
        X = X.drop([drop_these], axis=1)
        data = process_data(self.code_dir, X)
        return data

            
