import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import label_binarize
from joblib import dump, load
from category_encoders import OrdinalEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import logging

def write_schema(code_dir, data):
    schema = data.dtypes.to_json()
    schema = [(k,v["name"]) for k,v in json.loads(schema).items()]
    with open(os.path.join(code_dir, "schema.json"), "w") as f:
        d = json.dumps(dict(schema))
        f.write(d)

code_dir = os.environ.get("code_dir", "/Users/timothy.whittaker/Desktop/git/dr-mlops-git-integration/insurance-lgbm-fit")
logging.info("fit code_dir -> {}".format(code_dir))
with open( os.path.join(code_dir,"feature_detail.yaml"), "r") as f:
    feature_type_dict = yaml.load(f, Loader=yaml.FullLoader)

numeric_features = feature_type_dict["Numeric"]
categorical_features = feature_type_dict["Categorical"]
offset_col = feature_type_dict["Offset"]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("category_encoder", OrdinalEncoder()),
    ]
)

sparse_preprocessing_pipeline = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)

dense_preprocessing_pipeline = Pipeline(
    steps=[
        ("preprocessing", sparse_preprocessing_pipeline),
    ]
)

model = lgb.LGBMRegressor(objective="tweedie", learning_rate= 0.08)

def make_regressor():
    return Pipeline(steps=[("preprocessing", dense_preprocessing_pipeline), ("model", model)])