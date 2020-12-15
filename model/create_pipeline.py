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
import os

code_dir = os.environ["code_dir"]

with open(os.path.join(code_dir, "feature_detail.yaml"), "r") as f:
    feature_type_dict = yaml.load(f, Loader=yaml.FullLoader)

numeric_features = feature_type_dict["Numeric"]
categorical_features = feature_type_dict["Categorical"]

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

model = lgb.LGBMClassifier()

def make_regressor():
    return Pipeline(steps=[("preprocessing", dense_preprocessing_pipeline), ("model", model)])