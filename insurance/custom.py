import pickle
import pandas as pd
import numpy as np
import yaml
from typing import List, Optional, Any, Dict
import logging
import os
import json


def init(code_dir, **kwargs):
    logging.info("init call, code_dir -> {}".format(code_dir))
    os.environ["code_dir"] = code_dir
    logging.info("init call, kwargs -> {}".format(kwargs))


def fit(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str,
    class_order: Optional[List[str]] = None,
    row_weights: Optional[np.ndarray] = None,
    **kwargs,
):
    from create_pipeline import make_regressor
    offset = np.log(X["Exposure"].values)
    estimator = make_regressor()
    estimator.fit(X, y, model__init_score = offset)
    with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
        pickle.dump(estimator, fp)

def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    Predict with the pickled custom model.

    If your model is for classification, you likely want to ensure this function
    calls `predict_proba()`, whereas for regression it should use `predict()`
    """
    offset = data["Exposure"].values
    predictions = np.exp(model.predict(data, raw_score=True)) * offset
    return pd.DataFrame(predictions, columns = ["Predictions"])
    # return pd.DataFrame(np.ones(X.shape[0],1), columns = ["Predictions"])
