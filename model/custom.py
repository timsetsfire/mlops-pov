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
    """
    This hook must be implemented with your fitting code, for running drum in the fit mode.

    This hook MUST ALWAYS be implemented for custom training models.
    For inference models, this hook can stick around unimplemented, and won’t be triggered.

    Parameters
    ----------
    X: pd.DataFrame - training data to perform fit on
    y: pd.Series - target data to perform fit on
    output_dir: the path to write output. This is the path provided in '--output' parameter of the
        'drum fit' command.
    class_order : A two element long list dictating the order of classes which should be used for
        modeling. Class order will always be passed to fit by DataRobot for classification tasks,
        and never otherwise. When models predict, they output a likelihood of one class, with a
        value from 0 to 1. The likelihood of the other class is 1 - this likelihood. Class order
        dictates that the first element in the list will be the 0 class, and the second will be the
        1 class.
    row_weights: An array of non-negative numeric values which can be used to dictate how important
        a row is. Row weights is only optionally used, and there will be no filtering for which
        custom models support this. There are two situations when values will be passed into
        row_weights, during smart downsampling and when weights are explicitly provided by the user
    kwargs: Added for forwards compatibility

    Returns
    -------
    Nothing
    """
    code_dir = os.environ["code_dir"]
    from create_pipeline import make_regressor
    from create_data import process_data, write_schema

    X = process_data(code_dir, X)   
    write_schema(code_dir, X)
    estimator = make_regressor()
    estimator.fit(X, y)
    with open("{}/artifact.pkl".format(output_dir), "wb") as fp:
        pickle.dump(estimator, fp)

def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    Predict with the pickled custom model.

    If your model is for classification, you likely want to ensure this function
    calls `predict_proba()`, whereas for regression it should use `predict()`
    """
    print(kwargs)
    code_dir = os.environ["code_dir"]
    from create_data import process_data
    data = process_data(code_dir, data)
    schema = json.load(open("./training-code/schema.json", "r"))
    feats = list(schema.keys())
    predictions = model.predict_proba(data[feats])
    pos_label = kwargs["positive_class_label"]
    neg_label = kwargs["negative_class_label"]
    return pd.DataFrame(predictions, columns = [neg_label, pos_label])
