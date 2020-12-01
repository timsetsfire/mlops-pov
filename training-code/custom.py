import pickle
import pandas as pd
import numpy as np
import yaml
from typing import List, Optional, Any, Dict
import logging
import os

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
