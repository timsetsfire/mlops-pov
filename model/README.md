This folder contains all the model assests necessary to using `artifact.pkl` with datarobot mlops

* `create_data.py` - contains functions used to create the dataset feed into the model pipeline.  
* `create_pipeline.py` - creates the classificaiton pipeline using `sklearn`, `categoryencoders`, and `lightgbm`.
* `custom_model.py` - wrapper for ease of instantiation
* `custom.py` - contains all the necessary hooks so DataRobot knows how to use the model.
* `schema.json` - schema of the training dataset
* `feature_detail.yaml` - list of variables used as categorical, text, numerics, and target.  
* `model-config.yaml` - info about the model used for created the custom model