"""
Usage:
    python datarobot-predict.py <input-file.csv>
 
This example uses the requests library which you can install with:
    pip install requests
We highly recommend that you update SSL certificates with:
    pip install -U urllib3[secure] certifi
"""
import sys
import json
import requests
 
API_URL = 'https://mlops.dynamic.orm.datarobot.com/predApi/v1.0/deployments/{deployment_id}/predictions'    # noqa
API_KEY = 'NWZkMGY2ZTcwZGU1NDkyZjRjMzdlZmU3OjRJeXJrR2dnRHAyb2RpK0JNN2toRkdtTVBtaWVXemVwV3AwQXE1aGJZVWM9'
DATAROBOT_KEY = '544ec55f-61bf-f6ee-0caf-15c7f919a45d'
 
DEPLOYMENT_ID = '5ff78d092cb534e535a9af6f'
 
MAX_PREDICTION_FILE_SIZE_BYTES = 52428800  # 50 MB
 
 
class DataRobotPredictionError(Exception):
    """Raised if there are issues getting predictions from DataRobot"""
 
 
def make_datarobot_deployment_predictions(data, deployment_id):
    """
    Make predictions on data provided using DataRobot deployment_id provided.
    See docs for details:
         https://app.datarobot.com/docs/users-guide/predictions/api/new-prediction-api.html
 
    Parameters
    ----------
    data : str
        If using CSV as input:
        Feature1,Feature2
        numeric_value,string
 
        Or if using JSON as input:
        [{"Feature1":numeric_value,"Feature2":"string"}]
 
    deployment_id : str
        The ID of the deployment to make predictions with.
 
    Returns
    -------
    Response schema:
        https://app.datarobot.com/docs/users-guide/predictions/api/new-prediction-api.html#response-schema
 
    Raises
    ------
    DataRobotPredictionError if there are issues getting predictions from DataRobot
    """
    # Set HTTP headers. The charset should match the contents of the file.
    headers = {
        # As default, we expect CSV as input data.
        # Should you wish to supply JSON instead,
        # comment out the line below and use the line after that instead:
        'Content-Type': 'text/plain; charset=UTF-8',
        # 'Content-Type': 'application/json; charset=UTF-8',
 
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }
 
    url = API_URL.format(deployment_id=deployment_id)
 
    # Prediction Explanations:
    # See the documentation for more information:
    # https://app.datarobot.com/docs/users-guide/predictions/api/new-prediction-api.html#request-pred-explanations
    # Should you wish to include Prediction Explanations or Prediction Warnings in the result,
    # Change the parameters below accordingly, and remove the comment from the params field below:
 
    # params = {
    #     # If explanations are required, uncomment the line below
    #     'maxExplanations': 3,
    #     'thresholdHigh': 0.5,
    #     'thresholdLow': 0.15,
    #     # Uncomment this for Prediction Warnings, if enabled for your deployment.
    #     # 'predictionWarningEnabled': 'true',
    # }
    # Make API request for predictions
    predictions_response = requests.post(
        url,
        data=data,
        headers=headers,
        # Prediction Explanations:
        # Uncomment this to include explanations in your prediction
        # params=params,
    )
    _raise_dataroboterror_for_status(predictions_response)
    # Return a Python dict following the schema in the documentation
    return predictions_response.json()
 
 
def _raise_dataroboterror_for_status(response):
    """Raise DataRobotPredictionError if the request fails along with the response returned"""
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError:
        err_msg = '{code} Error: {msg}'.format(
            code=response.status_code, msg=response.text)
        raise DataRobotPredictionError(err_msg)
 
 
def main(filename, deployment_id):
    """
    Return an exit code on script completion or error. Codes > 0 are errors to the shell.
    Also useful as a usage demonstration of
    `make_datarobot_deployment_predictions(data, deployment_id)`
    """
    if not filename:
        print(
            'Input file is required argument. '
            'Usage: python datarobot-predict.py <input-file.csv>')
        return 1
    data = open(filename, 'rb').read()
    data_size = sys.getsizeof(data)
    if data_size >= MAX_PREDICTION_FILE_SIZE_BYTES:
        print(
            'Input file is too large: {} bytes. '
            'Max allowed size is: {} bytes.'
        ).format(data_size, MAX_PREDICTION_FILE_SIZE_BYTES)
        return 1
    try:
        predictions = make_datarobot_deployment_predictions(data, deployment_id)
    except DataRobotPredictionError as exc:
        print(exc)
        return 1
    print(json.dumps(predictions, indent=4))
    return 0
 
 
if __name__ == "__main__":
    filename = sys.argv[1]
    sys.exit(main(filename, DEPLOYMENT_ID))
 