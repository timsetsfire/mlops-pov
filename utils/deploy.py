import yaml
import datarobot as dr
import os
import argparse
import logging
import pprint

from datarobot.mlops.mlops import MLOps
from datarobot.mlops.common.enums import OutputType
from datarobot.mlops.connected.client import MLOpsClient
from datarobot.mlops.common.exception import DRConnectedException
from datarobot.mlops.constants import Constants

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="execution environment push")
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
parser.add_argument("--model-dir", default=None)
parser.add_argument("--mlops-endpoint", default="https://app.datarobot.com")
parser.add_argument("--mlops-api-token", default=os.environ["DATAROBOT_API_TOKEN"])
parser.add_argument("--external", default=False, type = str2bool)
parser.add_argument("--logging-level", default="INFO")
parser.add_argument("--max-wait", default=600, type = int)

logging.basicConfig(
    format="{} - %(levelname)s - %(asctime)s - %(message)s".format(__name__)
)
logger = logging.getLogger("deploy")
       
def deploy_external_model(model_dir, token, endpoint, max_wait):
    logger.info("load model config for external deployment")
    try:
        with open( os.path.join(model_dir,"model-config.yaml"), "r") as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
    except:
        raise Exception("no model config yaml found")    

    target_type_dict = { 
        "regression": dr.enums.TARGET_TYPE.REGRESSION,
        "binary": dr.enums.TARGET_TYPE.BINARY,
        "multiclass": dr.enums.TARGET_TYPE.MULTICLASS, 
        "unstructured": None
    }

    logger.info(model_config)
    env_id = model_config["environmentID"]
    version_id = model_config.get("modelVersionID")
    model_id = model_config.get("id")
    name = model_config["name"]
    target_type = model_config["targetType"]
    target_name = model_config["targetName"]
    language = model_config.get("language")
    description = model_config.get("description")
    positive_class_label = model_config.get("positiveClassLabel")
    negative_class_label = model_config.get("negativeClassLabel")
    prediction_threshold = model_config.get("predictionThreshold")
    major_update = model_config.get("majorVersion")
    training_data_path = model_config.get("trainingData")
    training_data_catalog_id = model_config.get("datasets", {}).get("trainingDataCatalogId")

    model_info = {
            "name": "Lending Club External Model MLOps POV",
            "modelDescription": {
                "description": "no decription" if description is None else description
            },
            "target": {
                "type": target_type_dict[target_type],
                "name": target_name,
                "classNames": [positive_class_label, negative_class_label], 
                "predictionThreshold": prediction_threshold
            }
    }


    # Create connected client
    logger.info("Connecting to MLOps")
    mlops_client = MLOpsClient(endpoint, token)

    # Add training_data to model configuration
    logger.info("Uploading training data - {}. This may take some time...".format(training_data_path))
    if training_data_catalog_id is None:
        dataset_id = mlops_client.upload_dataset(training_data_path)
        logger.info("Training dataset uploaded. Catalog ID {}.".format(dataset_id))
        model_config["datasets"] = {"trainingDataCatalogId": dataset_id}
    else:
        dataset_id = training_data_catalog_id

    # Create the model package
    logger.info('Create model package')
    model_pkg_id = mlops_client.create_model_package(model_info)
    model_pkg = mlops_client.get_model_package(model_pkg_id)
    model_id = model_pkg["modelId"]
    logger.info(f"model id: {model_id}")

    # Deploy the model package
    logger.info('Deploy model package')
    deployment_id = mlops_client.deploy_model_package(model_pkg["id"],name)
    logger.info(f"deployment id: {deployment_id}")

    # Enable data drift tracking
    logger.info('Enable feature drift')
    enable_feature_drift = training_data_path is not None
    mlops_client.update_deployment_settings(deployment_id, target_drift=True,
                                                    feature_drift=enable_feature_drift)

    deployment_settings = mlops_client.get_deployment_settings(deployment_id)                                            
    logger.info(f"deployment settings {pprint.pformat(deployment_settings)}")

    logger.info("Done with deployment.")

    model_config["id"] = model_id
    model_config["deploymentID"] = deployment_id
    model_config["deploymentType"] = "external"

    logger.info("update model metadata yaml")
    logger.info(model_config)
    with open(os.path.join(model_dir,"model-config.yaml"), "w") as f:
        yaml.dump(model_config, f) 

def deploy_custom_model(model_dir, token, endpoint, max_wait):

    client = dr.Client(token = token, endpoint = f"{endpoint}/api/v2")
    logger.info(f"connected {client.endpoint}")
    logger.info("load model config for custom model deployment")
    try:
        with open( os.path.join(model_dir,"model-config.yaml"), "r") as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
    except:
        raise Exception("no model config yaml found")    

    target_type_dict = { 
        "regression": dr.enums.TARGET_TYPE.REGRESSION,
        "binary": dr.enums.TARGET_TYPE.BINARY,
        "multiclass": dr.enums.TARGET_TYPE.MULTICLASS, 
        "unstructured": None
    }

    logger.info(pprint.pformat(model_config))
    env_id = model_config["environmentID"]
    version_id = model_config.get("modelVersionID")
    model_id = model_config.get("id")
    name = model_config["name"]
    target_type = model_config["targetType"]
    target_name = model_config["targetName"]
    language = model_config.get("language")
    description = model_config.get("description")
    positive_class_label = model_config.get("positiveClassLabel")
    negative_class_label = model_config.get("negativeClassLabel")
    prediction_threshold = model_config.get("predictionThreshold")
    major_update = model_config.get("majorVersion")
    training_data_path = model_config.get("trainingData")
    training_data_catalog_id = model_config.get("datasets", {}).get("trainingDataCatalogId")

    pred_server = dr.PredictionServer.list()[0]
    deployment = dr.Deployment.create_from_custom_model_version(
        custom_model_version_id = version_id, 
        label = name, 
        description = description, 
        max_wait = max_wait, 
        default_prediction_server_id = pred_server.id
    )
    
    model_config["deploymentID"] = deployment.id
    model_config["deploymentType"] = "custom inference"
    logger.info("update model metadata yaml")
    logger.info(model_config)
    with open(os.path.join(model_dir,"model-config.yaml"), "w") as f:
        yaml.dump(model_config, f) 


                  
def main(model_dir, token, endpoint, max_wait, external): 
    if external is True:
        logger.info("deploying external model")
        dep = deploy_external_model(model_dir, token, endpoint, max_wait)
    else:
        logger.info("deploying custom inference model")
        dep = deploy_custom_model(model_dir, token, endpoint, max_wait)
    
 
if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    logger.setLevel(args.logging_level)
    logger.info(args)
    main(args.model_dir, args.mlops_api_token, args.mlops_endpoint, args.max_wait, args.external)
