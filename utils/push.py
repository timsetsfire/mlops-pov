import yaml
import datarobot as dr
import os
import argparse
import logging

parser = argparse.ArgumentParser(description="execution environment push")
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
parser.add_argument("--env-dir", default=None)
parser.add_argument("--model-dir", default=None)
parser.add_argument("--logging-level", default="INFO")
parser.add_argument("--max-wait", default=600, type = int)
parser.add_argument("--update-env", default=False, type = bool)

logging.basicConfig(
    format="{} - %(levelname)s - %(asctime)s - %(message)s".format(__name__)
)
logger = logging.getLogger("drum push")

def push_environment(env_dir, max_wait):
    try:
        with open( os.path.join(env_dir, "env-config.yaml"), "r") as f:
            env_config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info(env_config)
        env_id = env_config.get("id")
        env_desc = env_config.get("description")
        env_name = env_config.get("name")
        env_language = env_config.get("programmingLanguage")
        version_label = env_config.get("versionLabel")
        version_desc = env_config.get("versionDescription")
    except Exception as e:
        logger.error("no environment info yaml found")
        raise Exception("no environment info yaml found: {}".format(e))
    if env_id is None:
        logger.info("creating execution environment")
        env = dr.ExecutionEnvironment.create(
            name = env_name,
            description = env_desc,
            programming_language= env_language)
        env_config["id"] = env.id
    else:
        env = dr.ExecutionEnvironment.get(env_id)
        logger.info("creating new execution environment version")
    version = dr.ExecutionEnvironmentVersion.create(
        execution_environment_id = env.id,
        docker_context_path = env_dir,
        label = version_label,
        description = version_desc, 
        max_wait = max_wait
    )
    logger.info("create environment version complete")
    env_config["environmentVersionID"] = version.id
    logger.info("update env info yaml")
    with open(os.path.join(env_dir,"env-config.yaml"), "w") as f:
        yaml.dump(env_config, f)
         
def push_model(model_dir, env_dir):
    logger.info("load model config")
    try:
        with open( os.path.join(model_dir,"model-config.yaml"), "r") as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
    except:
        raise Exception("no model config yaml found")    
    if env_dir is not None:
        with open( os.path.join(env_dir, "env-config.yaml"), "r") as f:
            env_config = yaml.load(f, Loader=yaml.FullLoader)
        model_config["environmentID"] = env_config["id"]
        model_config["environmentVersionID"] = env_config["environmentVersionID"]

    logger.info(model_config)
    env_id = model_config["environmentID"]
    model_id = model_config.get("id")
    name = model_config["name"]
    target_type = model_config["targetType"]
    target_name = model_config["targetName"]
    language = model_config.get("language")
    description = model_config.get("description")
    positive_class_label = model_config.get("positive_class_label")
    negative_class_label = model_config.get("negative_class_label")
    prediction_threshold = model_config.get("prediction_threshold")
    major_update = model_config.get("majorVersion")

    target_type_dict = { 
        "regression": dr.enums.TARGET_TYPE.REGRESSION,
        "binary": dr.enums.TARGET_TYPE.BINARY,
        "multiclass": dr.enums.TARGET_TYPE.MULTICLASS, 
        "unstructured": None
    }

    if model_id is None:
        logger.info("create new inference model")
        cm = dr.CustomInferenceModel.create(
            name,
            target_type_dict[target_type],
            target_name,
            language,
            description,
            positive_class_label,
            negative_class_label,
            prediction_threshold,
        )
    else:
        logger.info("grab existing inference model")
        cm = dr.CustomInferenceModel.get(model_id)
    
    logger.info("creating a new model version")
    model_version = dr.CustomModelVersion.create_clean(
        custom_model_id = cm.id,
        base_environment_id = env_id,
        is_major_update = major_update,
        folder_path = model_dir
    )  

    model_config["id"] = cm.id
    model_config["modelVersionID"] = model_version.id

    logger.info("update model metadata yaml")
    logger.info(model_config)
    with open(os.path.join(model_dir,"model-config.yaml"), "w") as f:
        yaml.dump(model_config, f) 
                  
def main(env_dir, model_dir, max_wait, update_env): 
    client = dr.Client(os.environ["DATAROBOT_API_TOKEN"], os.environ["DATAROBOT_ENDPOINT"])
    if env_dir is not None:
        logger.info("updating environment")
        if update_env:
            push_environment(env_dir, max_wait)
    if model_dir is not None:
        push_model(model_dir, env_dir)
 
if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    logger.setLevel(args.logging_level)
    logger.info(args)
    main(args.env_dir, args.model_dir, args.max_wait, args.update_env)
