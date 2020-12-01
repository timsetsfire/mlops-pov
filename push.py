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
parser.add_argument("--max-wait", default=None)

logging.basicConfig(
    format="{} - %(levelname)s - %(asctime)s - %(message)s".format(__name__)
)
logger = logging.getLogger("drum push")
                    
def main(path, model_dir, max_wait): 
    client = dr.Client(os.environ["DATAROBOT_API_TOKEN"], os.environ["DATAROBOT_ENDPOINT"])
    try:
        with open( os.path.join(path, "env-info.yaml"), "r") as f:
            env_config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info(env_config)
        env_id = env_config.get("id")
        env_desc = env_config.get("description")
        env_name = env_config.get("name")
        env_language = env_config.get("programmingLanguage")
        version_label = env_config.get("versionLabel")
        version_desc = env_config.get("versionDescription")
    except Exception as e:
        logger.error(e)
        logger.error("no environment info yaml found")
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
        docker_context_path = path,
        label = version_label,
        description = version_desc, 
        max_wait = max_wait
    )
    env_config["environmentVersionID"] = version.id
    if model_dir is not None:
        with open( os.path.join(model_dir,"model-metadata.yaml"), "r") as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info("update model metadata yaml")
        model_config["environmentID"] = env.id
        with open(os.path.join(model_dir,"model-metadata.yaml"), "w") as f:
            yaml.dump(model_config, f)
    logger.info("update env info yaml")
    with open(os.path.join(path,"env-info.yaml"), "w") as f:
            yaml.dump(env_config, f)
 
if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    logger.setLevel(args.logging_level)
    logger.info(args)
    main(args.env_dir, args.model_dir, args.max_wait)
