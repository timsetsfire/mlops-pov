import yaml
import datarobot as dr
import os
import argparse
import logging

parser = argparse.ArgumentParser(description="execution environment push")
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
parser.add_argument("--execution-env-path", default=None)
parser.add_argement("--model-dri", default=None)
parser.add_argument("--logging-level", default="INFO")
parser.add_argument("--max-wait", default=None)

logging.basicConfig(
    format="{} - %(levelname)s - %(asctime)s - %(message)s".format(__name__)
)
logger = logging.getLogger("drum push")
                    
def main(path, model_dir, max_wait): 
    client = dr.Client(os.environ["DATAROBOT_API_TOKEN"], os.environ["DATAROBOT_ENDPOINT"])
    try:
        with open( os.path.join(model_dir,"model-metadata.yaml"), "r") as f:
            model_metadata = yaml.load(f, Loader=yaml.FullLoader)
        logger.info(model_metadata)
        env_id = model_metadata.get("environmentID")
    except:
        model_metadata = None
        env_id = None
    if env_id is None:
        env = dr.ExecutionEnvironment.create(
            name = "Execution Environment",
            description = "some description",
            programming_language= "python")
    else:
        env = dr.ExecutionEnvironment.get(env_id)
    dr.ExecutionEnvironmentVersion.create(
        execution_environment_id = env.id,
        docker_context_path = "./env",
        label = "version 1",
        description = "initializing the execution environment", 
        max_wait = max_wait
    )
    if model_metadata is not None:
        model_metadata["environmentID"] = env.id
        with open(os.path.join(code_dir,"model-metadata.yaml"), "w") as f:
            yaml.dump(model_metadata, f)
 
if __name__ == "main":
    args = parser.parse_args()
    logger.setLevel(args.logging_level)
    logger.info(args)
    main(args.execution_env_path, args.model_dir, args.max_wait)
