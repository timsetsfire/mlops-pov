import yaml
import datarobot as dr
import os
client = dr.Client(os.environ["DATAROBOT_API_TOKEN"], os.environ["DATAROBOT_ENDPOINT"])

with open( os.path.join(code_dir,"model-metadata.yaml"), "r") as f:
    model_metadata = yaml.load(f, Loader=yaml.FullLoader)
env_id = model_metadata.get("environmentID")
if env_id is None:
    env = dr.ExecutionEnvironment.create(
        name = "Execution Environment",
        description = "some description",
        programming_language= "python")
    model_metadata["environmentID"] = env.id
else:
    env = dr.ExecutionEnvironment.get(env_id)
dr.ExecutionEnvironmentVersion.create(
    execution_environment_id = env.id,
    docker_context_path = "./env",
    label = "version 1",
    description = "initializing the execution environment", 
    max_wait = 4400
)
with open(os.path.join(code_dir,"model-metadata.yaml"), "w") as f:
    yaml.dump(model_metadata, f
