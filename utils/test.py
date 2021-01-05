import yaml
import datarobot as dr
import os
import argparse
import logging

parser = argparse.ArgumentParser(description="MLOps test model")
parser.add_argument("--model-dir", default=None)
parser.add_argument("--logging-level", default="INFO")
parser.add_argument("--max-wait", default=3600, type=int)

client = dr.Client(os.environ["DATAROBOT_API_TOKEN"], os.environ["DATAROBOT_ENDPOINT"])

logging.basicConfig(
    format="{} - %(levelname)s - %(asctime)s - %(message)s".format(__name__)
)
logger = logging.getLogger("custom model test")

def main(model_dir, max_wait):
    logger.info("loading model config")
    try:
        with open( os.path.join(model_dir,"model-config.yaml"), "r") as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
    except:
        raise Exception("no model config yaml found")  
    test_dataset_id = model_config.get("testDatasetID")
    if test_dataset_id is None:
        path_to_dataset = model_config.get("testData")
        logger.info("loading dataset to ai catalog")
        test_dataset = dr.Dataset.create_from_file(file_path = path_to_dataset)
        test_dataset_id = test_dataset.id
        model_config["testDatasetID"] = test_dataset_id
        with open(os.path.join(model_dir,"model-config.yaml"), "w") as f:
            yaml.dump(model_config, f) 
    logger.info("starting custom model test")
    cm_test = dr.CustomModelTest.create(
        custom_model_id = model_config["id"],
        custom_model_version_id = model_config["modelVersionID"],
        dataset_id = test_dataset_id, 
        max_wait = max_wait
    )
    logger.info("test complete")
    if cm_test.overall_status == "succeeded":
        logger.info("success! \U0001F37E")
    elif cm_test.overall_status == "warning":
        logger.warning("{} \U0001F631".format(cm_test.overall_status))
        for name, test in cm_test.detailed_status.items():
            logging.warning('Test: {}'.format(name))
            logging.warning('Status: {}'.format(test['status']))
            logging.warning('Message: {}'.format(test['message']))
    else:
        logger.info("{} \U0001F631".format(cm_test.overall_status))
        for name, test in cm_test.detailed_status.items():
            logging.error('Test: {}'.format(name))
            logging.error('Status: {}'.format(test['status']))
            logging.error('Message: {}'.format(test['message']))
    with open("test.log", "w") as f:
        f.write(cm_test.get_log())


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    logger.setLevel(args.logging_level)
    logger.info(args)
    main(args.model_dir, args.max_wait)
