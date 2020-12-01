Ford MLOps Showcase

# Requirements

docker

python packages
* `datarobot==2.23.0b0`
* `datarobot-drum` - need to build from source to have all functionality.

# Fit a model using drum

`drum fit --code-dir ./training-code --input ./data/loss_cost_demo.csv --output ./model --target-type regression --target IncurredClaims  --docker env --verbose`

# Test model

Test model performance and get its latency times and memory usage with respect to the image you plan to pair with the model.  In this mode, the model is started with a prediction server. 

`drum perf-test --code-dir ./model --input ./data/loss_cost_demo_inference.csv --target-type regression --docker env`

# Validate Model

Validate the model on a set of various checks. DRUM only supports missing value checks, but DR MLOps runs several others.  Again, complete with the docker image you plan to pair with the model.  

`drum validation --code-dir ./model --input ./data/loss_cost_demo_inference.csv --target-type regression --docker env`

# Push Execution Environment to MLOps

env-config.yaml should be completed.  

`python push.py --env-dir ./env --model-dir ./model`

# Push Custom Inference Model to MLOps

model-metadata.yaml should be comleted.  if a model id is present in the yaml, only major version attribute is evaluated.  

`drum push --code-dir ./model --verbose --logging-level info`

