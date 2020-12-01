Ford MLOps Showcase

# Requirements

docker

python packages
* `datarobot==2.23.0b0`
* `datarobot-drum` - need to build from source to have all functionality.

# Fit a model using drum

`drum fit --code-dir ./training-code --input ./data/loss_cost_demo.csv --output ./model --target-type regression --target IncurredClaims  --docker env --verbose`

# Test model

`drum perf-test --code-dir ./model --input ./data/loss_cost_demo_inference.csv --target-type regression --docker env`

# Validate Model

`drum validation --code-dir ./model --input ./data/loss_cost_demo_inference.csv --target-type regression --docker env`

# Push Execution Environment

`python push.py --env-dir ./env --model-dir ./model`

# Push Custom Inference Model

`drum push --code-dir ./model --verbose --logging-level info`

