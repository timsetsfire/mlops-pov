Ford MLOps Showcase

# Fit a model using drum

drum fit --code-dir ./training-code --input ./data/loss_cost_demo.csv --output ./model --target-type regression --target IncurredClaims  --docker env --verbose

# Test model

drum perf-test --code-dir ./model --input ./data/loss_cost_demo_inference.csv --target-type regression --docker env


# Validate Model

drum validation --code-dir ./model --input ./data/loss_cost_demo_inference.csv --target-type regression --docker env


