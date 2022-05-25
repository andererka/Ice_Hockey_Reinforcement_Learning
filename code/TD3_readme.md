# TD3 Code structure
## Main files
* TD3_main_training - start of each training run, sets all hyperparameters
* TD3_main_load_and_run - evaluates run based on weight directory over 1000 episodes
* TD3/DDPG - main implementation of TD3
* TD3/Training_DDPG - implementation of the training loop

## Weights from runs
* saved under TD3/saves
* final evaluation always done on weights from folder /best_weights
* vars.json describes hyperparameters