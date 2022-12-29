# Reinforcement learning with Deep Q Network

## Description

This project aims at training a deep q network to stabilize a pendulum (_single_ or _double_) at the high top position (when angle and velocity equal to 0).

Two functionalities are available:

* train
* load

### Train the model

This project trains an empty neural network to solve the problem on your local machine using the CPU (support for GPU usage is not currently available).
It follows the deep-Q-learning algorithm presented in 2015 and starts several times from random position to better train the parameters.
After the training, it is possible to save the weights in a file.

### Load the model

It is possible to load the trained weights in a model and simulate the environment to see the behaviour of the agent in the environment.

## How to use it

### Prerequisites

1. This project works with the _pinocchio_ libraries in a ros environment. Please, prepare the environment before going on.

2. Install the requirements
```bash
pip install -r requirements.txt
```

### Run the project

3. Move to _src_ and run the _main.py_ file.

```bash
cd src
```

The _main.py_ file accepts a lot of parameters that can be selected. The table below describes all the possible 
variables that can be adjusted:

| Argument                   | Description                                                                                     | Required | Default         |
|----------------------------|-------------------------------------------------------------------------------------------------|----------|-----------------|
| `-- model`                 | The selected model                                                                              | `False`  | state           |
| `--train`                  | Boolean value that describes if train or load the model                                         | `False`  | `True`          |
| `--weight-path`            | String value where to save/load the weights                                                     | `True`   | -               |
 | `--optimizer`              | The optimizer used to train the model                                                           | `False`  | adam            |
| `--env`                    | The environment to interact with                                                                | `False`  | single_pendulum |
 | `--discount-factor`        | The discount factor of the reinforcement algorithm                                              | `False`  |                 |
| `--learning-rate`          | The learning rate                                                                               | `False`  |                 |
| `--experience-replay-size` | The size of the experience replay                                                               | `False`  ||
| `--batch-size`             | The batch size                                                                                  | `False`  ||
| `--update-target-param`    | Every which steps update the parameters of the target model                                     | `False`  ||
| `--epsilon-start`          | The initial value of the epsilon parameter (describes the probability to select a random choice | `False`  ||
| `--epsilon-decay`          | The epsilon decay                                                                               | `False`  |                 |
| `--epsilon-min`            | The minimum value of epsilon                                                                    | `False`  ||
| `--max-iterations`         | The maximum number of iterations per episode                                                    | `False`  |                 |
| `--episodes`               | The number of episodes                                                                          | `False`  ||
| `--experience-to-learn`    | How many steps are required to start to train the weights of the model                          | `False`  ||

The above parameters can be selected calling the _main.py_ file and adding the desired values
```bash
python main.py --arg_name arg_value
```

