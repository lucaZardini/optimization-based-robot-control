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
| `--train`                  | Boolean value that run the project in training mode                                             | `False`  |                 |
| `--eval`                   | Boolean value that run the project in evaluating mode                                           | `False`  |                 |
| `--weight-path`            | String value where to save/load the weights                                                     | `True`   | -               |
 | `--optimizer`              | The optimizer used to train the model                                                           | `False`  | adam            |
| `--env`                    | The environment to interact with                                                                | `False`  | single_pendulum |
 | `--discount-factor`        | The discount factor of the reinforcement algorithm                                              | `False`  | 0.99            |
| `--learning-rate`          | The learning rate                                                                               | `False`  | 1e-3            |
| `--experience-replay-size` | The size of the experience replay                                                               | `False`  | 10000           |
| `--batch-size`             | The batch size                                                                                  | `False`  | 32              |
| `--update-target-param`    | Every which steps update the parameters of the target model                                     | `False`  | 1000            |
| `--update-critic-param`    | Every which steps update the parameters of the critic model                                     | `False`  | 10              |
| `--epsilon-start`          | The initial value of the epsilon parameter (describes the probability to select a random choice | `False`  | 1               |
| `--epsilon-decay`          | The epsilon decay                                                                               | `False`  | 0.9995          |
| `--epsilon-min`            | The minimum value of epsilon                                                                    | `False`  | 0.002           |
| `--max-iterations`         | The maximum number of iterations per episode                                                    | `False`  | 500             |
| `--max-iterations-eval`    | The maximum number of iterations to evaluate the model                                          | `False`  | 500             |
| `--episodes`               | The number of episodes                                                                          | `False`  | 500             |
| `--experience-to-learn`    | How many steps are required to start to train the weights of the model                          | `False`  | 32              |

The above parameters can be selected calling the _main.py_ file and adding the desired values
```bash
python main.py --arg_name arg_value
```

It is required to specify if you want to train a new model or evaluate an existing one.

To train a neural network, run the following command line:
```bash
python main.py --weight-path your_name.h5 --train
```

By default, the best model so far found by the algorithm is saved into the `weight_models` folder.
Inside this folder, depending on the environment selected, the weights are stored into the `single_pendulum` or `double_pendulum` folder.
When the algorithm terminates, it saves the best model in that folder with the name specified.
Additionally, two other files are stored there:
- A file named _buffer.npy_ which contains the experience buffer saved (if you want to train again the model, also the buffer replay is required). 
At the moment, the possibility to train a model starting from an existing one and a buffer is not implemented.
- A file named _parameters.npy_ that contains all the info associated with the training, i.e., the training time, the evaluation time, the cost and the loss per each episode.

Instead, to evaluate an existing model, run the following command:
```bash
python main.py --weight-path your_name.h5 --eval
```

In this case, there is not a default to the weight path, so you need to specify the relative path from `src` folder.

Two models are already present in the `weight_models` folder.
To evaluate the _single_pendulum_ model provided, run the following:
```bash
python main.py --weight-path weight_models/single_pendulum/model_single_pendulum.h5 --eval
```

To evaluate the _double_pendulum_ model provided, run the following:
```bash
python main.py --weight-path weight_models/double_pendulum/model_double_pendulum.h5 --env double_pendulum --eval
```


If you want to plot the training info, run the following script located in `src` folder:
```bash
python plot_training_data.py --weight-path path_to_parameters.npy
```
