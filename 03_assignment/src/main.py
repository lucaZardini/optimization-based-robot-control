import argparse

from environment.environment_type import EnvironmentType
from manager import Manager
from model.model import DQNType
from model.optimizer import OptimizerType


class DefaultValues:
    DQN = "discrete"
    ENV = "single_pendulum"
    DISCOUNT = 0.99
    LEARNING_RATE = 1e-3
    EXPERIENCE_REPLAY = 10000
    BATCH_SIZE = 32
    UPDATE_TARGET_PARAMS = 1000
    EPSILON_START = 1.00
    UPDATE_CRITIC_WEIGHTS = 10
    EPSILON_DECAY = 0.9995  # TODO: update epsilon decay and max iterations together because they are connected.
    EPSILON_MIN = 0.002
    MAX_ITERATIONS = 500  # TODO
    MAX_ITERATIONS_EVAL = 2000000
    EPISODES = 300  # TODO


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Reinforcement learning")
    arg_parser.add_argument("--model", type=str, required=False, default=DefaultValues.DQN, help="The model to train or to lead")
    arg_parser.add_argument("--train", default=False, action="store_true", required=False, help="Create a network and train it")
    arg_parser.add_argument("--eval", default=False, action="store_true", required=False, help="Evaluate a pretrained model, the path is required")
    arg_parser.add_argument("--weight-path", type=str, required=True, help="The path to the weights of the pretrained model, or where to store the model")
    arg_parser.add_argument("--optimizer", type=str, required=False, help="Optimizer used to train the model, default is adam", default="adam")
    arg_parser.add_argument("--env", type=str, required=False, default=DefaultValues.ENV, help="The environment to train/load the model")
    arg_parser.add_argument("--update-critic", type=float, required=False, default=DefaultValues.UPDATE_CRITIC_WEIGHTS, help="At which steps updating the critic weights")
    arg_parser.add_argument("--discount-factor", type=float, required=False, default=DefaultValues.DISCOUNT, help="Discount factor")
    arg_parser.add_argument("--learning-rate", type=float, required=False, default=DefaultValues.LEARNING_RATE, help="Learning rate")
    arg_parser.add_argument("--experience-replay-size", type=float, required=False, default=DefaultValues.EXPERIENCE_REPLAY, help="Experience replay size")
    arg_parser.add_argument("--batch-size", type=float, required=False, default=DefaultValues.BATCH_SIZE, help="Batch size")
    arg_parser.add_argument("--update-target-param", type=int, required=False, default=DefaultValues.UPDATE_TARGET_PARAMS, help="At which steps updating the target parameters")
    arg_parser.add_argument("--epsilon-start", type=float, required=False, default=DefaultValues.EPSILON_START, help="Epsilon start")
    arg_parser.add_argument("--epsilon-decay", type=float, required=False, default=DefaultValues.EPSILON_DECAY, help="Epsilon decay")
    arg_parser.add_argument("--epsilon-min", type=float, required=False, default=DefaultValues.EPSILON_MIN, help="Epsilon min")
    arg_parser.add_argument("--max-iterations", type=int, required=False, default=DefaultValues.MAX_ITERATIONS, help="max iterations")
    arg_parser.add_argument("--max-iterations-eval", type=int, required=False, default=DefaultValues.MAX_ITERATIONS_EVAL, help="max iterations eval")
    arg_parser.add_argument("--episodes", type=int, required=False, default=DefaultValues.EPISODES, help="episodes")
    arg_parser.add_argument("--experience-to-learn", type=int, required=False, default=2 * DefaultValues.BATCH_SIZE, help="Number of experience to collect before starting to train the model")

    args = arg_parser.parse_args()

    model_type = DQNType(args.model)
    optimizer_type = OptimizerType(args.optimizer)

    env_type = EnvironmentType(args.env)

    if not args.train and not args.eval:
        raise ValueError("You need to specify if you want to train a model or evaluate a pretrained one")

    if args.eval:
        if args.weight_path is None:
            raise ValueError("You need to specify the path to the pretrained weights")

    manager = Manager(args.discount_factor, args.learning_rate, optimizer_type, model_type, model_type, env_type,
                      args.batch_size, args.update_target_param, args.epsilon_start, args.epsilon_decay,
                      args.epsilon_min, args.experience_replay_size, args.max_iterations, args.episodes,
                      args.experience_to_learn, args.update_critic, args.max_iterations_eval)

    if args.train:
        manager.train(args.weight_path) # to train
    else:
        manager.load(model_type, args.weight_path) # to evaluate
