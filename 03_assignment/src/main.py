import argparse

from environment.environment_type import EnvironmentType
from manager import Manager
from model.model import DQNType
from model.optimizer import OptimizerType


class DefaultValues:
    DQN = "state"
    ENV = "single_pendulum"
    DISCOUNT = 0.99
    LEARNING_RATE = 1e-3
    EXPERIENCE_REPLAY = 10000
    BATCH_SIZE = 32
    UPDATE_TARGET_PARAMS = 4
    EPSILON_START = 1.00
    EPSILON_DECAY = 0.999985
    EPSILON_MIN = 0.02
    MAX_ITERATIONS = 1000  # TODO
    EPISODES = 10  # TODO


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Reinforcement learning")
    arg_parser.add_argument("--model", type=str, required=False, default=DefaultValues.DQN, help="The model to train or to lead")
    arg_parser.add_argument("--train", type=bool, default=False, required=False, help="Train or lead a pretrained model. If you want to load a pretrained model, the path is required")
    arg_parser.add_argument("--weight-path", type=str, required=True, help="The path to the weights of the pretrained model, or where to store the model")
    arg_parser.add_argument("--optimizer", type=str, required=False, help="Optimizer used to train the model, default is adam", default="adam")
    arg_parser.add_argument("--env", type=str, required=False, default=DefaultValues.ENV, help="The environment to train/load the model")
    arg_parser.add_argument("--discount-factor", type=float, required=False, default=DefaultValues.DISCOUNT, help="Discount factor")
    arg_parser.add_argument("--learning-rate", type=float, required=False, default=DefaultValues.LEARNING_RATE, help="Learning rate")
    arg_parser.add_argument("--experience-replay-size", type=float, required=False, default=DefaultValues.EXPERIENCE_REPLAY, help="Experience replay size")
    arg_parser.add_argument("--batch-size", type=float, required=False, default=DefaultValues.BATCH_SIZE, help="Batch size")
    arg_parser.add_argument("--update-target-param", type=int, required=False, default=DefaultValues.UPDATE_TARGET_PARAMS, help="At which steps upading the target parameters")
    arg_parser.add_argument("--epsilon-start", type=float, required=False, default=DefaultValues.EPSILON_START, help="Epsilon start")
    arg_parser.add_argument("--epsilon-decay", type=float, required=False, default=DefaultValues.EPSILON_DECAY, help="Epsilon decay")
    arg_parser.add_argument("--epsilon-min", type=float, required=False, default=DefaultValues.EPSILON_MIN, help="Epsilon min")
    arg_parser.add_argument("--max-iterations", type=int, required=False, default=DefaultValues.MAX_ITERATIONS, help="max iterations")
    arg_parser.add_argument("--episodes", type=int, required=False, default=DefaultValues.EPISODES, help="episodes")
    arg_parser.add_argument("--experience-to-learn", type=int, required=False, default=2 * DefaultValues.BATCH_SIZE, help="Number of experience to collect before starting to train the model")

    args = arg_parser.parse_args()

    model_type = DQNType(args.model)
    optimizer_type = OptimizerType(args.optimizer)
    if not args.train:
        if args.weight_path is None:
            raise ValueError("You need to specify the path to the pretrained weights")
    env_type = EnvironmentType(args.env)

    manager = Manager(args.discount_factor, args.learning_rate, optimizer_type, model_type, model_type, env_type,
                      args.batch_size, args.update_target_param, args.epsilon_start, args.epsilon_decay,
                      args.epsilon_min, args.experience_replay_size, args.max_iterations, args.episodes,
                      args.experience_to_learn)

    if args.train:
        manager.train(args.weight_path)
    else:
        manager.load(model_type, args.weight_path)
