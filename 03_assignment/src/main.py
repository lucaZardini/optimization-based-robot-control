import argparse

from environment.environment_type import EnvironmentType
from manager import Manager
from model.model import DQNType
from model.optimizer import OptimizerType


class DefaultValues:
    DISCOUNT = 0.99
    LEARNING_RATE = 1e3
    EXPERIENCE_REPLAY = 10000
    BATCH_SIZE = 32
    # TODO: How many times update the target parameters
    # TODO: count on how many frames to collect before start training
    NX = 2
    NU = 1


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Reinforcement learning")
    arg_parser.add_argument("--model", type=str, required=False, default="standard", help="The model to train or to lead")
    arg_parser.add_argument("--train", type=bool, default=True, required=False,
                            help="Train or lead a pretrained model. If you want to load a pretrained model, the path is required")
    arg_parser.add_argument("--weight_path", type=str, required=False, help="The path to the weights of the pretrained model")  # TODO default
    arg_parser.add_argument("--optimizer", type=str, required=False, help="Optimizer used to train the model, default is adam", default="adam")
    arg_parser.add_argument("--env", type=str, required=False, default="single_pendulum", help="The environment to train/load the model")
    arg_parser.add_argument("--discount_factor", type=float, required=False, default=DefaultValues.DISCOUNT, help="Discount factor")
    arg_parser.add_argument("--learning_rate", type=float, required=False, default=DefaultValues.LEARNING_RATE, help="Learning rate")
    arg_parser.add_argument("--nx", type=int, required=False, default=DefaultValues.NX, help="Number of states")  # TODO: probably property of the environment itself
    arg_parser.add_argument("--nu", type=int, required=False, default=DefaultValues.NU, help="Number of controls")  # TODO: probably property of the environment itself
    arg_parser.add_argument("--experience_replay_size", type=float, required=False, default=DefaultValues.EXPERIENCE_REPLAY,
                            help="Experience replay size")
    arg_parser.add_argument("--bach_size", type=float, required=False, default=DefaultValues.BATCH_SIZE,
                            help="Batch size")

    args = arg_parser.parse_args()

    model_type = DQNType(args.model)
    optimizer_type = OptimizerType(args.optimizer)
    if not args.train:
        if args.weight_path is None:
            raise ValueError("You need to specify the path to the pretrained weights")
    env_type = EnvironmentType(args.env)

    # TODO: add the new parameter to the manager
    manager = Manager(args.discount_factor, args.learning_rate, optimizer_type, model_type, model_type, args.nx, args.nu, env_type)
