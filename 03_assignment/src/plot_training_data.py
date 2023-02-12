import argparse
import datetime

import numpy as np
from manager import Manager

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Plot training info")
    arg_parser.add_argument("--parameter-path", type=str, required=True, help="The path to the parameters file")

    args = arg_parser.parse_args()
    params = np.load(args.parameter_path, allow_pickle=True)
    parameters = params.reshape(1, -1)[0][0]
    print(f"Training time: {str(datetime.timedelta(seconds=sum(parameters['time'])))}")
    print(f"Evaluating time: {str(datetime.timedelta(seconds=sum(parameters['eval_time'])))}")
    print(f"Total time: {str(datetime.timedelta(seconds=sum(parameters['time']) + sum(parameters['eval_time'])))}")
    Manager.plot_cost_and_loss(parameters['cost_to_go'], parameters['loss'], parameters['discount_factor'], parameters['episodes'])
