import numpy as np
from matplotlib import pyplot as plt


class TrajectoryAnalyzer:
    """
    Given two path to two csv files that contain numpy array, it analyzes the differences between the two files.
    """

    @staticmethod
    def read(path: str, delimiter: str):
        return np.genfromtxt(path, delimiter=delimiter)

    @staticmethod
    def read_and_subtract(first_path: str, second_path: str, delimiter: str = ",") -> np.ndarray:
        first_numpy_values = TrajectoryAnalyzer.read(first_path, delimiter)
        second_numpy_values = TrajectoryAnalyzer.read(second_path, delimiter)
        return np.subtract(first_numpy_values, second_numpy_values)


if __name__ == "__main__":
    difference = TrajectoryAnalyzer.read_and_subtract("selection_matrix_trajectory.csv", "mu_factor_selection_matrix_trajectory.csv")
    time_vec = TrajectoryAnalyzer.read("time_vec.csv", ",")
    first_method = TrajectoryAnalyzer.read("selection_matrix_trajectory.csv", ",")
    second_method = TrajectoryAnalyzer.read("mu_factor_selection_matrix_trajectory.csv", ",")

    plt.figure()
    plt.plot(time_vec, difference[:, 0], 'b')
    plt.plot(time_vec, difference[:, 1], 'r')
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('[rad]')
    plt.legend(["1st joint position difference", "2nd joint position difference"],
               loc='upper right')

    plt.figure()
    plt.plot(time_vec, difference[:, 2], 'b')
    plt.plot(time_vec, difference[:, 3], 'r')
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('[rad/s]')
    plt.legend(["1st joint velocity difference", "2nd joint velocity difference"],
               loc='upper right')

    plt.figure()
    plt.plot(time_vec, first_method[:, 0], 'b')
    plt.plot(time_vec, first_method[:, 1], 'r')
    plt.plot(time_vec, second_method[:, 0], 'b--')
    plt.plot(time_vec, second_method[:, 1], 'r--')
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('[rad]')
    plt.legend(["1st joint position first", "2nd joint position first", "1st joint position second", "2nd joint position second"],
               loc='upper right')

    plt.figure()
    plt.plot(time_vec, first_method[:, 2], 'b')
    plt.plot(time_vec, first_method[:, 3], 'r')
    plt.plot(time_vec, second_method[:, 2], 'b--')
    plt.plot(time_vec, second_method[:, 3], 'r--')
    plt.gca().set_xlabel('Time [s]')
    plt.gca().set_ylabel('[rad/s]')
    plt.legend(["1st joint velocity first", "2nd joint velocity first", "1st joint velocity second", "2nd joint velocity second"],
               loc='upper right')

    plt.show()
