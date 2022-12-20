from enum import Enum


class EnvironmentType(Enum):
    """
    Enumerator that describes which types of environment the project interact with.
    At every element corresponds a different environment.
    """
    SINGLE_PENDULUM = "single_pendulum"


class EnvironmentManager:

    @staticmethod
    def get_environment(env_type: EnvironmentType):
        """
        Return the desired environment.

        :param env_type: the type of environment.
        :return: the desired environment.
        """
        pass
