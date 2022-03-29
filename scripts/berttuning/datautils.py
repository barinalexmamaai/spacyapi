import yaml
import sys


def loadconfig(path: str) -> dict:
    """
    :param path: path to a configuration file
    :return: configurations as a dictionary
    """
    with open(path) as f:
        try:
            return yaml.load(stream=f, Loader=yaml.FullLoader)
        except IOError as e:
            sys.exit(f"FAILED TO LOAD CONFIG {path}: {e}")
