import sys
import getopt
from scripts.tuning.pathutils import gettimestamp
from scripts.tuning.datautils import loadconfig
from scripts.tuning.finetuner import FineTuner
from scripts.constants import CONFIG_DIR


def initconfig() -> dict:
    """
    :return: default configuration
    """
    config = loadconfig(f"{CONFIG_DIR}/default.yaml")
    config["outpath"] = gettimestamp()
    return config


def optstoconfig(opts, maxsymbols: int = 30) -> dict:
    """
    :param opts: command line options
    :param maxsymbols: maximum number of symbols per argument
    :return: configuration
    """
    config = initconfig()
    for opt, arg in opts:
        if len(arg) > maxsymbols:
            sys.exit(f"Malicious input danger. Keep parameters under {maxsymbols} symbols")
        elif opt in ("-m", "--model"):
            config["modelname"] = arg
        elif opt in ("-d", "--data"):
            config["datapath"] = arg
        elif opt in ("-l", "--lrate"):
            try:
                config["lr"] = float(arg)
            except ValueError:
                sys.exit("Learning rate must be a float value")
        elif opt in ("-e", "--epochs"):
            try:
                config["nepochs"] = int(arg)
            except ValueError:
                sys.exit("Number of epochs must be an integer value")
        elif opt in ("-o", "--out"):
            config["outpath"] = arg.replace(" ", "_")
    return config


def train(opts):
    """
    :param opts: command line options
    :return:
    """
    config = optstoconfig(opts=opts)
    tuner = FineTuner(config=config)
    tuner.train()
    tuner.savemodel()
