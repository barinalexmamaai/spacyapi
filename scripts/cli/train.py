import sys
import getopt
from scripts.tuning.pathutils import gettimestamp
from scripts.tuning.datautils import loadconfig
from scripts.tuning.finetuner import FineTuner
from scripts.constants import CONFIG_DIR

HELP = "-h/--help - info about command line arguments" \
       "\n-m/--model= - name of a hugging face model to fine tune" \
       "\n-d/--data= - path to a directory with training data" \
       "\n-l/--lrate= - learning rate" \
       "\n-e/--epochs= - number of training epochs" \
       "\n-o/--out= - path to a directory where to store trained model"


def initconfig() -> dict:
    """
    :return: default configuration
    """
    config = loadconfig(f"{CONFIG_DIR}/default.yaml")
    config["outpath"] = gettimestamp()
    return config


def argstoconfig(args, maxsymbols: int = 30) -> dict:
    """
    :param args: command line arguments
    :param maxsymbols: maximum number of symbols per argument
    :return: configuration
    """
    config = initconfig()
    try:
        opts, args = getopt.getopt(args, "hm:d:l:e:o:", ["help", "model=", "data=", "lrate=", "epochs=", "out="])
    except getopt.GetoptError:
        sys.exit("Input format error. Try -h or --help")
    for opt, arg in opts:
        if len(arg) > maxsymbols:
            sys.exit(f"Malicious input danger. Keep parameters under {maxsymbols} symbols")
        if opt in ("-h", "--help"):
            print(HELP)
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
        else:
            pass
            # sys.exit("Unknown parameter specified")
    return config


def train(args):
    """
    :param args: command line arguments
    :return:
    """
    config = argstoconfig(args=args)
    tuner = FineTuner(config=config)
    tuner.train()
    tuner.savemodel()
