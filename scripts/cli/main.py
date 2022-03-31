#!/usr/bin/python3
import sys
import getopt
from scripts.cli import train


def processargs(args, maxsymbols: int = 30) -> dict:
    """
    :param args: command line arguments
    :param maxsymbols: maximum number of symbols per argument
    :return: configuration
    """
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
