#!/usr/bin/python3
import sys
import getopt
from scripts.cli.train import train
from scripts.cli.classify import classificationloop

SHORTOPTS = "htc:m:d:l:e:o:"
LONGOPTS = ["help", "train", "classify=", "model=", "data=", "lrate=", "epochs=", "out="]
HELP = "-h/--help - info about command line arguments" \
       "\n-t/--train - run training with specified parameters" \
       "\n-c/--classify= - start classification loop with the specified model" \
       "\n-m/--model= - name of a hugging face model to fine tune" \
       "\n-d/--data= - path to a directory with training data" \
       "\n-l/--lrate= - learning rate" \
       "\n-e/--epochs= - number of training epochs" \
       "\n-o/--out= - path to a directory where to store trained model"


def processargs(args, maxsymbols: int = 30):
    """
    :param args: command line arguments
    :param maxsymbols: maximum number of symbols per argument
    """
    try:
        opts, args = getopt.getopt(args=args, shortopts=SHORTOPTS, longopts=LONGOPTS)
    except getopt.GetoptError:
        sys.exit("Input format error. Try -h or --help")
    for opt, arg in opts:
        if len(arg) > maxsymbols:
            sys.exit(f"Malicious input danger. Keep parameters under {maxsymbols} symbols")
        elif opt in ("-h", "--help"):
            print(HELP)
        elif opt in ("-t", "--train"):
            train(opts=opts)
            break
        elif opt in ("-c", "--classify"):
            classificationloop(modelpath=arg)
            break


processargs(args=sys.argv[1:])
