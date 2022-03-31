import os


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, os.pardir))
DATA_DIR = f"{ROOT_DIR}/data"
CONFIG_DIR = f"{ROOT_DIR}/configs"
OUTPUT_DIR = f"{ROOT_DIR}/output"
