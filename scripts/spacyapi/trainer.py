from spacy.cli import init_config
from spacy.cli.train import train
from scripts.constants import CONFIG_DIR, DATA_DIR, OUTPUT_DIR
from scripts.pathutils import gettimestamp, create_directories


class Trainer:
    def __init__(self):
        self.basecfg = {'lang': 'en',
                        'pipeline': ['textcat_multilabel'],
                        'opt': 'accuracy',
                        'gpu': False}
        self.cfgpath = f"{CONFIG_DIR}/cfg.cfg"
        self.trainpath = f"{DATA_DIR}/train.spacy"
        self.devpath = f"{DATA_DIR}/valid.spacy"
        self.modelpath = f"{OUTPUT_DIR}/{gettimestamp()}"
        create_directories(path=self.modelpath)

    def initialize_config(self):
        """
        create base.cfg and fill config according to spacy api
        """
        cfg = init_config(lang=self.basecfg['lang'],
                          pipeline=self.basecfg['pipeline'],
                          optimize=self.basecfg['opt'],
                          gpu=self.basecfg['gpu'])
        cfg.to_disk(self.cfgpath)

    def basictrain(self):
        """
        run spacy training based on configuration file
        """
        train(config_path=self.cfgpath,
              output_path=self.modelpath,
              overrides={"paths.train": self.trainpath, "paths.dev": self.devpath})


if __name__ == "__main__":
    t = Trainer()
    t.basictrain()

