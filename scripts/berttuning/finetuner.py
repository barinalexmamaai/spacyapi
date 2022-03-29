from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import time
import matplotlib.pyplot as plt
from scripts.berttuning.datamanager import DataManager


def showlearningcurve(loss: list, evalloss: list):
    """
    :param loss: list of train loss values
    :param evalloss: list of evaluation loss values
    """
    plt.figure()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    epochs = np.arange(len(loss))
    plt.plot(epochs, loss, color='b')
    plt.plot(epochs, evalloss, color='r')
    plt.legend(['train loss', 'test loss'])
    plt.show()


class FineTuner:
    def __init__(self, config: dict):
        """
        :param config: configuration with training parameters.
            required are: 'modelname', 'datapath'
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["modelname"])
        self.dm = DataManager(path=self.config["datapath"], tokenizer=self.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config["modelname"],
                                                                        num_labels=self.dm.nlabels)
        self.args = self.getargs()
        self.trainer = self.gettrainer()

    def getargs(self) -> TrainingArguments:
        """
        :return: configured training arguments
        """
        return TrainingArguments(
            output_dir="./tunedbert",
            do_eval=True,
            evaluation_strategy="epoch",
            learning_rate=2e-4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            logging_strategy="epoch",
            num_train_epochs=10,
            weight_decay=0.01,
        )

    def gettrainer(self) -> Trainer:
        """
        :return: configured trainer
        """
        return Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.dm.trainset,
            eval_dataset=self.dm.testset,
            tokenizer=self.tokenizer,
        )

    def train(self, learningcurve: bool = False) -> (list, list):
        """
        :param learningcurve: if true show learning curve after training
        :return: (training loss, evaluation loss)
        """
        self.trainer.train()
        history = np.asarray(self.trainer.state.log_history[:-1])
        loss = [entry['loss'] for entry in history[::2]]
        evalloss = [entry['eval_loss'] for entry in history[1::2]]
        if learningcurve:
            showlearningcurve(loss=loss, evalloss=evalloss)
        return loss, evalloss

    def predictbatch(self, batch: list) -> np.ndarray:
        """
        :param batch: list of encoded input
        :return: numpy array of predicted labels
        """
        rawpredictions = self.trainer.predict(batch)

    def humanpredict(self, sentence: str) -> str:
        """
        :return: predicted label
        """
        pass
