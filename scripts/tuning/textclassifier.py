from transformers import TextClassificationPipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification as AMSC
from scripts.constants import TUNED_DIR


class TextClassifier:
    def __init__(self, modelpath: str):
        """
        :param modelpath: path to a stored fine tuned model
        """
        path = f"{TUNED_DIR}/{modelpath}"
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AMSC.from_pretrained(path)
        self.pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    def predict(self, text: str) -> str:
        """
        :param text: input string to classify
        :return: predicted label
        """
        out = self.pipeline(text)
        return out[0]['label']

    def predictbatch(self, batch: list) -> list:
        """
        :param batch: list of strings to classify
        :return: list of predicted labels
        """
        out = self.pipeline(batch)
        return [prediction['label'] for prediction in out]


if __name__ == "__main__":
    modelpath = "basic"
    clf = TextClassifier(modelpath=modelpath)
    import time
    s = time.time()
    label = clf.predict("vrať se zpátky")
    e = time.time() - s
    print(label, e * 1000)
    s = time.time()
    labels = clf.predictbatch(["Potřebuju zablokovat kartu", "dál už nic"])
    e = time.time() - s
    print(labels, e * 1000)
