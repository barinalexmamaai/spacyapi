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

    def classify(self, text: str) -> str:
        """
        :param text: input string to classify
        :return: predicted label
        """
        out = self.pipeline(text)
        return out[0]['label']


if __name__ == "__main__":
    modelpath = "2022_03_30_15_28_38_883828"
    clf = TextClassifier(modelpath=modelpath)
    import time
    s = time.time()
    label = clf.classify("vrať se zpátky")
    e = time.time() - s
    print(label, e * 1000)
    # model = AMSC.from_pretrained(f"{TUNED_DIR}/{modelpath}")
    # print(model.config)
