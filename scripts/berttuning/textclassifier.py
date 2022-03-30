from transformers import TextClassificationPipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification as AMSC
from scripts.constants import TUNED_DIR


class TextClassifier:
    def __init__(self, modelname: str, modelpath: str):
        """
        :param modelname: name of a tuned model
        :param modelpath: path to a stored fine tuned model
        """
        tokenizer = AutoTokenizer.from_pretrained(modelname)
        model = AMSC.from_pretrained(f"{TUNED_DIR}/{modelpath}")
        self.pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    def classify(self, text: str):
        """
        :param text:
        :return:
        """
        out = self.pipeline(text)
        return out[0]['label']


if __name__ == "__main__":
    modelpath = "2022_03_30_15_28_38_883828"
    clf = TextClassifier(modelname="Seznam/small-e-czech", modelpath=modelpath)
    print(clf.classify("Díky"))
    # model = AMSC.from_pretrained(f"{TUNED_DIR}/{modelpath}")
    # print(model.config)
