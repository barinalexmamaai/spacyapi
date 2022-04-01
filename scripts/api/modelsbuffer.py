from scripts.tuning.textclassifier import TextClassifier


class ModelsBuffer:
    def __init__(self, size: int = 10):
        """
        :param size: maximum number of models at once
        """
        self.buffer = {}

    def load(self, modelpath: str) -> TextClassifier:
        """
        :param modelpath: local path to a model
        :return: loaded model
        """
        self.buffer[modelpath] = TextClassifier(modelpath=modelpath)
        return self.buffer[modelpath]

    def remove(self, modelpath: str):
        """
        :param modelpath: local path to a model
        :return:
        """
        self.buffer.pop(key=modelpath)

    def get(self, modelpath: str) -> TextClassifier:
        """
        :param modelpath: local path to a model
        :return: model
        """
        if modelpath in self.buffer.keys():
            return self.buffer[modelpath]
        return self.load(modelpath=modelpath)
