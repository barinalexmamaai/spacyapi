from scripts.berttuning.finetuner import FineTuner
from scripts.berttuning.datautils import loadconfig
from scripts.constants import CONFIG_DIR
import numpy as np


class Evaluator:
    def __init__(self):
        self.trials = []
        config = loadconfig(path=f"{CONFIG_DIR}/finetune.yaml")
        self.tuner = FineTuner(config=config)

    def evaluate(self, batch: list) -> dict:
        """
        Train and evaluate fine tuned model
        :param batch: list of dictionaries representing encoded samples
        :return: dictionary with results
        """
        self.tuner.train()
        predictions = self.tuner.predictbatch(batch=batch)
        groundtruth = np.array([entry['label'] for entry in batch])
        correct = np.sum(predictions == groundtruth)
        accuracy = correct / groundtruth.shape[0]
        return {"accuracy": accuracy,
                "correct": correct,
                "total": groundtruth.shape[0],
                "predicted": predictions,
                "groundtruth": groundtruth}

    def processresults(self) -> dict:
        """
        :return: mean and total results respective to the metrics for all trials combined
        """
        results = {"accuracy": 0, "correct": 0, "total": 0}
        for trial in self.trials:
            for key in results.keys():
                results[key] += trial[key]
        results["accuracy"] /= len(self.trials)
        return results

    def runevaluation(self, n: int = 5) -> dict:
        """
        Run evaluations n times and return mean score

        :param n: number of evaluation iterations
        :return: dictionary with results
        """
        for i in range(n):
            self.trials.append(self.evaluate(batch=self.tuner.dm.testset))
            print(f"TRIAL {i}; ACCURACY: {self.trials[-1]['accuracy']}")
            self.tuner.resample()
            self.tuner.reloadmodel()
        return self.processresults()


if __name__ == "__main__":
    evaluator = Evaluator()
    results = evaluator.runevaluation(n=3)
    print(results)
