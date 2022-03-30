from scripts.berttuning.finetuner import FineTuner
from scripts.berttuning.datautils import loadconfig
from scripts.constants import CONFIG_DIR
import numpy as np


class Evaluator:
    def __init__(self):
        self.trials = []
        config = loadconfig(path=f"{CONFIG_DIR}/finetune.yaml")
        self.tuner = FineTuner(config=config)

    def evaluate(self) -> dict:
        """
        Train and evaluate fine tuned model
        :return: dictionary with results
        """
        self.tuner.train()
        batch = self.tuner.dm.testset
        predictions = self.tuner.predictbatch(batch=batch)
        groundtruth = np.array([entry['label'] for entry in batch])
        correct = np.sum(predictions == groundtruth)
        accuracy = correct / groundtruth.shape[0]
        return {"accuracy": accuracy,
                "correct": correct,
                "total": groundtruth.shape[0],
                "predicted": predictions,
                "groundtruth": groundtruth}

    def runevaluation(self, n: int = 5) -> dict:
        """
        Run evaluations n times and return mean score

        :param n: number of evaluation iterations
        :return: dictionary with results
        """
        for i in range(n):
            self.trials.append(self.evaluate())
            print(f"TRIAL {i}; ACCURACY: {self.trials[-1]['accuracy']}")
            self.tuner.resample()
            self.tuner.reloadmodel()
        results = {"accuracy": 0, "correct": 0, "total": 0}
        for trial in self.trials:
            for key in results.keys():
                results[key] += trial[key]
        results["accuracy"] /= n
        return results


if __name__ == "__main__":
    evaluator = Evaluator()
    results = evaluator.runevaluation(n=3)
    print(results)
