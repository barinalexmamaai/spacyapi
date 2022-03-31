import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from transformers import AutoTokenizer
from scripts.constants import DATA_DIR
import numpy as np


def applylimit(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """
    :param df: data frame with 'label' column
    :param n: min number of samples required
    :return: data frame with minimum number of samples per label
    """
    dfcounts = df.groupby('label').size().reset_index(name='counts')
    dfmincounts = dfcounts[dfcounts.counts > n]
    return df[df.label.isin(dfmincounts.label.tolist())]


def encodelabels(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: data frame with 'label' column
    :return: data frame with 'intlabel' column containing encoded labels
    """
    df['intlabel'] = df['label'].rank(method='dense', ascending=False).astype(int) - 1
    return df


def loadpreprocesseddata(path: str) -> pd.DataFrame:
    """
    :param path: absolute path to a csv file with 'label' column
    :return: data frame with labels mapped to
        integer values in the intlabel column
    """
    df = pd.read_csv(path)
    df = applylimit(df=df, n=2)
    df = encodelabels(df=df)
    return df


def getmapping(data: pd.DataFrame) -> (dict, dict):
    """
    :param data: data frame with 'intlabel' columns containing int values
        and 'label' column containing str values
    :return: mappings from int values to str and reverse
    """
    id2label = {}
    label2id = {}
    for key in data.intlabel.unique():
        value = data.loc[data['intlabel'] == key, 'label'].unique()[0]
        id2label[f"{key}"] = value
        label2id[value] = f"{key}"
    return id2label, label2id


def splitdata(data: pd.DataFrame, ratio: float = 0.2) -> dict:
    """
    :param data: data frame with 'text' and 'intlabel' columns
    :param ratio: ratio of a test set to a data set
    :return: train and test data sets
    """
    texts = data.text.tolist()
    labels = data.intlabel.tolist()
    trntxt, tsttxt, trnlbl, tstlbl = train_test_split(texts, labels, test_size=ratio)
    return {"train": {"text": trntxt, "label": trnlbl},
            "test": {"text": tsttxt, "label": tstlbl}}


def balancedata(data: dict) -> dict:
    """
    :param data: dictionary with 'text' and 'label' keys
    :return: balanced dataset
    """
    sampler = RandomOverSampler(random_state=42)
    txt = np.asarray(data["text"])
    txt = txt[:, np.newaxis]
    txt, lbl = sampler.fit_resample(txt, data["label"])
    txt = txt.flatten().tolist()
    return {"text": txt, "label": lbl}


def encodefeatures(data: dict, tokenizer) -> list:
    """
    :param data: dictionary with 'text' and 'label' keys
    :param tokenizer: encode text into vectors with integer values
    :return: list of dicts with encoded data
    """
    encodings = tokenizer(data["text"], truncation=True, padding=True)
    zipped = zip(data["label"], encodings['input_ids'], encodings['attention_mask'])
    return [{'label': label,
             'input_ids': input_id,
             'attention_mask': attention_mask} for label, input_id, attention_mask in zipped]


def countlabels(data: dict) -> pd.DataFrame:
    """
    :param data: dictionary with a 'label' key and one feature key
    :return: data frame with 'counts' column containing
        number of samples per label
    """
    df = pd.DataFrame(data)
    return df.groupby('label').size().reset_index(name='counts')


class DataManager:
    def __init__(self, path: str, tokenizer):
        """
        :param path: relative to a data folder path to a csv file with two columns 'text', 'label'
        :param tokenizer: encode text into vectors with integer values.
            loaded with from_pretrained() function for a model that is about to be tuned
        """
        self.data = loadpreprocesseddata(path=f"{DATA_DIR}/{path}")
        self.id2label, self.label2id = getmapping(data=self.data)
        self.nlabels = len(self.id2label.values())
        self.tokenizer = tokenizer
        self.datasets = {}
        self.trainset = []
        self.testset = []
        self.resamplesets()

    def reloaddata(self, path: str):
        """
        Reload and preprocess again raw data

        :param path: relative to a data folder path to a csv file with two columns 'text', 'label'
        """
        self.data = loadpreprocesseddata(path=f"{DATA_DIR}/{path}")
        self.labelmapping = getmapping(data=self.data)
        self.nlabels = len(self.id2label.values())
        self.resamplesets()

    def resamplesets(self):
        """
        Randomly split data into train and test sets
        """
        self.datasets = splitdata(data=self.data)
        self.datasets["train"] = balancedata(data=self.datasets["train"])
        self.trainset = encodefeatures(data=self.datasets["train"], tokenizer=self.tokenizer)
        self.testset = encodefeatures(data=self.datasets["test"], tokenizer=self.tokenizer)

    def getdistribution(self, name: str):
        """
        :param name: name of a subset: train/test
        :return: data frame containing number of
            samples per label in a train dataset
        """
        return countlabels(data=self.datasets[name])


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Seznam/small-e-czech")
    dm = DataManager(path=f"banking.csv", tokenizer=tokenizer)
