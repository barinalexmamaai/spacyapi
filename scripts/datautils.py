from spacy.tokens import DocBin


def makedocs(data: list, labels: list, package):
    """
    :param data: tuples of text with labels
    :param labels: list of existing labels
    :param package: spacy language package
    :return: list of spacy doc files
    """
    docs = []
    for doc, label in package.pipe(data, as_tuples=True):
        for key in labels:
            doc.cats[key] = int(label == key)
        docs.append(doc)
    return (docs)


def datatodocbin(data: list, labels: list, path: str, package):
    """
    :param data: tuples of text with labels
    :param labels: list of existing labels
    :param path: where to store docbin
    :param package: spacy language package
    """
    docs = makedocs(data, labels, package)
    docbin = DocBin(docs=docs)
    docbin.to_disk(path)
