import sys
from scripts.tuning.textclassifier import TextClassifier


def classificationloop(modelpath: str):
    """
    Read user input and make classification

    :param modelpath: path to a stored fine tuned model
    :return:
    """
    try:
        classifier = TextClassifier(modelpath=modelpath)
    except:
        sys.exit("ERROR WHILE LOADING THE CLASSIFIER")
    print("TO EXIT THE PROGRAM TYPE 0\n"
          "ENTER NEW TEXT TO CLASSIFY:")
    userinput = input()
    while userinput != "0":
        prediction = classifier.predict(text=userinput)
        print("PREDICTION:", prediction)
        print("ENTER NEW TEXT TO CLASSIFY:")
        userinput = input()

