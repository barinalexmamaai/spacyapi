from fastapi import FastAPI, Path
from scripts.tuning.textclassifier import TextClassifier


app = FastAPI()
clf = TextClassifier(modelpath="basic")


@app.get("/intent/classify/{model}")
def intentclassification(text: str,
                         model: str = Path(default=None,
                                           description="The name of a classifier you want to use")):
    return {"model": model, "intent": clf.predict(text)}
