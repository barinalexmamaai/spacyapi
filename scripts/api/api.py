from fastapi import FastAPI, Path
from scripts.api.modelsbuffer import ModelsBuffer


app = FastAPI()
buffer = ModelsBuffer()


@app.get("/intent/{model}/classify/")
def intentclassification(text: str,
                         model: str = Path(default=None,
                                           description="The name of a classifier you want to use")):
    classifier = buffer.get(modelpath=model)
    return {"model": model, "intent": classifier.predict(text)}
