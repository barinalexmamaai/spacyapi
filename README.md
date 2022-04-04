# Intent classification with fine tuned BERT models

## TUNING
### CLI
Default tuning parameters are specified in the config file default.yaml.
To run the tuning process use the command from the project's root directory:
```
python -m scripts.cli.main -t
```
All the training parameters are optional and could be specified as following:
```
python -m scripts.cli.main -t --model="Seznam/small-e-czech" --data="banking.csv" --lrate=0.0001
```
You can always see all the available options with the help command:
```
python -m scripts.cli.main -h
```

## CLASSIFICATION

### CLI

In order to start classifying intents 
with a desired model execute the following command:
```
python -m scripts.cli.main -c "default"
```
This command will start the classification loop that requires user input
per each iteration and make a prediction using the pretuned model specified earlier.

### API
So far only classification with pretuned models is available 
and can be tested locally. First run the app:
```
python -m scripts.api.api
```
Then go to the browser and type for example:
```
http://127.0.0.1:8000/intent/default/classify/?text=ahoj
```
Where *default* is the name of a pretuned model and 
*ahoj* is the text to be classified assigned to the *text* parameter