
# 1. Import Libraries
import uvicorn
import pickle
from BankNotes import BankNote
from fastapi import FastAPI
import numpy as np
import pandas as pd

# 2. Create the app object
predict_app = FastAPI()
pickle_in = open("classifier.pickle","rb")
classifier = pickle.load(pickle_in)

# 3. Index route
@predict_app.get('/')
def index():
    return{'message':'Hello Amrit'}

# 4. Route with single parameter , returns parameter within a message 
@predict_app.get('/{name}')
def get_name(name:str):
    return{'message':f'Hello ,{name}'}

# 5. Exposet the Prediction functionality , make a prediction from the passed
# JSON data and return the predicted Bank Note with the confidence score
@predict_app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
    print(data)
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']
    print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction = 'Fake Note'
    else:
        prediction = 'Genuine Bank Note'
    return {'prediction':prediction}

#6. Run the API with uvicorn; will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(predict_app,host='127.0.0.1',port=8000)

# uvicorn banknote_authetication_prediction_model_FastAPI:predict_app --reload