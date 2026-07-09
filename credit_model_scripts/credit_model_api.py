from fastapi import FastAPI
from pydantic import BaseModel
from Scoring_model2 import model as model2
import os
import numpy as np
import pandas as pd

#instantiate fastapi app
app = FastAPI()

#get method for the '/home' path showing introductory message
@app.get('/home', status_code=200)
def home():
    return {'message':'Welcome to the Credit Scoring model API'}

#create BaseModel to input features
class Features(BaseModel):
    index : int
    limit_bal : int
    sex : str
    education : str
    marriage : str
    age : float
    payment_status_sep : str
    payment_status_aug : str
    payment_status_jul : str
    payment_status_jun : str
    payment_status_may : str
    payment_status_apr : str
    bill_statement_sep : int
    bill_statement_aug : int
    bill_statement_jul : int
    bill_statement_jun : int
    bill_statement_may : int
    bill_statement_apr : int
    previous_payment_sep : int
    previous_payment_aug : int
    previous_payment_jul : int
    previous_payment_jun : int
    previous_payment_may : int
    previous_payment_apr : int

class FitOut(Features):
    success : bool
    message : str
    predictions : list

    
#create fitpredict path
@app.post('/predict', status_code=200)
def predict(data:Features):
    #create response dictionary
    response = data.dict()
    try:
        #features
        features = pd.DataFrame([data.model_dump()])
        #instantiate models
        clf2 = model2()
        model = clf2
        filename = model.dump()
        #fit model
        model.build_model()
        #load model
        model.load()
        #get predictions
        predictions = model.get_predictions(features)
        response['success'] = True
        response['message'] = f"model succesfully trained and saved at {filename}. "
        response['prediction'] = predictions.tolist()

    except Exception as e:
        response['success'] = False
        response['message'] = str(e)
        response['prediction'] = {}
        
    return response
        
            
        


    
    
    
        
    

    
    