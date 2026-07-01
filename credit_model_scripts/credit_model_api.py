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
    predictions : dict

    
#create fitpredict path
@app.post('/predict', status_code=200)
def predict(data:Features):
    #create response dictionary
    response = data.dict()
    try:
        #features
        features = np.array([[data.index,
        data.limit_bal,
        data.sex,
        data.education,
        data.marriage,
        data.age,
        data.payment_status_sep,
        data.payment_status_aug,
        data.payment_status_jul,
        data.payment_status_jun,
        data.payment_status_may,
        data.payment_status_apr,
        data.bill_statement_sep,
        data.bill_statement_aug,
        data.bill_statement_jul,
        data.bill_statement_jun,
        data.bill_statement_may,
        data.bill_statement_apr,
        data.previous_payment_sep,
        data.previous_payment_aug,
        data.previous_payment_jul,
        data.previous_payment_jun,
        data.previous_payment_may,
        data.previous_payment_apr]])
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
        response['prediction'] = predictions

    except Exception as e:
        response['success'] = False
        response['message'] = str(e)
        response['prediction'] = {}
        
    return response
        
            
        


    
    
    
        
    

    
    