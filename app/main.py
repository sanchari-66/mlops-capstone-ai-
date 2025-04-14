from fastapi import FastAPI, HTTPException

import logging
import sys
import os

# Add src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), r"C:\Users\basin\Dropbox\capstone_project_5\mlops-capstone-ai\src")))

# Import the prediction function
from predict import prediction
from schemas import InputData

import logging
logging.getLogger("py4j").setLevel(logging.WARNING)

#logging.basicConfig(level=logging.DEBUG)
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the FastAPI application!"}

@app.post("/predict")
def predict_endpoint(data: InputData):
    try:
        logging.debug(f"Received input: {data.features}")
        result = prediction(data)
        return {"prediction": result}
    except HTTPException as http_exc:
        logging.error(f"HTTPException: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")