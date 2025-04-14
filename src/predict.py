from fastapi import FastAPI, HTTPException

from pypmml import Model
from train import preprocess_data 
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
from schemas import InputData  # Import InputData from schemas.py


# Load the PMML model
model = Model.load(r"C:\Users\basin\Dropbox\capstone_project_5\mlops-capstone-ai\src\model.pmml")

import pandas as pd

def prediction(data: InputData):
    try:
        
        logging.debug(f"Received input: {data.features}")
        
        # Preprocess the input data using the function from train.py
        preprocessed_data = preprocess_data(data.features)


        # Validate input length
        if len(preprocessed_data.features) != expected_feature_count:
            raise HTTPException(status_code=400, detail="Invalid input dimensions")
        
        if not all(np.isfinite(preprocessed_data.features)):
            logging.error("Input contains NaN or infinite values")
            raise HTTPException(status_code=400, detail="Input contains NaN or infinite values")
        # Prepare the input array
        input_array = np.array(preprocessed_data.features).reshape(1, -1)
        logging.debug(f"Input array for model: {input_array}")

        # Perform prediction
        prediction = model.predict({"input": input_array.tolist()})
        
        if any(not np.isfinite(val) for val in result.values()):
            logging.error("Model returned NaN or infinite values")
            raise HTTPException(status_code=500, detail="Model returned NaN or infinite values")
        
        logging.debug(f"Prediction result: {prediction}")
        return {"prediction": prediction}

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")







# Load the preprocessed data
#company_data_processed = pd.read_csv(r"C:\Users\basin\Dropbox\capstone_project_5\mlops-capstone-ai\data\company_data_processed.csv")
#company_data = company_data_processed.drop("Valuation.B.", axis = 1)

# Check the data
#print(company_data_processed.head())

#input_data = company_data.iloc[1].to_dict()  # Convert the first row to a dictionary
#print(input_data)

# Make a prediction
#prediction = model.predict(input_data)

# Print the result
#print(f"Prediction: {prediction}")