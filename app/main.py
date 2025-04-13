from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pypmml import Model
import numpy as np

app = FastAPI()

# Load the model
try:
    model = Model.load(r"C:\Users\basin\Dropbox\capstone_project_5\mlops-capstone-ai\src\model.pmml")  # Replace with correct path
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

class InputData(BaseModel):
    features: list[float]  # Specify that the list should contain floats

expected_feature_count = 10  # Replace with your model's expected number of features

@app.post("/predict")
def predict(data: InputData):
    try:
        # Validate input length
        if len(data.features) != expected_feature_count:
            raise HTTPException(status_code=400, detail="Invalid input dimensions")

        # Prepare the input array
        input_array = np.array(data.features).reshape(1, -1)

        # Perform prediction
        prediction = model.predict({"input": input_array.tolist()})
        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")