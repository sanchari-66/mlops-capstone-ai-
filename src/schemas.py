from pydantic import BaseModel

class InputData(BaseModel):
    features: dict  # Ensure input is a dictionary with feature names and values
    
    