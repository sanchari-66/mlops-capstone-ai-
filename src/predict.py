from pypmml import Model

# Load the PMML model
model = Model.load(r"C:\Users\basin\Dropbox\capstone_project_5\mlops-capstone-ai\src\model.pmml")

import pandas as pd

# Load the preprocessed data
company_data_processed = pd.read_csv(r"C:\Users\basin\Dropbox\capstone_project_5\mlops-capstone-ai\data\company_data_processed.csv")


# Check the data
print(company_data_processed.head())

input_data = company_data_processed.iloc[0].to_dict()  # Convert the first row to a dictionary

# Make a prediction
prediction = model.predict(input_data)

# Print the result
print(f"Prediction: {prediction}")