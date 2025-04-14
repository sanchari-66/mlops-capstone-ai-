import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder



import pandas as pd
import requests

# Load the dataset

# data = pd.read_csv(r"C:\Users\basin\Dropbox\capstone_project_5\mlops-capstone-ai\data\unicorn.csv") 
# # Send one row at a time to the FastAPI endpoint
# url = "http://127.0.0.1:8000/predict"

# for _, row in data.iterrows():
#     input_data = {"features": row.to_dict()}
#     response = requests.post(url, json=input_data)
#     print(response.json())

def preprocess_data(input_data: dict) -> dict:
    df = pd.DataFrame([input_data])

    non_numeric_columns = df.select_dtypes(include=['object']).columns

    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_features = pd.DataFrame(encoder.fit_transform(df[non_numeric_columns]),
                                columns=encoder.get_feature_names_out(non_numeric_columns))

# Drop original non-numeric columns and add encoded ones
    df = pd.concat([df.drop(non_numeric_columns, axis=1), encoded_features], axis=1)
    return df.iloc[0].to_dict()


# Step 1: Load the dataset
#company_data = pd.read_csv(r"C:\Users\basin\Dropbox\capstone_project_5\mlops-capstone-ai\data\unicorn.csv")  # Replace 'company_data.csv' with your actual file name

# Step 2: Convert Non-Numeric Columns
# Identify and encode non-numeric columns using OneHotEncoder

#company_data_encoded.to_csv('data/updated_company_data_processed.csv', index=False)