import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Step 1: Load the dataset
company_data = pd.read_csv(r"C:\Users\basin\Dropbox\capstone_project_5\mlops-capstone-ai\data\unicorn.csv")  # Replace 'company_data.csv' with your actual file name

# Step 2: Convert Non-Numeric Columns
# Identify and encode non-numeric columns using OneHotEncoder
non_numeric_columns = company_data.select_dtypes(include=['object']).columns

encoder = OneHotEncoder(drop='first', sparse=False)
encoded_features = pd.DataFrame(encoder.fit_transform(company_data[non_numeric_columns]),
                                columns=encoder.get_feature_names_out(non_numeric_columns))

# Drop original non-numeric columns and add encoded ones
company_data_encoded = pd.concat([company_data.drop(non_numeric_columns, axis=1), encoded_features], axis=1)

company_data_encoded.to_csv('data/updated_company_data_processed.csv', index=False)