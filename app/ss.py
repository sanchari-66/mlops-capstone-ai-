from pypmml import Model
model = Model.load(r"C:\Users\basin\Dropbox\capstone_project_5\mlops-capstone-ai\src\model.pmml")

input_data = {"input": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]}
prediction = model.predict(input_data)
print(prediction)
