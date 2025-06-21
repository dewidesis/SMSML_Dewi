import requests
import json

# URL untuk serving model
URL = "http://localhost:5001/invocations"

# Load data JSON
with open("MLModel/model/serving_input_example.json") as f:
    data = json.load(f)

# Kirim request
response = requests.post(URL, json=data)

# Tampilkan hasil
print("Prediction response:", response.text)
