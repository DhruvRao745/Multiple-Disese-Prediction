import requests
import json

url = "http://localhost:5000/predict_diabetes"
headers = {"Content-Type": "application/json"}

data = {
    "Age": 45,
    "Pregnancy": 2,
    "Glucose": 120,
    "Blood Pressure": 70,
    "Skin Thickness": 20,
    "Insulin": 85,
    "Body Mass Index": 25
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
