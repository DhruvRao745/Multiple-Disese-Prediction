import requests
import json

url_base = "http://localhost:5000"
headers = {"Content-Type": "application/json"}

# Test diabetes
diabetes_data = {
    "Age": 45,
    "Pregnancy": 2,
    "Glucose": 120,
    "Blood Pressure": 70,
    "Skin Thickness": 20,
    "Insulin": 85,
    "Body Mass Index": 25
}

response = requests.post(f"{url_base}/predict_diabetes", headers=headers, data=json.dumps(diabetes_data))
print("Diabetes Status Code:", response.status_code)
print("Diabetes Response:", response.json())

# Test heart
heart_data = {
    "age": 50,
    "sex": 1,
    "cp": 2,
    "trestbps": 130,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 0,
    "thal": 2
}

response = requests.post(f"{url_base}/predict_heart", headers=headers, data=json.dumps(heart_data))
print("Heart Status Code:", response.status_code)
print("Heart Response:", response.json())

# Test parkinsons (dummy data, since many features)
parkinsons_data = {f"feature_{i}": 0.5 for i in range(22)}

response = requests.post(f"{url_base}/predict_parkinsons", headers=headers, data=json.dumps(parkinsons_data))
print("Parkinsons Status Code:", response.status_code)
print("Parkinsons Response:", response.json())

# Test invalid data
print("\nTesting invalid data:")

# Missing field
invalid_data = {
    "Age": 45,
    "Pregnancy": 2,
    "Glucose": 120
    # Missing others
}

response = requests.post(f"{url_base}/predict_diabetes", headers=headers, data=json.dumps(invalid_data))
print("Invalid Diabetes Status Code:", response.status_code)
print("Invalid Diabetes Response:", response.json())

# Wrong type
wrong_type_data = {
    "Age": "forty-five",
    "Pregnancy": 2,
    "Glucose": 120,
    "Blood Pressure": 70,
    "Skin Thickness": 20,
    "Insulin": 85,
    "Body Mass Index": 25
}

response = requests.post(f"{url_base}/predict_diabetes", headers=headers, data=json.dumps(wrong_type_data))
print("Wrong Type Diabetes Status Code:", response.status_code)
print("Wrong Type Diabetes Response:", response.json())
