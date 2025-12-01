import requests
import json

url = "http://localhost:5000/predict_parkinsons"
headers = {"Content-Type": "application/json"}

# Example data from the notebook (should predict 0)
data = {
    'MDVP:Fo(Hz)': 197.076,
    'MDVP:Fhi(Hz)': 206.896,
    'MDVP:Flo(Hz)': 192.055,
    'MDVP:Jitter(%)': 0.00289,
    'MDVP:Jitter(Abs)': 0.00001,
    'MDVP:RAP': 0.00166,
    'MDVP:PPQ': 0.00168,
    'Jitter:DDP': 0.00498,
    'MDVP:Shimmer': 0.01098,
    'MDVP:Shimmer(dB)': 0.097,
    'Shimmer:APQ3': 0.00563,
    'Shimmer:APQ5': 0.0068,
    'MDVP:APQ': 0.00802,
    'Shimmer:DDA': 0.01689,
    'NHR': 0.00339,
    'HNR': 26.775,
    'RPDE': 0.422229,
    'DFA': 0.741367,
    'spread1': -7.3483,
    'spread2': 0.177551,
    'D2': 1.743867,
    'PPE': 0.085569
}

response = requests.post(url, headers=headers, data=json.dumps(data))

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
