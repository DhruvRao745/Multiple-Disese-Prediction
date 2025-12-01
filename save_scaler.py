import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

parkinsons_data = pd.read_csv('C:/Users/HP/OneDrive/Desktop/MDP/Data/Parkinson Disease Prediction/parkinsons.csv')
x = parkinsons_data.drop(columns=['name','status'],axis=1)
y = parkinsons_data['status']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=2)
scaler = StandardScaler()
scaler.fit(x_train)
joblib.dump(scaler, 'parkinsons_scaler.pkl')
