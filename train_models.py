import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# Generate dummy data for diabetes
# Features: Age, Pregnancy, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI
np.random.seed(42)
n_samples = 1000

diabetes_data = {
    'Age': np.random.randint(20, 80, n_samples),
    'Pregnancy': np.random.randint(0, 10, n_samples),
    'Glucose': np.random.randint(70, 200, n_samples),
    'Blood Pressure': np.random.randint(60, 120, n_samples),
    'Skin Thickness': np.random.randint(10, 50, n_samples),
    'Insulin': np.random.randint(15, 300, n_samples),
    'Body Mass Index': np.random.uniform(18, 40, n_samples),
    'Outcome': np.random.randint(0, 2, n_samples)  # 0 or 1
}

df_diabetes = pd.DataFrame(diabetes_data)
X_diabetes = df_diabetes.drop('Outcome', axis=1)
y_diabetes = df_diabetes['Outcome']

# Train diabetes model
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)
diabetes_model = LogisticRegression(random_state=42)
diabetes_model.fit(X_train_d, y_train_d)
joblib.dump(diabetes_model, 'diabetes_model.pkl')

# Generate dummy data for heart disease
# Assuming features like age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
# For simplicity, use numerical features
heart_data = {
    'age': np.random.randint(29, 77, n_samples),
    'sex': np.random.randint(0, 2, n_samples),
    'cp': np.random.randint(0, 4, n_samples),
    'trestbps': np.random.randint(94, 200, n_samples),
    'chol': np.random.randint(126, 564, n_samples),
    'fbs': np.random.randint(0, 2, n_samples),
    'restecg': np.random.randint(0, 3, n_samples),
    'thalach': np.random.randint(71, 202, n_samples),
    'exang': np.random.randint(0, 2, n_samples),
    'oldpeak': np.random.uniform(0, 6.2, n_samples),
    'slope': np.random.randint(0, 3, n_samples),
    'ca': np.random.randint(0, 4, n_samples),
    'thal': np.random.randint(0, 3, n_samples),
    'target': np.random.randint(0, 2, n_samples)
}

df_heart = pd.DataFrame(heart_data)
X_heart = df_heart.drop('target', axis=1)
y_heart = df_heart['target']

# Train heart model
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)
heart_model = RandomForestClassifier(random_state=42)
heart_model.fit(X_train_h, y_train_h)
joblib.dump(heart_model, 'Heart_model.pkl')

# Generate dummy data for Parkinson's
# Features like MDVP:Fo(Hz), MDVP:Fhi(Hz), etc. - many features
# For simplicity, use a subset
parkinsons_data = {}
for i in range(22):  # Assume 22 features
    parkinsons_data[f'feature_{i}'] = np.random.uniform(0, 1, n_samples)
parkinsons_data['status'] = np.random.randint(0, 2, n_samples)

df_parkinsons = pd.DataFrame(parkinsons_data)
X_parkinsons = df_parkinsons.drop('status', axis=1)
y_parkinsons = df_parkinsons['status']

# Train parkinsons model
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_parkinsons, y_parkinsons, test_size=0.2, random_state=42)
parkinsons_model = SVC(random_state=42)
parkinsons_model.fit(X_train_p, y_train_p)
joblib.dump(parkinsons_model, 'Parkinson_model.pkl')

print("Models trained and saved successfully!")
