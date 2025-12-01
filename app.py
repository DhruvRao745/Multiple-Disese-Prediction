"""
Flask API for Disease Prediction

This API provides endpoints for predicting diabetes, heart disease, and Parkinson's disease
using pre-trained machine learning models.
"""

from flask import Flask, request, jsonify, send_file
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load pre-trained models
diabetes_model = joblib.load('diabetes_model.pkl')
heart_model = joblib.load('Heart_model.pkl')
parkinsons_model = joblib.load('Parkinsons_Model.pkl')
parkinsons_scaler = joblib.load('parkinsons_scaler.pkl')

# Routes to serve HTML pages
@app.route('/')
def home():
    return send_file('FrontPage.Html')

@app.route('/diabetes')
def diabetes_page():
    return send_file('Diabetes.html')

@app.route('/heart')
def heart_page():
    return send_file('Heart.html')

@app.route('/parkinsons')
def parkinsons_page():
    return send_file('parkinsons.html')

# Routes to serve CSS files
@app.route('/frontpage.css')
def frontpage_css():
    return send_file('frontpage.css')

@app.route('/Diabetes.css')
def diabetes_css():
    return send_file('Diabetes.css')

@app.route('/heart.css')
def heart_css():
    return send_file('heart.css')

@app.route('/Heart.html')
def heart_html_redirect():
    return '', 404

@app.route('/parkinsons.css')
def parkinsons_css():
    return send_file('parkinsons.css')

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    """
    Predict diabetes based on input features.

    Expects a JSON payload with feature values.
    Returns a JSON response with the prediction (0 or 1).
    """
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid input. Please provide a JSON object with feature values.'}), 400

        if not hasattr(diabetes_model, 'predict'):
            return jsonify({'error': 'Model not loaded properly. Please check the model file.'}), 500

        input_data = pd.DataFrame([data])
        result = diabetes_model.predict(input_data)
        return jsonify({'prediction': int(result[0])})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    """
    Predict heart disease based on input features.

    Expects a JSON payload with feature values.
    Returns a JSON response with the prediction (0 or 1).
    """
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid input. Please provide a JSON object with feature values.'}), 400

        if not hasattr(heart_model, 'predict'):
            return jsonify({'error': 'Model not loaded properly. Please check the model file.'}), 500

        input_data = pd.DataFrame([data])
        result = heart_model.predict(input_data)
        return jsonify({'prediction': int(result[0])})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict_parkinsons', methods=['POST'])
def predict_parkinsons():
    """
    Predict Parkinson's disease based on input features.

    Expects a JSON payload with feature values.
    Returns a JSON response with the prediction (0 or 1).
    """
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid input. Please provide a JSON object with feature values.'}), 400

        if not hasattr(parkinsons_model, 'predict'):
            return jsonify({'error': 'Model not loaded properly. Please check the model file.'}), 500

        # Map the incoming feature names to the expected order for the model
        feature_order = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
            'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
            'spread1', 'spread2', 'D2', 'PPE'
        ]

        # Extract values in the correct order
        input_values = [data[feat] for feat in feature_order]

        # Standardize the input
        input_array = np.array([input_values])
        std_input = parkinsons_scaler.transform(input_array)

        result = parkinsons_model.predict(std_input)
        return jsonify({'prediction': int(result[0])})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/favicon.ico')
def favicon():
    return '', 404

if __name__ == '__main__':
    app.run(debug=True)
