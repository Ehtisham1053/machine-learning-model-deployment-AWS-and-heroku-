# app.py

from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np

application = Flask(__name__)



# Load the preprocessor and model from the artifacts folder
artifacts_dir = 'artifacts'
with open(os.path.join(artifacts_dir, 'preprocessor.pkl'), 'rb') as f:
    preprocessor = pickle.load(f)

with open(os.path.join(artifacts_dir, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

# Define route for the home page
@application.route('/')
def index():
    return render_template('index.html')

# Define route for prediction
@application.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data
        fLength = float(request.form['fLength'])
        fWidth = float(request.form['fWidth'])
        fSize = float(request.form['fSize'])
        fConc = float(request.form['fConc'])
        fConc1 = float(request.form['fConc1'])
        fAsym = float(request.form['fAsym'])
        fM3Long = float(request.form['fM3Long'])
        fM3Trans = float(request.form['fM3Trans'])
        fAlpha = float(request.form['fAlpha'])
        fDist = float(request.form['fDist'])

        # Create an input array
        input_data = np.array([[fLength, fWidth, fSize, fConc, fConc1, fAsym, fM3Long, fM3Trans, fAlpha, fDist]])

        # Preprocess the input using the loaded preprocessor (scaling)
        input_data_scaled = preprocessor['scaler'].transform(input_data)

        # Predict using the loaded model
        prediction = model.predict(input_data_scaled)

        # Interpret the result
        if prediction == 0:
            result = 'Class G'
        else:
            result = 'Class H'

        return render_template('index.html', prediction_text=f'Predicted Class: {result}')
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    application.run(host="0.0.0.0")
