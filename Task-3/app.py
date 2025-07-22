# app.py

from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load saved model
model = pickle.load(open('model.pkl', 'rb'))

# Create app
app = Flask(__name__)

@app.route('/')
def home():
    return "Iris Classifier API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_features = [data['sepal_length'], data['sepal_width'],
                      data['petal_length'], data['petal_width']]
    prediction = model.predict([input_features])
    return jsonify({'species': str(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
