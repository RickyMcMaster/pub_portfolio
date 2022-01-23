import pickle
import numpy as np

from flask import Flask, request, jsonify


dv_file = 'dv.bin'
model_file = 'houses-model.bin'

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('houses_service')

def predict_single(house, dv, model):
    X = dv.transform([house])
    y_pred = model.predict(X)[:, 1]
    return y_pred[0]

@app.route('/predict', methods=['POST'])
def predict():
    house = request.get_json()

    prediction = predict_single(house, dv, model)
    price = prediction
    
    result = {
        'price': float(price),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)