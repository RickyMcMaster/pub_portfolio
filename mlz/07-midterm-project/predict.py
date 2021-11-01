import pickle
import numpy as np

from flask import Flask, request, jsonify


dv_file = 'dv.bin'
model_file = 'resig-model.bin'

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('resig_service')

def predict_single(employee, dv, model):
    X = dv.transform([employee])
    y_pred = model.predict(X)[:, 1]
    return y_pred[0]

@app.route('/predict', methods=['POST'])
def predict():
    employee = request.get_json()

    prediction = predict_single(employee, dv, model)
    resig = prediction
    
    result = {
        # 'resig_probability': float(prediction),
        'resig': bool(resig),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)