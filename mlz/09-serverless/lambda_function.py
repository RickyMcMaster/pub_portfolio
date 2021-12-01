import json
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor


model = 'cats-dogs-v2.tflite'
preprocessor = create_preprocessor('xception', target_size=(150, 150))

interpreter = tflite.Interpreter(model_path=model)
interpreter.allocate_tensors()


input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def predict(url):
    X = preprocessor.from_url(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_index)
    return pred[0].tolist()


def lambda_handler(event, context):
    # print("parameters:", event)
    url = event['url']
    results = predict(url)
    return results
