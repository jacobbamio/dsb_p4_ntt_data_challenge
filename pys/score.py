import joblib
import numpy as np
import os

def init():

    global model, scaler

    # Load model

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'handmade_model')
    model = joblib.load(model_path)
    
    # Load x_scaler

    scaler_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'handmade_x_scaler')
    scaler = joblib.load(scaler_path)


def run(raw_data):

    input = []

    for i in raw_data:

        input.append(raw_data[i])

    input = np.array([input])

    scaled_input = scaler.transform(input)

    prediction = model.predict(scaled_input)

    if int(prediction[0]) == 1:

        return "This client would contract the fixed deposit!"
    
    else:
        
        return "This client wouldn't contract the fixed deposit"

