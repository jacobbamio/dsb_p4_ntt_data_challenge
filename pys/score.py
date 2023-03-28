import joblib
import numpy as np
import os
import json

def init():

    global model, scaler

    # Load model

    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)


def run(data):

    l = []

    scaled_input = json.loads(data)

    for i in scaled_input:

        l.append(scaled_input[i])

    input = np.array([l])

    print(model, type(model))

    prediction = model.predict(input)

    print("Prediction done")

    if int(prediction[0]) == 1:

        return "This client would contract the fixed deposit!"
    
    else:
        
        return "This client wouldn't contract the fixed deposit"
    
    

