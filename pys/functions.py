from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from pymongo import MongoClient
import requests

import credentials
import numpy as np
import pandas as pd
import json
import urllib
import time

def train_classification_models(X_train, y_train, X_test, y_test):

    '''Trains diferent clasification models'''

    models = [GaussianNB(), LogisticRegression(),DecisionTreeClassifier(), RandomForestClassifier(), SVC(), AdaBoostClassifier(), GradientBoostingClassifier()]
        
    metrics = list()

    for model in models:
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)

        js = jaccard_score(y_test, yhat, average = "macro")
        acc = accuracy_score(y_test, yhat)
        precision = precision_score(y_test, yhat, average = "macro")
        recall = recall_score(y_test, yhat, average = "macro")
        f1 = f1_score(y_test, yhat, average = "macro")

    
        metrics.append([str(model), js, acc, precision, recall, f1, model])

    df_metrics = pd.DataFrame(data = metrics, columns = ["model_name", "jaccard_score", "accuracy_score", "precision_score", "recall_score", "f1_score", "model"])
    df_metrics.sort_values(by = "accuracy_score", ascending = False, inplace= True)

    return df_metrics

def cosmos_request(cliente):

    client = MongoClient(credentials.cosmos)

    db = client.db 

    clients_collection = db.clients

    model_input = clients_collection.find_one({"cliente":cliente})

    if type(model_input) != dict:

        return None
    
    else:

        del model_input["_id"]
        del model_input["cliente"]

    return model_input

def scale_model_input(model_input, scaler):

    input = []

    for i in model_input:

        input.append(model_input[i])

    input = np.array([input])

    scaled_input = scaler.transform(input)

    for enum, i in enumerate(model_input):
        
        model_input[i] = scaled_input.tolist()[0][enum]

    return model_input


def mls_model_request(scaled_input):

    data = {"Inputs": {"input1" : [scaled_input]}, "GlobalParameters":{}}

    body = str.encode(json.dumps(data))

    url = credentials.mls_predict_rest_endpoint
    # Replace this with the primary/secondary key or AMLToken for the endpoint
    api_key = credentials.mls_predict_api_key

    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()
        
        result = json.loads(result.decode("utf-8"))

        result = result["Results"]["WebServiceOutput0"][0]["Fixed Deposit Prediction"]

        if result == 0.0:

            output = "This client wouldn't contract the fixed deposit"

        else:

            output = "This client would contract the fixed deposit!"

        return output


    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

def handmade_model_request(scaled_input):

    body = str.encode(json.dumps(scaled_input))

    url = credentials.azure_handmade_predict_rest_endpoint
    # Replace this with the primary/secondary key or AMLToken for the endpoint
    api_key = credentials.azure_handmade_predict_api_key

    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")


    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    req = urllib.request.Request(url, body, headers)

    try:
        response = urllib.request.urlopen(req)

        result = response.read()
        return(json.loads(result.decode('utf-8')))

    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

