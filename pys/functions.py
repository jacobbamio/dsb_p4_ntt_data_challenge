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

#import credentials

import pandas as pd


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

def cosmos_request():

    credentials.cosmos

    print("Pending")


def mls_model_request():

    print("Pending")

def handmade_model_request():

    print("Pending")
