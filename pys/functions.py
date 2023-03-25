

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

