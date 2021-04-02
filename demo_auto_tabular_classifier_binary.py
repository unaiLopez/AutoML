from tabular.AutoTabularDataClassifier import AutoTabularDataClassifier

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score

import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    d = load_breast_cancer()
    y = d['target']
    X = pd.DataFrame(d['data'], columns=d['feature_names'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = AutoTabularDataClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    balanced_accuracy = balanced_accuracy_score(y_test, predictions)

    print()
    print('Best Model ', model.best_estimator)
    print()
    print('Best Pipeline ', model.best_pipeline)
    print()
    print('Balanced Accuracy ', balanced_accuracy)