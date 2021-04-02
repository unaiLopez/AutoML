from tabular.AutoTabularDataRegressor import AutoTabularDataRegressor

from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    boston = load_boston()

    y = boston['target']
    X = pd.DataFrame(boston['data'], columns=boston.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = AutoTabularDataRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)

    print()
    print('Best Model ', model.best_estimator)
    print()
    print('Best Pipeline ', model.best_pipeline)
    print()
    print('Mean Squared Error ', mse)
