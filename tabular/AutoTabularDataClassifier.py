import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, SGDRegressor, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC

from tabular.AutoTabularData import AutoTabularData

class AutoTabularDataClassifier(AutoTabularData):
    
    def __init__(self, scoring_function='balanced_accuracy', n_jobs=-1, n_iterations=50, cv=5):
        super().__init__(scoring_function, n_jobs, n_iterations, cv)

    def _clean_transform_data(self, X_train):
        super()._clean_transform_data(X_train)
    
    def _encode_target_categorical_variables(self, y_train):
        #Convert categorical variables to numerical values if the problem type is classification
        le = LabelEncoder()
        
        return le.fit_transform(y_train)

    def _model_selector_hyperparameter_tuning(self, X_train, y_train):
        optimization_grid = list()

        scalers = [RobustScaler(),StandardScaler(),MinMaxScaler()]
        clean_strategies = ['mean', 'median']

        #K-nearest neighbors
        optimization_grid.append({
            'preprocessor__numerical__scaler': scalers,
            'preprocessor__numerical__cleaner__strategy': clean_strategies,
            'estimator': [KNeighborsClassifier()],
            'estimator__weights': ['uniform','distance'],
            'estimator__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'estimator__leaf_size': np.arange(20, 50, 4),
            'estimator__metric': ['euclidean', 'manhattan', 'minkowski'],
            'estimator__n_neighbors': np.arange(1, 40, 5)
        })  

        #Random Forest
        optimization_grid.append({
            'preprocessor__numerical__scaler': [None],
            'preprocessor__numerical__cleaner__strategy': clean_strategies,
            'estimator': [RandomForestClassifier(random_state=0)],
            'estimator__n_estimators': np.arange(100, 1000, 10),
            'estimator__criterion':['gini','entropy'],
            'estimator__min_samples_split': np.arange(2, 10, 4),
            'estimator__min_samples_leaf': np.arange(1, 5, 4)
        })


        #Gradient boosting
        optimization_grid.append({
            'preprocessor__numerical__scaler': [None],
            'preprocessor__numerical__cleaner__strategy': clean_strategies,
            'estimator': [GradientBoostingClassifier(random_state=0)],
            'estimator__n_estimators': np.arange(5,500,10),
            'estimator__learning_rate': np.linspace(0.1, 0.9, 20),
            'estimator__criterion': ['friedman_mse', 'mse']
        })

        #Ada boosting
        optimization_grid.append({
            'preprocessor__numerical__scaler': [None],
            'preprocessor__numerical__cleaner__strategy': clean_strategies,
            'estimator': [AdaBoostClassifier(random_state=0)],
            'estimator__n_estimators': np.arange(50, 1000, 10),
            'estimator__learning_rate': np.linspace(0.01, 0.9, 20),
            'estimator__algorithm': ['SAMME', 'SAMME.R']
        })

        #Stochastic Gradient Descent
        optimization_grid.append({
            'preprocessor__numerical__scaler': scalers,
            'preprocessor__numerical__cleaner__strategy': clean_strategies,
            'estimator': [SGDClassifier(random_state=0)],
            'estimator__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
            'estimator__penalty': ['l2', 'l1', 'elasticnet'],
            'estimator__alpha': np.linspace(1e-5, 1e-2, 6)
        })

        n_classes = len(y_train.unique())

        if n_classes == 2:

            #Logistic regression
            optimization_grid.append({
                'preprocessor__numerical__scaler': scalers,
                'preprocessor__numerical__cleaner__strategy': clean_strategies,
                'estimator':[LogisticRegression()],
                'estimator__C': np.arange(0.1, 1, 3),
                'estimator__solver': ['sag', 'saga', 'lbfgs']
            })

            #SVM
            optimization_grid.append({
                'preprocessor__numerical__scaler': scalers,
                'preprocessor__numerical__cleaner__strategy': clean_strategies,
                'estimator': [SVC()],
                'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'estimator__gamma': ['scale', 'auto'],
                'estimator__C': np.arange(0.1, 1, 3)
            })

        search = RandomizedSearchCV(
            self.model_pipeline,
            optimization_grid,
            n_iter=self.n_iterations,
            scoring=self.scoring_function, 
            n_jobs=self.n_jobs, 
            random_state=0, 
            verbose=3,
            cv=self.cv
        )
        
        search.fit(X_train, y_train)
        self.best_estimator = search.best_estimator_
        self.best_pipeline = search.best_params_

    def fit(self, X_train, y_train):
        X_train = self.source_data_adapter.adapt_source_data(X_train)
        y_train = self.source_data_adapter.adapt_source_data(y_train)

        self._clean_transform_data(X_train)
        train_y = self._encode_target_categorical_variables(y_train)
        self._model_selector_hyperparameter_tuning(X_train, y_train)
    
    def predict(self, X_test):
        return super().predict(X_test)