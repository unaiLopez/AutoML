import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA

from tabular.AutoTabularData import AutoTabularData

class AutoTabularDataRegressor(AutoTabularData):
    
    def __init__(self, scoring_function='neg_mean_squared_error', n_jobs=-1, n_iterations=50, cv=5):
        super().__init__(scoring_function, n_jobs, n_iterations, cv)

    def _clean_transform_data(self, X_train):
        super()._clean_transform_data(X_train)

    def _model_selector_hyperparameter_tuning(self, X_train, y_train):
        optimization_grid = list()

        scalers = [RobustScaler(),StandardScaler(),MinMaxScaler()]
        clean_strategies = ['mean', 'median']

        #Linear regression
        optimization_grid.append({
            'preprocessor__numerical__scaler': scalers,
            'preprocessor__numerical__cleaner__strategy': clean_strategies,
            'dimensionality_reduction': [None, PCA(), KernelPCA(), IncrementalPCA()],
            'estimator': [LinearRegression()]
        })

        #K-nearest neighbors
        optimization_grid.append({
            'preprocessor__numerical__scaler': scalers,
            'preprocessor__numerical__cleaner__strategy': clean_strategies,
            'dimensionality_reduction': [None, PCA(), KernelPCA(), IncrementalPCA()],
            'estimator': [KNeighborsRegressor()],
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
            'dimensionality_reduction': [None, PCA(), KernelPCA(), IncrementalPCA()],
            'estimator': [RandomForestRegressor(random_state=0)],
            'estimator__n_estimators': np.arange(100, 1000, 10),
            'estimator__criterion':['mse','mae'],
            'estimator__min_samples_split': np.arange(2, 10, 4),
            'estimator__min_samples_leaf': np.arange(1, 5, 4)
        })


        #Gradient boosting
        optimization_grid.append({
            'preprocessor__numerical__scaler': [None],
            'preprocessor__numerical__cleaner__strategy': clean_strategies,
            'dimensionality_reduction': [None, PCA(), KernelPCA(), IncrementalPCA()],
            'estimator': [GradientBoostingRegressor(random_state=0)],
            'estimator__n_estimators': np.arange(5,500,10),
            'estimator__learning_rate': np.linspace(0.1, 0.9, 20),
            'estimator__criterion': ['friedman_mse', 'mse'],
            'estimator__loss': ['ls', 'huber', 'lad', 'quantile']
        })

        #Ada boosting
        optimization_grid.append({
            'preprocessor__numerical__scaler': [None],
            'preprocessor__numerical__cleaner__strategy': clean_strategies,
            'dimensionality_reduction': [None, PCA(), KernelPCA(), IncrementalPCA()],
            'estimator': [AdaBoostRegressor(random_state=0)],
            'estimator__n_estimators': np.arange(50, 1000, 10),
            'estimator__learning_rate': np.linspace(0.01, 0.9, 20),
            'estimator__loss': ['linear', 'square']
        })

        #SVM
        optimization_grid.append({
            'preprocessor__numerical__scaler': scalers,
            'preprocessor__numerical__cleaner__strategy': clean_strategies,
            'dimensionality_reduction': [None, PCA(), KernelPCA(), IncrementalPCA()],
            'estimator': [SVR()],
            'estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'estimator__gamma': ['scale', 'auto'],
            'estimator__C': np.arange(0.1, 1, 3)
            
        })

        #Stochastic Gradient Descent
        optimization_grid.append({
            'preprocessor__numerical__scaler': scalers,
            'preprocessor__numerical__cleaner__strategy': clean_strategies,
            'dimensionality_reduction': [None, PCA(), KernelPCA(), IncrementalPCA()],
            'estimator': [SGDRegressor(random_state=0)],
            'estimator__loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'estimator__penalty': ['l2', 'l1', 'elasticnet'],
            'estimator__alpha': np.linspace(1e-5, 1e-2, 6)
            
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
        self._model_selector_hyperparameter_tuning(X_train, y_train)
    
    def predict(self, X_test):
        return super().predict(X_test)