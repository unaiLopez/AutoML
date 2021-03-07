import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, SGDRegressor, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA

class AutoTabularData:

    def __init__(self, scoring_function, n_jobs, n_iterations, cv):
        self.scoring_function = scoring_function
        self.n_jobs = n_jobs
        self.n_iterations = n_iterations
        self.cv = cv
        self.model_pipeline = None
        self.best_estimator = None
        self.best_pipeline = None

    def _clean_transform_data(self, X_train):
        #Create a pipeline to fill NaN values with the columns mean value and scale all the values with StandardScaler()
        num_pipeline = Pipeline([
            ('cleaner',SimpleImputer()),
            ('scaler', StandardScaler())
        ])
        
        #Get categorical values
        categorical_values = []
        cat_subset = X_train.select_dtypes(include = ['object','category','bool'])
        for i in range(cat_subset.shape[1]):
            categorical_values.append(list(cat_subset.iloc[:,i].dropna().unique()))

        #Create a pipeline to substitute NaN categorical values by the most frequent categorical value and encoding categorical values with OneHotEncoder()
        cat_pipeline = Pipeline([
            ('cleaner',SimpleImputer(strategy = 'most_frequent')),
            ('encoder', OneHotEncoder(sparse = False, categories=categorical_values))
        ])

        #Apply the previous pipelines: Apply the first to numerical values and the second to categorical values
        preprocessor = ColumnTransformer([
            ('numerical', num_pipeline, make_column_selector(dtype_exclude=['object','category','bool'])),
            ('categorical', cat_pipeline, make_column_selector(dtype_include=['object','category','bool']))
        ])
        
        model_pipeline_steps = list()
        model_pipeline_steps.append(('preprocessor', preprocessor))
        model_pipeline_steps.append(('dimensionality_reduction', PCA()))
        model_pipeline_steps.append(('estimator', LogisticRegression()))
        model_pipeline = Pipeline(model_pipeline_steps)

        self.model_pipeline = model_pipeline

    def _model_selector_hyperparameter_tuning(self, X_train, y_train):
        pass

    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        return self.best_estimator.predict(X_test) 