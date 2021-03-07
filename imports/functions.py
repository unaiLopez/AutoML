def functions():
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
