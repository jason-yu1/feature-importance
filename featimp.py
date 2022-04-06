import numpy as np
import pandas as pd
import sklearn.base
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def spearman_rank(df, target, col):
    # Calculate spearman_rank for dataset
    y = df[target]
    x = np.array(df[col]).astype(int)

    xbar = np.mean(x)
    ybar = np.mean(y)
    p = np.sum((x - xbar)*(y - ybar)) / np.sqrt(np.sum((x - xbar)**2) * np.sum((y - ybar)**2))

    return np.abs(p)

def pca_features(df, cols, n):
    X = df[cols].to_numpy()
    scaler = StandardScaler()
    # scale data
    X = scaler.fit_transform(X)
    # fit pca
    pca = PCA(n_components=n)
    X = pca.fit_transform(X)
    # get loadings
    loadings = pca.components_
    loadings = pd.DataFrame(np.abs(loadings)[0])
    loadings.index = cols

    return loadings.T

def permutation_importance(model, X, y):
    baseline = r2_score(y, model.predict(X))
    importances = []
    for col in X.columns:
        save = X.loc[:, col].copy()
        X.loc[:, col] = np.random.permutation(X[col])
        m = r2_score(y, model.predict(X))
        X.loc[:, col] = save
        importances.append(baseline - m)
    return importances

def drop_column_importance(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    baseline = r2_score(y_valid, model.predict(X_valid))
    importances = []
    for col in X_train.columns:
        X_train_ = X_train.drop(col, axis=1)
        X_valid_ = X_valid.drop(col, axis=1)
        model_ = sklearn.base.clone(model)
        model_.fit(X_train_, y_train)
        m = r2_score(y_valid, model_.predict(X_valid_))
        importances.append(baseline - m)
    return importances

def auto_feature_select(model, X, y, method='permutation'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=23)

    model.fit(X_train, y_train)
    baseline = mean_squared_error(y_test, model.predict(X_test))
    increase = False
    worst_cols = []
    while increase == False:
        worst = get_worst_feature(method, X, y)
        worst_cols.append(worst)
        X = X.drop(worst, axis=1)
        X_train_ = X_train.drop(worst, axis=1)
        X_test_ = X_test.drop(worst, axis=1)
        model_ = sklearn.base.clone(model)
        model_.fit(X_train_, y_train)
        m = mean_squared_error(y_test, model_.predict(X_test_))

        if m > baseline:
            return X_train.columns
        else:
            X_train = X_train_
            X_test = X_test_


    return X_train

def get_worst_feature(method, X, y):
    if method == 'permutation':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=23)
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)
        permutation_importances = pd.DataFrame({'feature':X_train.columns,
                                                'score':permutation_importance(model, X_test, y_test)})
        return permutation_importances.sort_values('score').iloc[0]['feature']
    elif method == 'drop_column':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=23)
        model = xgb.XGBRegressor()
        drop_column_importances = pd.DataFrame({'feature':X_train.columns,
                                                'score':drop_column_importance(model, X_train, y_train, X_test, y_test)})
        return drop_column_importances.sort_values('score').iloc[0]['feature']
    elif method == 'spearman':
        columns = X.columns
        target = 'mpg'
        df = pd.concat([X, y], axis=1)
        importances = []
        for col in columns:
            importances.append(spearman_rank(df, target, col))
        spearman_importances = pd.DataFrame({'feature':columns, 'p':importances})
        return spearman_importances.sort_values('p').iloc[0]['feature']

















