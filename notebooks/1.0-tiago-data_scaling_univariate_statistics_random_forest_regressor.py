import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Data scaling and feature selection by univariate statistics with random forest regressor model

def get_training_set():
    df = pd.read_csv('../data/raw/training_data_set.csv', na_values='na')

    X = df[df.columns[2:]]
    y = df[df.columns[1]]

    X = get_clean_data(X)
    y = get_clean_data(y)

    return X,y


def get_clean_data(df):
    df = df.fillna(0)   
    df = df.astype(int)

    return df


def get_normalization(X):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(X)

    X = pd.DataFrame(data_scaled, columns=X.columns)

    return X


def get_kbest(X, y):

    k_best = 100

    selectKBest = SelectKBest(chi2, k_best)
    selectKBest.fit(X, y)

    return selectKBest


def get_features(X, y):
    
    selectKBest = get_kbest(X, y)
    best_X = selectKBest.transform(X)

    idxs_selected = selectKBest.get_support(indices=True)
    best_X = X.iloc[:, idxs_selected]

    return best_X, idxs_selected


def get_balanced_data(best_X, y, idxs_selected):
    number_samples = 2500

    idxs_pos = y[y == 1].index
    idxs_neg = y[y == 0].sample(
        n=number_samples, replace=False, random_state=0).index
    idxs_balanced = np.concatenate((idxs_pos, idxs_neg))
    best_X_balanced = best_X.loc[idxs_balanced]
    y_balanced = y.loc[idxs_balanced]

    print(f'Proportion balanced: {int(number_samples/1000)}/1')

    return best_X_balanced, y_balanced


def get_training_data():
    X, y = get_training_set()

    # normalization
    X = get_normalization(X)

    # feature selection
    best_X, idxs_selected = get_features(X, y)

    # balancing
    best_X_balanced, y_balanced = get_balanced_data(best_X, y, idxs_selected)

    return best_X_balanced, y_balanced


def get_test_data():
    X, y = get_training_set()
    X = get_normalization(X)

    Y = pd.read_csv('../data/raw/test_data_set.csv', na_values='na')
    indices = Y["ID"]
    
    Y = get_clean_data(Y)

    Y = Y.set_index("ID")

    # normalization
    Y = get_normalization(Y)

    # feature selection
    selectKBest = get_kbest(X, y)
    best_Y = selectKBest.transform(Y)

    idxs_selected = selectKBest.get_support(indices=True)
    best_Y = Y.iloc[:, idxs_selected]

    return best_Y, indices


def get_forest_regressor():

    best_X_balanced, y_balanced = get_training_data()

    # model
    classifier = RandomForestRegressor(
        n_estimators=100, oob_score=True, random_state=0, n_jobs=-1)

    accuracy_summary(classifier, best_X_balanced, y_balanced, 'RandomForestRegressor')


def accuracy_summary(classifier, X, y, model_name):
         
    # Separation on training and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 555)

    # Train the model
    model = classifier.fit(x_train, y_train)
    
    # model prediction
    y_pred = model.predict(x_test)
    y_pred = np.round(y_pred)

    # model evaluation
    print("Mislabeled points: %d out of %d"% ( np.sum(np.array(y_test) != np.array(y_pred)), len(y_test)))
    print("Accuracy: ", accuracy_score(y_test, y_pred))  
    print("-"*80)
    print("report")

    report = classification_report(y_test, y_pred)
    print(report)

    # generate submission file
    df_test, index = get_test_data()

    sub_predict = model.predict(df_test)

    sub_predict = np.round(sub_predict)

    df_predictions = pd.DataFrame(sub_predict.astype(np.int), columns=['Predicted'], index=index)
    print(df_predictions.head())

    df_predictions.to_csv('../data/submissions/' + model_name + '_predictions.csv')


if __name__ == "__main__":
    get_forest_regressor()