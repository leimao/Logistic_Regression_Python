

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from logistic_regressor import LogisticRegressor
from utils import train_test_splitter, error_rate, plot_losses


def preprocess_data(file_path = 'data/wdbc.data'):

    data = pd.read_csv(file_path, header = None)
    base_names = ['radius', 'texture', 'perimeter', 'area', 'smooth', 'compact', 'concav', 
                 'conpoints', 'symmetry', 'fracdim']
    names = ['m' + name for name in base_names]
    names += ['s' + name for name in base_names]
    names += ['e' + name for name in base_names]
    features = names.copy()
    names = ['id', 'class'] + names
    data.columns = names
    data['color'] = pd.Series([(0 if x == 'M' else 1) for x in data['class']])
    target = 'color'

    '''
    label_encoder = LabelEncoder()
    label_encoder.fit(data['class'])
    data['class'] = label_encoder.transform(data['class'])
    '''

    # Data normalization
    # Use StandardScaler
    # Do not confuse with Normalizer!
    # Normalizer scales within one single sample.
    # StandaradScaler scales for all feature values.
    # This prevents overflow or underflow to some extent.
    
    scaler = StandardScaler()
    scaler.fit(data[features])

    X = scaler.transform(data[features])
    y = data[target].as_matrix()

    assert (X.shape[1] == 30 and y.ndim == 1), "Data did not load correctly."

    return data, X, y


def main():

    df, X, y = preprocess_data()
    X_train, X_test, y_train, y_test = train_test_splitter(X = X, y = y, ratio = 0.8)
    logistic_regressor = LogisticRegressor(alpha = 0.05, c = 0.01, T = 1000, random_seed = 0, intercept = True)
    losses = logistic_regressor.fit(X_train, y_train)
    plot_losses(losses = losses, savefig = True)

    train_error = error_rate(y_train, logistic_regressor.predict(X_train))
    test_error = error_rate(y_test, logistic_regressor.predict(X_test))

    print('Training Error: %f' %train_error)
    print('Test Error: %f' %test_error)


if __name__ == '__main__':
    
    main()


