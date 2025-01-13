from utils.data_utils import LinRegData, load_ITT, load_DHI
from models.regression import RegressionModel
import numpy as np
import pandas as pd
import argparse
import json
from utils.preprocess import preprocess_DHI
from models.classification import BinaryClassification
from utils.log_config import logger

def main(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)

    model = config['model']
    test_size = config['test_size']
    seed = config['seed']

    if model == 'linear':
        # load the parameters
        add_bias = config['add_bias']
        batch_size = config['batch_size']
        # load the data
        X, y = load_ITT()
        data = LinRegData(X, y, test_size, )
        regression = RegressionModel(add_bias)

        if batch_size is None:
            model_choice = "Closed form"
            logger.info("Fitting the closed form linear regression model")
            regression.fit(data.X_train, data.y_train)
        else:
            model_choice = "Mini-batch SGD"
            max_iters = config['max_iters']
            learning_rate = config['learning_rate']
            epsilon = config['epsilon']
            logger.info("Fitting the mini-batch SGD linear regression model")
            if add_bias:
                w_0 = np.random.randn(X.shape[1]+1)
            else:
                w_0 = np.random.randn(X.shape[1])

            regression.regression_MB_SGD(data.X_train, data.y_train, w_0, batch_size, learning_rate, max_iters, epsilon)
        
        r2_train = regression.score(data.X_train, data.y_train)
        r2_test = regression.score(data.X_test, data.y_test)

        print(f"R2 Score on training data: {r2_train} [{model_choice} linear Regression]")
        print(f"R2 Score on testing data: {r2_test} [{model_choice} linear regression]")

    elif model == 'logistic':
        # load the parameters
        learning_rate = config['learning_rate']
        max_iters = config['max_iters']
        batch_size = config['batch_size']
        beta_1 = config['beta_1']
        beta_2 = config['beta_2']
        _lambda = config['lambda']
        patience = config['patience']
        verbose = config['verbose']

        # load the data
        X, y = load_DHI()
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        
        # preprocess the data
        X_train, y_train, X_test, y_test = preprocess_DHI(X, y, test_size, seed=seed)

        model = BinaryClassification(learning_rate=learning_rate, max_iters=max_iters, batch_size=batch_size, _beta_1=beta_1, _beta_2=beta_2, seed=seed, _lambda=_lambda, patience=patience, verbose=verbose)
        print(model)
        model.train(X_train, y_train, X_test, y_test)
        accuracy, f1, roc_auc, confusion_matrix = model.evaluate(X_test, y_test)
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"ROC AUC Score: {roc_auc}")
        print(f"Confusion Matrix: {confusion_matrix}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the main script.')

    parser.add_argument('--config', type=str, help='The path to the config file', required=True)
    args = parser.parse_args()

    main(args.config)