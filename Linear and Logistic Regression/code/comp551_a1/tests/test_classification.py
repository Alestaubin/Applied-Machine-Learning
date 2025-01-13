from comp551_a1.utils.data_utils import load_DHI
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from comp551_a1.models.classification import BinaryClassification
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn import datasets
import itertools
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def plot(X,y,save_dir): 
    for col in X.columns:
        # use matplotlib to plot the distribution of the data
        plt.figure()
        plt.hist(X[col][y['Diabetes_binary'] == 0], bins=50, alpha=0.5, label='0')
        plt.hist(X[col][y['Diabetes_binary'] == 1], bins=50, alpha=0.5, label='1')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(f'{col} distribution')
        plt.savefig(f'{save_dir}/{col}_distribution.png')

def preprocess(X, y, test_size, seed):
    """
    This function preprocesses the data for training and testing of the classification model
    """
    # change the data type to int8 to save memory
    X = X.astype('int8')
    y = y.astype('int8')

    print(f'Target value count before balancing:\n {y.value_counts()}')
    X, y = SMOTE(random_state=seed).fit_resample(X, y)
    print(f'Target value count after balancing:\n {y.value_counts()}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # print info of the final data
    print(X_train.info())
    desc = X_train.describe()
    print(desc)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # shuffle the data
    X_train, y_train = shuffle(X_train, y_train, random_state=seed)
    X_test, y_test = shuffle(X_test, y_test, random_state=seed)
    return X_train, y_train, X_test, y_test

def main(X_train, y_train, X_test, y_test, seed, learning_rate, max_iters, batch_size, beta_1, beta_2, _lambda, patience, verbose):

    # for each column, plot the distribution with respect to the target variable
    #plot(X, y, save_dir)

    model = BinaryClassification(learning_rate=learning_rate, max_iters=max_iters, batch_size=batch_size, _beta_1=beta_1, _beta_2=beta_2, seed=seed, _lambda=_lambda, patience=patience, verbose=verbose)
    print(f"Training the model with parameters: {model}")
    model.train(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    accuracy, _, _, _ = model.evaluate(X_test, y_test)
    return accuracy

def grid_search(param_grid, seed, test_size=0.2):
    """
    This function performs a grid search over the given parameter grid
    """
    # Initialize variables to track the best parameters and the highest accuracy
    best_params = None
    best_accuracy = 0
    X, y = load_DHI()
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    
    X_train, y_train, X_test, y_test = preprocess(X, y, test_size, seed=seed)

    # Loop over all combinations of parameters
    i = 0 # counter to keep track of the number of iterations
    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        print(f"Iteration {i}")
        print (f"Testing parameters: {param_dict}")
        # Call the main function with the current set of parameters
        model = BinaryClassification(learning_rate=param_dict['learning_rate'], 
                                    max_iters=param_dict['max_iters'],
                                    batch_size=param_dict['batch_size'],
                                    beta_1=param_dict['beta_1'],
                                    beta_2=param_dict['beta_2'], 
                                    seed=seed, 
                                    _lambda=param_dict['lambda'],
                                    patience=param_dict['patience'],
                                    verbose=True
                                    )
        print(f"Training the model with parameters: {model}")
        model.train(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        accuracy, _, _, _ = model.evaluate(X_test, y_test)
        i += 1
        # If the current accuracy is better, update the best parameters
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = param_dict
    return best_params, best_accuracy

def box_plot_weights(features, weights):
    """
    This function plots the feature weights of the logistic regression model
    """
    print(features.shape)  # Should be (n,) for n features
    print(weights.shape)   # Should be (n,) for n weights
    plt.figure(figsize=(12, 10))
    plt.barh(features, weights, color='skyblue')
    plt.xlabel('Weight Value')
    plt.ylabel('Feature Name')
    plt.title(f"Feature Weights for Logistic Regression")
    plt.tight_layout()
    
    output_path = '../plots/feature_weights_plot.png'
    plt.savefig(output_path)
    plt.close()

def compare_dropped_features(X_train, y_train, X_test, y_test, learning_rate, max_iters, batch_size, _beta_1, _beta_2, seed, _lambda, patience):
    """
    This function was used to determine whether dropping features would improve the model
    """
    model = BinaryClassification(learning_rate=learning_rate, max_iters=max_iters, batch_size=batch_size, _beta_1=_beta_1, _beta_2=_beta_2, seed=seed, _lambda=_lambda, patience=patience)
    print(model)
    model.train(X_train, y_train, X_test, y_test)
    accuracy, f1, roc_auc, confusion_matrix = model.evaluate(X_test, y_test)
    print("+++++++++++++++++++++++++++++++++++++++++++++")
    print("+++++++++++ no dropped features +++++++++++++")
    print("+++++++++++++++++++++++++++++++++++++++++++++")

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC Score: {roc_auc}")
    print(f"Confusion Matrix: {confusion_matrix}")

    best_features = model.rfe(X_train, y_train, 16)
    print (best_features)
    X_train = X_train[:, best_features]
    X_test = X_test[:, best_features]

    model2 = BinaryClassification(learning_rate=learning_rate, max_iters=max_iters, batch_size=batch_size, _beta_1=_beta_1, _beta_2=_beta_2, seed=seed, _lambda=_lambda, patience=patience)
    print(model2)
    model2.train(X_train, y_train, X_test, y_test)
    accuracy, f1, roc_auc, confusion_matrix = model2.evaluate(X_test, y_test)
    print("+++++++++++++++++++++++++++++++++++++++++++++")
    print("+++++++++++++ dropped features ++++++++++++++")
    print("+++++++++++++++++++++++++++++++++++++++++++++")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC Score: {roc_auc}")
    print(f"Confusion Matrix: {confusion_matrix}")

def seeds(numseeds):
    """
    This funcion was used to compare the performance of the models with different seeds
    """
    one_accuracies = []
    two_accuracies = []
    one_f1 = []
    two_f1 = []
    one_roc_auc = []
    two_roc_auc = []
    three_accuracies = []
    three_f1 = []
    three_roc_auc = []

    for i in range(numseeds):
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print(f"++++++++++++++++ seed {i} ++++++++++++++++++")
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        X, y = load_DHI()
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        
        X_train, y_train, X_test, y_test = preprocess(X, y, 0.2, seed=i)
        learning_rate = 0.001
        max_iters = 2000
        batch_size = 64
        patience = 5
        _lambda = 0.01
        beta_1 = 0.99
        beta_2 = 0.999
        model2 = BinaryClassification(learning_rate=learning_rate, max_iters=max_iters, batch_size=batch_size, _beta_1=beta_1, _beta_2=beta_2, seed=i, _lambda=_lambda, patience=patience, verbose=True)
        print(model2)
        model2.train(X_train, y_train, X_test, y_test)
        accuracy, f1, roc_auc, confusion_matrix = model2.evaluate(X_test, y_test)
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"ROC AUC Score: {roc_auc}")
        print(f"Confusion Matrix: {confusion_matrix}")
        one_accuracies.append(accuracy)
        one_f1.append(f1)
        one_roc_auc.append(roc_auc)

        learning_rate = 0.1
        max_iters = 2000
        batch_size = 128
        patience = 5
        _lambda = 0
        beta_1 = None
        beta_2 = None
        model2 = BinaryClassification(learning_rate=learning_rate, max_iters=max_iters, batch_size=batch_size, _beta_1=beta_1, _beta_2=beta_2, seed=i, _lambda=_lambda, patience=patience, verbose=True)
        print(model2)
        model2.train(X_train, y_train, X_test, y_test)
        accuracy, f1, roc_auc, confusion_matrix = model2.evaluate(X_test, y_test)
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"ROC AUC Score: {roc_auc}")
        print(f"Confusion Matrix: {confusion_matrix}")
        two_accuracies.append(accuracy)
        two_f1.append(f1)
        two_roc_auc.append(roc_auc)
        
        learning_rate = 0.01
        max_iters = 2000
        batch_size = 64
        patience = 5
        _lambda = 0.01
        beta_1 = 0.99
        beta_2 = 0.99
        model2 = BinaryClassification(learning_rate=learning_rate, max_iters=max_iters, batch_size=batch_size, _beta_1=beta_1, _beta_2=beta_2, seed=i, _lambda=_lambda, patience=patience, verbose=True)
        print(model2)
        model2.train(X_train, y_train, X_test, y_test)
        accuracy, f1, roc_auc, confusion_matrix = model2.evaluate(X_test, y_test)
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"ROC AUC Score: {roc_auc}")
        print(f"Confusion Matrix: {confusion_matrix}")
        three_accuracies.append(accuracy)
        three_f1.append(f1)
        three_roc_auc.append(roc_auc)

    
    print(f"one_accuracies: {one_accuracies}")
    print(f"two_accuracies: {two_accuracies}")
    print(f"three_accuracies: {three_accuracies}")
    one_accuracies = np.array(one_accuracies)
    two_accuracies = np.array(two_accuracies)
    three_accuracies = np.array(three_accuracies)
    print(f'one_accuracies mean: {np.mean(one_accuracies)}')
    print(f'one_accuracies std: {np.std(one_accuracies)}')
    print(f'two_accuracies mean: {np.mean(two_accuracies)}')
    print(f'two_accuracies std: {np.std(two_accuracies)}')
    print(f'three_accuracies mean: {np.mean(three_accuracies)}')
    print(f'three_accuracies std: {np.std(three_accuracies)}')
    



if __name__ == "__main__":
    #seeds(5)
    # Define the parameter grid for grid search
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.001],
        'max_iters': [2000],
        'batch_size': [8, 32, 64, 128],
        'beta_1': [0.9, 0.99],
        'beta_2': [0.99, 0.999],
        'patience': [5],
        'lambda': [ 1, 0.1, 0.01]
    }

    # Run grid search
    best_params, best_accuracy = grid_search(param_grid, seed=0)
    
    print(f"Best Parameters: {best_params}")
    print(f"Best Accuracy: {best_accuracy}")

