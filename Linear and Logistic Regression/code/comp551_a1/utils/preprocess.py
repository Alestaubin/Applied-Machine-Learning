# Description: This file contains the functions to preprocess the data for training and testing 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE

def preprocess_DHI(X, y, test_size, seed):
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
