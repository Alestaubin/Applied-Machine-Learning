from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class Data():
    def __init__(self,
                 X,
                 y,
                 test_size=0.2):
        """
        Initialize the data.
        """

        #print(f"X shape before encoding: {X.shape}, y shape before encoding: {y.shape}")
        self.y_encoded, self.y_vocab = self.encode_categorical(y)
        self.X_encoded, self.X_vocab = self.encode_categorical(X)

        # NOTE: should we scale the encoded data as well? (alex)
        self.__scaler = preprocessing.StandardScaler().fit(self.X_encoded)
        self.__mean = self.__scaler.mean_
        self.__std = self.__scaler.scale_

        self.X_scaled = self.__scaler.transform(self.X_encoded)
        # split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_scaled, self.y_encoded, test_size=test_size, random_state=42)
    
    def encode_categorical(self, x):
        """
        Encode categorical Data.
        """
        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x)
        # turn each row of the dataframe into a dictionary
        x_dict = x.to_dict(orient='records')
        #print(f"First rows of the data: {x_dict[:5]}")
        # initialize the DictVectorizer
        dv_X = DictVectorizer(sparse=False) 
        # encode the data
        x_encoded = dv_X.fit_transform(x_dict) # returns a numpy array
        # get the vocabulary (feature name and corresponding column index)
        vocab = dv_X.vocabulary_
        #print(f"Vocabulary: {vocab}")
        #print first rows of the encoded data
        #print(f"First rows of the encoded data: {x_encoded[:5]}")
        return x_encoded, vocab
    
    def get_stats(self):
        """
        Get the statistics of the data.
        """
        return self.__mean, self.__std
"""
def load_ITT():
    
    Load and clean the Infrared Thermography Temperature dataset for the regression model.
    Returns:
        X : array, shape (n_samples, n_features)
            The features.
        y : array, shape (n_samples,)
            The target values.

    infrared_thermography_temperature = fetch_ucirepo(id=925) 
    X = infrared_thermography_temperature.data.features 
    y = infrared_thermography_temperature.data.targets 
    y = y[['aveOralM']]

    numerical_columns = X.select_dtypes(exclude=['object']).columns
    X.loc[:, numerical_columns] = X[numerical_columns].fillna(X[numerical_columns].mean())

    categorical_columns = X.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y = y.fillna(y.mean())

    return X_scaled, y
"""
def load_DHI():
    """
    Load the CDC Diabetes Health Indicators dataset for the classification model.
    Returns:
        X : array, shape (n_samples, n_features)
            The features.
        y : array, shape (n_samples,)
            The target values.
    """
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
    X = cdc_diabetes_health_indicators.data.features 
    y = cdc_diabetes_health_indicators.data.targets 
    return X, y



class LinRegData():
    """
    * must be given clean data
    * prepares data for regresion training
    """
    def __init__(self, X, y, test_size = 0.2, seed=42):
        X.replace("", np.nan, inplace=True)
        X = X.dropna(subset=['Distance']) # delete empty rows
        y = y.loc[X.index]
        self.features = X.columns
        X, y = X.to_numpy(), y.to_numpy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state = seed)
        
        

def load_ITT():
    infrared_thermography_temperature = fetch_ucirepo(id=925) 
    X = infrared_thermography_temperature.data.features 
    y = infrared_thermography_temperature.data.targets
    y = y[['aveOralM']]

    #1: encode data
    categorical_columns = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, dtype=int, columns=categorical_columns, drop_first=True)

    #2: scale data
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_encoded)
    X_scaled = pd.DataFrame(X_scaled, columns=X_encoded.columns)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    y_scaled = pd.DataFrame(y_scaled, columns=y.columns)

    return X_scaled, y_scaled








