import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class RegressionModel:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        self.w = None

    def fit(self, X, y):
        if self.add_bias:
            X = np.insert(X, 0, 1, axis=1)  
        
        #compute the weights using the closed-form solution
        Xt = X.T
        XtX = np.matmul(Xt, X)
        inverse_XtX = np.linalg.inv(XtX)
        inverse_XtX_times_Xt = np.matmul(inverse_XtX, Xt)
        self.w = np.matmul(inverse_XtX_times_Xt, y)

    def predict(self, X):
        # Predict outcomes for a given input feature matrix X
        if self.add_bias:
            X = np.insert(X, 0, 1, axis=1) 
        prediction = np.matmul(X, self.w)
        return prediction
        
    def score(self, X, y):
        # compute the R2 score to evaluate model performance
        y_pred = self.predict(X)  
        total_sum_of_squares = np.sum((y - np.mean(y))**2)  
        residual_sum_of_squares = np.sum((y - y_pred)**2)  
        r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
        return r2

    def pred_vs_act(self, X_test, y_test):
        y_pred = self.predict(X_test)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal line (y=x)')
        plt.xlabel('Actual values')
        plt.ylabel('Predicted values')
        plt.title('Predicted vs Actual values')
        plt.legend()
        plt.show()

    #### GRADIENT DESCENT ####

    def grad_of_j_cost(self, x_j, y_j, w):
        # Compute the gradient of the cost function w.r.t. a single data point
        prediction_error = np.dot(w, x_j) - y_j
        return prediction_error * x_j

    def grad_cost(self, X, y, w, batch_size, seed=42):
        np.random.seed(seed)
        B = np.random.choice(range(X.shape[0]), batch_size, replace=False)
        
        sum_of_grads = np.zeros(X.shape[1])  # Initialize sum of gradients
    
        for j in B:
            sum_of_grads += self.grad_of_j_cost(X[j], y[j], w)

        #Return the average gradient over the mini-batch
        return sum_of_grads * (1 / batch_size)
     
    def regression_MB_SGD(self, X, y, w_0, batch_size, learning_rate, max_iters, epsilon, seed=42):
        if self.add_bias:
            X = np.insert(X, 0, 1, axis=1)

        i = 0  
        w = w_0
        while i < max_iters:
            grad = self.grad_cost(X, y, w, batch_size) 
        
            # Check if gradient is small enough to stop (convergence)
            if np.linalg.norm(grad) < epsilon:
                break

            # Update weights using the gradient
            w = w - learning_rate * grad
            i += 1

        # Store the learned weights after optimization is done
        self.w = w[:, np.newaxis]