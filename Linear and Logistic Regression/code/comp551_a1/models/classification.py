import numpy as np
from matplotlib import pyplot as plt
import time
from comp551_a1.utils.log_config import logger # for logging
from tqdm import tqdm  # for the progress bar
from sklearn.metrics import accuracy_score, f1_score, auc, roc_auc_score, roc_curve, confusion_matrix
class BinaryClassification():

    def __init__(self, learning_rate, max_iters, eps=1e-2, batch_size=None, _beta_1=None, _beta_2=None, seed=None, verbose = False, _lambda = 0, patience = None):
        """
        Initialize the model parameters.
        """
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self._weights = None
        self._bias = None
        self.eps=eps, # termination condition for the gradient descent
        self._train_loss = [] # to plot the losses
        self._test_loss = []
        self.batch_size = batch_size
        self.adam = _beta_1 is not None and _beta_2 is not None # whether to use the Adam optimizer
        self._beta_1 = _beta_1 # Adam optimizer parameter
        self._beta_2 = _beta_2 # Adam optimizer parameter
        self.verbose = verbose 
        self._lambda = _lambda # regularization parameter
        self._patience = patience # early stopping parameter

        if seed is not None:
            np.random.seed(seed) # set the seed for reproducibility

    def __repr__(self):
        return str({
            'learning_rate': self.learning_rate,
            'max_iters': self.max_iters,
            'batch_size': self.batch_size,
            'beta_1': self._beta_1,
            'beta_2': self._beta_2,
            'lambda': self._lambda,
            'patience': self._patience
        })

    def cost(self,
            x, # N x D
            y # N
            ):
        """ This implementation of the binary cross entropy loss was taken from the slides """
        z = np.dot(x,self._weights) # N x 1
        J = np.mean( y * np.log1p(np.exp(-z)) + (1-y) * np.log1p(np.exp(z)) )
        return J

    def _sigmoid_activation(self, 
                           x 
                           ):
        """
        Compute the sigmoid activation function.
        """
        #logger.info("_sigmoid_activation")
        return 1 / (1 + np.exp(-x))
    
    def _feed_forward(self, 
                    X_train # n_instances x n_feat
                    ):
        """
        Get the output of the model
        """
        #logger.info("_feed_forward")
        y_pred = np.dot(X_train, self._weights) + self._bias
        y_pred = self._sigmoid_activation(x=y_pred)
        return y_pred

    def _gradient(self, 
                 X_train, 
                 y_train,
                 y_pred
                 ):
        """
        Taken from the slides
        """
        # make y_pred and y_train the same shape
        y_pred = y_pred.reshape(-1,1)
        y_train = y_train.reshape(-1,1)
        # calculate the gradient
        dp = y_pred - y_train
        grad = np.dot(X_train.T, dp) / self._n_feat
        # regularization
        #print("self._lambda * self._weights : ", self._lambda * self._weights)
        grad += self._lambda * self._weights 
        return grad
    
    def _plot(self):
        """
        Plot the losses
        """
        plt.plot(self._train_loss)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Train Loss vs. Iterations')
        output_path = f'plots/trainloss-bs{self.batch_size}-lr{self.learning_rate}-l{self._lambda}-b1{self._beta_1}-b2{self._beta_2}.png'
        plt.savefig(output_path)
        #plt.show()
        plt.close()
        plt.plot(self._test_loss)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Test Loss vs. Iterations')
        output_path = f'plots/testloss-bs{self.batch_size}-lr{self.learning_rate}-l{self._lambda}-b1{self._beta_1}-b2{self._beta_2}.png'
        plt.savefig(output_path)
        #plt.show()
        plt.close()

    def train(self,
            X_train, # n_instances x n_feat
            y_train, # n_instances
            X_test=None, # n_instances x n_feat -- for early stopping
            y_test=None # n_instances -- for early stopping
            ):
        """
        Taken from the slides
        """
        # NOTE: Need to implement early stopping
        logger.info(f'Training the model with {self.max_iters} iterations')
        start_time = time.time()  # Start the timer
        lr = self.learning_rate
        eps = self.eps

        M_t = 0
        S_t = 0

        self.n_instances, self._n_feat = X_train.shape
        # initialize the things
        self._weights = np.zeros((self._n_feat, 1))
        self._bias = 0
        grad = np.inf
        # Variables for early stopping
        best_loss = np.inf
        no_improvement_count = 0

        ############################
        ##### Gradient Descent #####
        ############################
    
        for i in tqdm(range(self.max_iters), desc="Training Progress"):
            if np.linalg.norm(grad) <= eps:
                break

            if self.batch_size is not None:
                idx = np.random.choice(X_train.shape[0], self.batch_size, replace=False)
                X_train_batch = X_train[idx]
                y_train_batch = y_train[idx]
            else:
                X_train_batch = X_train
                y_train_batch = y_train

            y_pred = self._feed_forward(X_train=X_train_batch)
            grad = self._gradient(X_train=X_train_batch, y_train=y_train_batch, y_pred=y_pred)
            #self._train_loss.append(self._BCE_loss(y_train=y_train_batch, y_pred=y_pred))
            self._train_loss.append(self.cost(X_train_batch, y_train_batch))
            # Adam optimization
            if self.batch_size is not None and self.adam:
                grad, M_t, S_t = self._adam_optimizer(grad, M_t, S_t, i)

            # Update weights and bias
            self._weights -= lr * grad
            self._bias -= lr * np.mean(y_pred - y_train_batch)

            # Early stopping logic
            if self._patience is not None and X_test is not None and y_test is not None:
                test_loss = self.cost(x=X_test, y=y_test)
                self._test_loss.append(test_loss)
                if test_loss < best_loss:
                    best_loss = test_loss
                    no_improvement_count = 0  # Reset counter if we see improvement
                else:
                    no_improvement_count += 1  # Increment if no improvement

                if no_improvement_count >= self._patience:
                    logger.info(f'Early stopping at iteration {i} with best loss {best_loss:.4f}')
                    break

        
        logger.info(f"Converged at iteration {i+1} with training loss {self._train_loss[-1]:.4f} and test loss {self._test_loss[-1]:.4f}")
        
        end_time = time.time()  # Stop the timer
        elapsed_time = end_time - start_time  # Calculate elapsed time
        logger.info(f"Training concluded in {elapsed_time:.4f} seconds")
        # plot the losses
        if self.verbose:
            self._plot()
    
    def _adam_optimizer(self, grad, M_t, S_t, i):
        # Adam optimization
        beta_1 = self._beta_1
        beta_2 = self._beta_2

        M_t = beta_1 * M_t + (1 - beta_1) * grad
        S_t = beta_2 * S_t + (1 - beta_2) * grad**2
        M_t_hat = M_t / (1 - beta_1**(i+1))
        S_t_hat = S_t / (1 - beta_2**(i+1))
        grad = M_t_hat / (np.sqrt(S_t_hat) + 1e-8)
        return grad, M_t, S_t
    
    def rfe(self, X_train, y_train, num_features_to_select):
        """
        Perform Recursive Feature Elimination (RFE).
        """
        n_features = X_train.shape[1]
        selected_features = list(range(n_features))
        print("selected_features: ", selected_features)
        for _ in range(n_features - num_features_to_select):
            print(X_train[:, selected_features].shape)
            self.train(X_train[:, selected_features], y_train)  # Train with current features
            feature_importances = np.abs(self._weights).flatten()
            least_important = np.argmin(feature_importances)  # Find the least important feature
            del selected_features[least_important]  # Remove the least important feature

            if self.verbose:
                logger.info(f'Removed feature {least_important}, remaining features: {len(selected_features)}')

        logger.info(f"Selected features: {selected_features}")
        return selected_features


    def predict(self, x):
        """
        Predict the target values.
        """
        #logger.info("predict")
        if self._weights is None:
            raise ValueError("Fit the model before making predictions")
        
        y_pred = self._feed_forward(x)
        for i in range(y_pred.shape[0]):
            if y_pred[i] > 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using sklearn metrics.
        """
        if self._weights is None:
            raise ValueError("Fit the model before making predictions")
        
        y_preds = self.predict(X_test)
        y_true = np.array(y_test)
        accuracy = accuracy_score(y_true, y_preds)
        f1 = f1_score(y_true, y_preds)
        roc_auc = roc_auc_score(y_true, y_preds)
        conf_mat = confusion_matrix(y_test, y_preds)

        print("accuracy : {}".format(accuracy))
        print("f1 score : {}".format(f1))
        print("roc auc score : {}".format(roc_auc))
        print("confusion matrix : {}".format(conf_mat))

        return accuracy, f1, roc_auc, conf_mat

    def get_weights(self):
        return self._weights
    
