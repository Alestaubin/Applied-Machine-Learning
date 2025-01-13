from comp551_a1.utils.data_utils import LinRegData, load_ITT
from comp551_a1.models.regression import RegressionModel
import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt

def eval_linreg_performance(add_bias, model_choice, batch_size, learning_rate, max_iters, epsilon):
    X, y = load_ITT()
    data = LinRegData(X, y, 0.2, )
    regression = RegressionModel(add_bias)

    if model_choice == "regression":
        regression.fit(data.X_train, data.y_train)
    else:
        w_0 = np.random.randn(X.shape[1])
        regression.regression_MB_SGD(data.X_train, data.y_train, w_0, batch_size, learning_rate, max_iters, epsilon)
    r2_train = regression.score(data.X_train, data.y_train)
    r2_test = regression.score(data.X_test, data.y_test)

    print(f"R2 Score on training data: {r2_train} [{model_choice} linear Regression]")
    print(f"R2 Score on testing data: {r2_test} [{model_choice} linear regression]")

def analyze_features(add_bias, model_choice, batch_size, learning_rate, max_iters, epsilon):
    X, y = load_ITT()
    data = LinRegData(X, y)
    regression = RegressionModel(add_bias)

    if model_choice == "Standard":
        regression.fit(data.X_train, data.y_train)
    else:
        w_0 = np.random.randn(X.shape[1])
        regression.regression_MB_SGD(data.X_train, data.y_train, w_0, batch_size, learning_rate, max_iters, epsilon)
    weights = np.array(regression.w).flatten()
    features = data.features

    if add_bias:
        weights = weights[1:]

    plt.figure(figsize=(12, 10))
    plt.barh(features, weights, color='skyblue')
    plt.xlabel('Weight Value')
    plt.ylabel('Feature Name')
    plt.title(f"Feature Weights for {model_choice} Linear Regression")
    plt.tight_layout()
    
    output_path = '/Users/jakeg/Desktop/CSVs/feature_weights_plot.png'
    plt.savefig(output_path)
    plt.close()

def analyze_data_split(add_bias, train_sizes, model_choice, batch_size, learning_rate, max_iters, epsilon):
    X, y = load_ITT()
    regression = RegressionModel(add_bias)
    train_scores = [] 
    test_scores = []
    seeds = list(range(1,11))

    for size in train_sizes:

        sum_of_r2s_train = 0
        sum_of_r2s_test = 0

        if model_choice == "regression":

            for seed in seeds:
                data = LinRegData(X, y, size, seed)
                regression.fit(data.X_train, data.y_train)
                train = regression.score(data.X_train, data.y_train)
                test = regression.score(data.X_test, data.y_test)
                sum_of_r2s_train += train if math.fabs(train) <= 1 else 0
                sum_of_r2s_test += test if math.fabs(test) <= 1 else 0

            train_scores.append(sum_of_r2s_train / 10)
            test_scores.append(sum_of_r2s_test / 10)
            

            print(f"Training on {(1-size) * 100}% of data, performance (train): {sum_of_r2s_train / 10}")
            print(f"Training on {(1-size) * 100}% of data, performance (test): {sum_of_r2s_test / 10}")

        else:

            for seed in seeds:
                data = LinRegData(X, y, size, seed)
                w_0 = np.random.randn(X.shape[1] + 1) if add_bias else np.random.randn(X.shape[1])
                regression.regression_MB_SGD(data.X_train, data.y_train, w_0, batch_size, learning_rate, max_iters, epsilon, seed)
                train = regression.score(data.X_train, data.y_train)
                test = regression.score(data.X_test, data.y_test)
                sum_of_r2s_train += train if math.fabs(train) <= 1 else 0
                sum_of_r2s_test += test if math.fabs(test) <= 1 else 0

            train_scores.append(sum_of_r2s_train / 10)
            test_scores.append(sum_of_r2s_test / 10)

            print(f"Training on {(1-size) * 100}% of data, performance (train): {sum_of_r2s_train / 10}")
            print(f"Training on {(1-size) * 100}% of data, performance (test): {sum_of_r2s_test / 10}")
            
    plt.figure(figsize=(10, 6))
    index = np.arange(len(train_sizes))
    bar_width = 0.35

    plt.bar(index, train_scores, bar_width, label='Train Performance', color='skyblue')
    plt.bar(index + bar_width, test_scores, bar_width, label='Test Performance', color='orange')
    plt.xlabel('Training Set Size (%)')
    plt.ylabel('R² Score')
    plt.title('Performance vs Training Set Size (Averaged Over 100 Seed Values)')

    plt.xticks(index + bar_width / 2, [f'{round((1-size) * 100)}%' for size in train_sizes])
    plt.legend()
    plt.tight_layout()
    
    output_path = '/Users/jakeg/Desktop/CSVs/performance_vs_training_size_bar_chart.png'
    plt.savefig(output_path)
    plt.close()

def analyze_batch_sizes(add_bias, batch_sizes, learning_rate, max_iters, epsilon, seeds): 

    X, y = load_ITT()
    regression = RegressionModel(add_bias)
    w_0 = np.random.randn(X.shape[1] + 1) if add_bias else np.random.randn(X.shape[1])
    results = []
    times = []

    for size in batch_sizes:

        r2_sum = 0
        time_sum = 0

        for seed in seeds:
            data = LinRegData(X, y, 0.2, seed)
            start_time = time.time()
            regression.regression_MB_SGD(data.X_train, data.y_train, w_0, size, learning_rate, max_iters, epsilon, seed)
            end_time = time.time()
            r2 = regression.score(data.X_test, data.y_test)
            r2_sum += r2
            time_sum += (end_time - start_time)
        
        results.append(r2_sum / len(seeds))
        times.append(time_sum / len(seeds))
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('R² Score', color=color)
    ax1.plot(batch_sizes, results, marker='o', linestyle='-', color=color, label='R² Score')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('Runtime (seconds)', color=color)  
    ax2.plot(batch_sizes, times, marker='o', linestyle='--', color=color, label='Runtime (seconds)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title('Performance and Runtime vs Batch Size')
    plt.grid(True)
    
    plt.show()

def analyze_learning_rates(lr1, lr2, lr3): 
    max_iters = 1000
    epsilon = 0.001
    batch_size = 200
    analyze_data_split(True, [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1], "gradient", batch_size, lr1, max_iters, epsilon)
    analyze_data_split(True, [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1], "gradient", batch_size, lr2, max_iters, epsilon)
    analyze_data_split(True, [0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1], "gradient", batch_size, lr3, max_iters, epsilon)

    

def regular_vs_minibatch(add_bias, max_iterations):
    X, y = load_ITT()
    data = LinRegData(X, y, 0.2)
    regression = RegressionModel(add_bias)
    regression.fit(data.X_train, data.y_train)
    grad_regression = RegressionModel(add_bias)
    r2_closed_form = regression.score(data.X_train, data.y_train)

    
    epsilon = 0.001
    batch_size = 350
    learning_rate = 0.05
    results = []
    w_0 = np.random.randn(X.shape[1])

    for iters in max_iterations:
        regression.regression_MB_SGD(data.X_train, data.y_train, w_0, batch_size, learning_rate, iters, epsilon)
        r2_train = regression.score(data.X_train, data.y_train)
        results.append(r2_train)


    plt.figure(figsize=(10, 6))
    
    plt.axhline(y=r2_closed_form, color='r', linestyle='--', label=f'Closed-form R² = {r2_closed_form:.4f}')
    
    plt.plot(max_iterations, results, marker='o', linestyle='-', color='b', label='R² (Mini-batch GD)')
    
    plt.xlabel('Number of Iterations')
    plt.ylabel('R² Score')
    plt.title('R² Score: Closed-Form vs Gradient Descent')
    plt.legend()
    plt.grid(True)
    plt.show()


######### GENERAL DATA STATISTICS ############

file_path = "/Users/jakeg/Desktop/CSVs/infrared_thermography_temperature.csv"
data = pd.read_csv(file_path)

data['Gender'].value_counts().plot(kind='bar', color=['blue', 'pink'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()