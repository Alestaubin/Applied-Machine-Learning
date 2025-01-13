from data_utils import load_ITT, load_DHI
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


######### GENERAL DATA STATISTICS ############

def analyze_correlation_heatmap(which_dataset):
    # Load the dataset
    if which_dataset == 1:
        X, y = load_ITT()
        target_name = 'aveOralM'
    else:
        X, y = load_DHI()
        target_name = 'Diabetes_binary'

    # Combine X and y into a single DataFrame  
    df = pd.DataFrame(X)
    df[target_name] = y

    correlations = df.corr()

    plt.figure(figsize=(30, 30))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f'Correlation Heatmap for All Features with {target_name}')
    plt.show()
    

def plot_distribution(which_dataset, numerical_column):
    # Load the dataset
    if which_dataset == 1:
        X, y = load_ITT()
        target_name = 'aveOralM'
    else:
        X, y = load_DHI()
        target_name = 'Diabetes_binary'

    # Combine X and y into a single DataFrame    
    df = pd.DataFrame(X)
    df[target_name] = y
    
    # Create a distribution plot (histogram + KDE)
    plt.figure(figsize=(8, 6))
    sns.histplot(df[numerical_column], kde=True, bins=20)
    plt.title(f'Distribution of {numerical_column}')
    plt.xlabel(numerical_column)
    plt.ylabel('Frequency')
    plt.show()


def analyze_numerical_feature(numerical_column):
    
    X, y = load_ITT()
    
    df = pd.DataFrame(X)
    df['aveOralM'] = y

    mean_value = df[numerical_column].mean()
    median_value = df[numerical_column].median()
    mode_value = df[numerical_column].mode()[0] 
    std_dev = df[numerical_column].std()

    # Output the statistics
    print(f"Statistics for {numerical_column}:")
    print(f"Mean: {mean_value}")
    print(f"Median: {median_value}")
    print(f"Mode: {mode_value}")
    print(f"Standard Deviation: {std_dev}")
    
    return mean_value, median_value, mode_value, std_dev


