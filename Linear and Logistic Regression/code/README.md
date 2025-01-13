## Info 
- The linear regression model is in `comp551_a1/models/regression.py`
- The logistic regression model is in `comp551_a1/models/classification.py`

## Running the code
Make sure you have Conda installed, or install it from [Anaconda Website](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).

Create and activate a new conda environment from the `environment.yml` file,

```
conda env create -f environment.yml
conda activate COMP551
```
Then, build the package by running
```
pip install -e .
```

To run the Logistic Regression on the DHI dataset, simply 

```
python main.py --config "config/logistic_config.json"
```

To run the Linear Regression on the ITT dataset, simply

```
python main.py --config "config/linear_config.json"
```

For linear regression, if `batch_size = null` in the config file, the model will be trained using the closed form solution. For logistic regression, if `batch_size = null` in the config file, the model will be trained using full batch gradient descent.

