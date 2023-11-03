from utilities import load_digits_dataset,split_train_dev_test
import os
import numpy as np
from utilities import create_hparam_combo,tune_hyperparameters

# def test_create_hparam_combo():
#     gamma_range = [0.001, 0.01, 0.1, 1.0, 10]
#     C_range = [0.1, 1.0, 2, 5, 10]
#     param_combinations = create_hparam_combo(gamma_range, C_range)
#     assert len(param_combinations) == len(gamma_range) * len(C_range)
# def test_data_splitting():
#     X, y = load_digits_dataset()
    
#     X = X[:100,:,:]
#     y = y[:100]
    
    
#     test_size = .1
#     dev_size = .2

#     X_train, X_test, X_dev, y_train, y_test, y_dev = split_train_dev_test(X, y, test_size=test_size, dev_size=dev_size)

#     assert (len(X_train) == 70) 
#     assert (len(X_test) == 10)
#     assert  ((len(X_dev) == 20))

def test_create_hparam_combo():
    gamma_range = [0.001, 0.01, 0.1, 1.0, 10]
    C_range = [0.1, 1.0, 2, 5, 10]
    param_combinations = create_hparam_combo(gamma_range, C_range)
    assert len(param_combinations) == len(gamma_range) * len(C_range)

# def test_for_hparam_cominations_count():
#     # a test case to check that all possible combinations of paramers are indeed generated
#     gamma_list = [0.001, 0.01, 0.1, 1]
#     C_list = [1, 10, 100, 1000]
#     h_params={}
#     h_params['gamma'] = gamma_list
#     h_params['C'] = C_list
#     h_params_combinations = tune_hyperparameters(h_params)
    
#     assert len(h_params_combinations) == len(gamma_list) * len(C_list)

# def test_for_hparam_combinations_count():
#     # A test case to check that all possible combinations of parameters are indeed generated
#     gamma_list = [0.001, 0.01, 0.1, 1]
#     C_list = [1, 10, 100, 1000]
    
#     # Generate param_combinations
#     param_combinations = [{'gamma': [gamma], 'C': [C]} for gamma in gamma_list for C in C_list]

#     # Create placeholder values for X_train, y_train, X_dev, and y_dev using synthetic data
#     n_samples = 100  # Choose an appropriate number of samples
#     n_features = 20  # Choose an appropriate number of features
    
#     X_train = np.random.rand(n_samples, n_features)
#     y_train = np.random.randint(0, 2, size=n_samples)
#     X_dev = np.random.rand(n_samples // 5, n_features)
#     y_dev = np.random.randint(0, 2, size=n_samples // 5)

#     # Specify the model type (e.g., 'svm' or 'decision_tree')
#     model_type = 'svm'
    
#     h_params_combinations = tune_hyperparameters(X_train, y_train, X_dev, y_dev, param_combinations, model_type)
    
#     # Check that the number of combinations matches the expected count
#     #assert len(h_params_combinations) == len(gamma_list) * len(C_list)
#     assert len(h_params_combinations) == len(gamma_list) * len(C_list)