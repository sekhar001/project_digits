
from sklearn import datasets
#from api.hello import app
import numpy as np
import requests
import pytest
import json
import os
import joblib
from sklearn.linear_model import LogisticRegression
import glob



# def test_post_predict():
#     data = get_digit_array(0)
#     response = app.test_client().post('/predict', json=data)
#     #print('==>',response.get_json()['predicted_digit'])
#     assert response.get_json()['predicted_digit'] == 0, f"Assertion failed for digit {0}"
    

#     data = get_digit_array(1)
#     response = app.test_client().post('/predict', json=data)    
#     assert response.get_json()['predicted_digit'] == 1, f"Assertion failed for digit {1}"
    
#     data = get_digit_array(2)
#     response = app.test_client().post('/predict', json=data)    
#     assert response.get_json()['predicted_digit'] == 2, f"Assertion failed for digit {2}"

#     data = get_digit_array(3)
#     response = app.test_client().post('/predict', json=data)    
#     assert response.get_json()['predicted_digit'] == 3, f"Assertion failed for digit {3}"

#     data = get_digit_array(4)
#     response = app.test_client().post('/predict', json=data)    
#     assert response.get_json()['predicted_digit'] == 4, f"Assertion failed for digit {4}"
    
#     data = get_digit_array(5)
#     response = app.test_client().post('/predict', json=data)    
#     assert response.get_json()['predicted_digit'] == 5, f"Assertion failed for digit {5}"

#     data = get_digit_array(6)
#     response = app.test_client().post('/predict', json=data)    
#     assert response.get_json()['predicted_digit'] == 6, f"Assertion failed for digit {6}"
    
#     data = get_digit_array(7)
#     response = app.test_client().post('/predict', json=data)    
#     assert response.get_json()['predicted_digit'] == 7, f"Assertion failed for digit {7}"
    
#     data = get_digit_array(8)
#     response = app.test_client().post('/predict', json=data)    
#     assert response.get_json()['predicted_digit'] == 8, f"Assertion failed for digit {8}"
    
#     data = get_digit_array(9)
#     response = app.test_client().post('/predict', json=data)    
#     assert response.get_json()['predicted_digit'] == 9, f"Assertion failed for digit {9}"

#     assert response.status_code == 200


# def get_digit_array(digit):
#     # Load the digits dataset
#     digits = datasets.load_digits()
#     digit_arr = {}

#     # for digit in range(0,10):
#     # Select samples for digit "i"
#     digit_samples = digits.data[digits.target == digit]

#     # Take a random sample from the selected digit samples
#     random_index = np.random.randint(0, digit_samples.shape[0])
#     random_digit = digit_samples[random_index]

#     # Convert the NumPy array to a JSON-formatted string
#     json_image_array = random_digit.tolist()

#     # Prepare the dict
#     digit_arr['image'] = f"{json_image_array}"
    
#     return digit_arr

# def test_get_root():
#     response = app.test_client().get("/")
#     assert response.status_code == 200
#     assert response.get_data() == b"Hello, World!"


# test_post_predict()

# ## Adding test case to check if model is indeed LR.
# def test_loaded_model_is_logistic_regression():
#     # Assuming you have the filename of the saved model
#     model_filename = "M22AIE238_best_logistic_regression_model_logistic_regression_solvernewton-cg.pkl"  # Updated with the actual filename
#     model_path = r"C:\Users\Niladri\mlops\project_digits\models\M22AIE238_best_logistic_regression_model_logistic_regression_solvernewton-cg.pkl"  # Updated the actual path

#     # Load the model
#     loaded_model = joblib.load(model_path)

#     # Check if the loaded model is an instance of LogisticRegression
#     assert isinstance(loaded_model, LogisticRegression)
# import joblib
# from sklearn.linear_model import LogisticRegression


def test_model_is_lr():
    #solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
    #for solver in solvers:
    loaded_model = joblib.load(r"./models/M22AIE238_best_logistic_regression_model_logistic_regression_solver_newton-cg.pkl")
       #check if loaded model is logistic regression model
    assert loaded_model.__class__.__name__ == 'LogisticRegression'


def test_solver_name_matches_model_file_name():
    # Replace with your actual roll number and solver name used in the test case
    roll_number = "M22AIE238"
    model_solver_list = ["liblinear", "lbfgs", "newton-cg"]
    model_files = glob.glob(f"./models/{roll_number}_best_logistic*")
    print(model_files)

    for model_path in model_files:
        # Extract solver name from the model file name
        model_solver_name = model_path.split('_')[-1].split('.')[0]
        print('==>',model_solver_name)
        # # Check if the solver name in the model file name belongs to the given solver list
        assert model_solver_name in model_solver_list

