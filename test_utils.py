
from sklearn import datasets
from api.hello import app
import numpy as np
import requests
import pytest
import json
import os



def test_post_predict():
    data = get_digit_array(0)
    response = app.test_client().post('/predict', json=data)
    #print('==>',response.get_json()['predicted_digit'])
    assert response.get_json()['predicted_digit'] == 0, f"Assertion failed for digit {0}"
    

    data = get_digit_array(1)
    response = app.test_client().post('/predict', json=data)    
    assert response.get_json()['predicted_digit'] == 1, f"Assertion failed for digit {1}"
    
    data = get_digit_array(2)
    response = app.test_client().post('/predict', json=data)    
    assert response.get_json()['predicted_digit'] == 2, f"Assertion failed for digit {2}"

    data = get_digit_array(3)
    response = app.test_client().post('/predict', json=data)    
    assert response.get_json()['predicted_digit'] == 3, f"Assertion failed for digit {3}"

    data = get_digit_array(4)
    response = app.test_client().post('/predict', json=data)    
    assert response.get_json()['predicted_digit'] == 4, f"Assertion failed for digit {4}"
    
    data = get_digit_array(5)
    response = app.test_client().post('/predict', json=data)    
    assert response.get_json()['predicted_digit'] == 5, f"Assertion failed for digit {5}"

    data = get_digit_array(6)
    response = app.test_client().post('/predict', json=data)    
    assert response.get_json()['predicted_digit'] == 6, f"Assertion failed for digit {6}"
    
    data = get_digit_array(7)
    response = app.test_client().post('/predict', json=data)    
    assert response.get_json()['predicted_digit'] == 7, f"Assertion failed for digit {7}"
    
    data = get_digit_array(8)
    response = app.test_client().post('/predict', json=data)    
    assert response.get_json()['predicted_digit'] == 8, f"Assertion failed for digit {8}"
    
    data = get_digit_array(9)
    response = app.test_client().post('/predict', json=data)    
    assert response.get_json()['predicted_digit'] == 9, f"Assertion failed for digit {9}"

    assert response.status_code == 200


def get_digit_array(digit):
    # Load the digits dataset
    digits = datasets.load_digits()
    digit_arr = {}

    # for digit in range(0,10):
    # Select samples for digit "i"
    digit_samples = digits.data[digits.target == digit]

    # Take a random sample from the selected digit samples
    random_index = np.random.randint(0, digit_samples.shape[0])
    random_digit = digit_samples[random_index]

    # Convert the NumPy array to a JSON-formatted string
    json_image_array = random_digit.tolist()

    # Prepare the dict
    digit_arr['image'] = f"{json_image_array}"
    
    return digit_arr

def test_get_root():
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert response.get_data() == b"Hello, World!"


test_post_predict()