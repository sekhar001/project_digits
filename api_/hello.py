# import joblib
# from flask import Flask, request, jsonify
# from ..utilities import preprocess
# import os
# import numpy as np
# import json

# from markupsafe import escape

# app = Flask(__name__)

# def compare_arrays(arr1, arr2):
#     if arr1 == arr2:
#         return True
#     else:
#         return False

# @app.route('/')
# def hello_world():
#     return 'Hello, World!'

# @app.route('/check_arrays', methods=['POST'])
# def check_arrays():
#     try:
#         data = request.get_json()

#         if 'array1' in data and 'array2' in data:
#             array1 = data['array1']
#             array2 = data['array2']

#             are_arrays_same = compare_arrays(array1, array2)

#             return jsonify({"are_arrays_same": are_arrays_same})
#         else:
#             return jsonify({"error": "Missing 'array1' or 'array2' in request data."}), 400
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# @app.route('/predict', methods=['POST'])
# def predict_digit():
#     data = request.get_json()
#     image_array = data['image']

#     # Preprocess the image data
#     # preprocessed_image = preprocess(np.array(image_array))

#     image_array = np.array(json.loads(image_array))
#     preprocessed_image = preprocess(image_array)

#     # Dynamically load the first model in the 'models/' folder
#     model_files = os.listdir('models/')
#     model_files = [file for file in model_files if file.endswith('.pkl')]

#     if not model_files:
#         raise FileNotFoundError("No model files found in the 'models/' folder")

#     first_model_file = model_files[0]
#     first_model_path = f"models/{first_model_file}"
#     best_model = joblib.load(first_model_path)

#     # Use the loaded model for prediction
#     predicted_digit = best_model.predict(preprocessed_image.reshape(1, -1))[0]

#     response = {
#         "predicted_digit": int(predicted_digit)
#     }

#     return jsonify(response)

import joblib
import json
import os
# print('==>',os.getcwd())
from flask import Flask, request, jsonify
from utilities import preprocess
import numpy as np

# from markupsafe import escape

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict_digit():
    data = request.get_json()
    image_array = data['image']

    image_array = np.array(json.loads(image_array))
    preprocessed_image = preprocess(image_array)

    # Dynamically load the first model in the 'models/' folder
    model_files = os.listdir('models/')
    model_files = [file for file in model_files if file.endswith('.pkl')]

    if not model_files:
        raise FileNotFoundError("No model files found in the 'models/' folder")

    first_model_file = model_files[1] 
    # first_model_file = '/Users/sanjib/Desktop/IITJ/Classes/Sem-2/ML-Ops/Labs/digit-classifications/models/decision_tree_max_depth:10.joblib'
    first_model_path = f"models/{first_model_file}"
    best_model = joblib.load(first_model_path)

    # Use the loaded model for prediction
    predicted_digit = best_model.predict(preprocessed_image.reshape(1, -1))[0]

    response = {
        "predicted_digit": int(predicted_digit)
    }

    return jsonify(response)
    #return str(predicted_digit)