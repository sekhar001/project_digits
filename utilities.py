from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Function to load and return the digits dataset
def load_digits_dataset():
    digits = datasets.load_digits()
    images = digits.images
    labels = digits.target 
    return images, labels

# # Define a function to preprocess the data
# def preprocess(data):
#     num_samples = len(data)
#     data = data.reshape((num_samples, -1))
#     return data

# Split the data into training and testing sets
def split_dataset(X, y, test_size=0.5, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# # Train a specified model on the given data
# def train_classifier(X, y, model_params, model_type='svm'):
#     if model_type == 'svm':
#         classifier = svm.SVC(**model_params)
#     classifier.fit(X, y)
#     return classifier

# Function for splitting the data set inot train, test and dev set
def split_train_dev_test(X, y, test_size=0.2, dev_size=0.25, random_state=1):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=dev_size / (dev_size + test_size), random_state=random_state
    )
    return X_train, X_dev, X_test, y_train, y_dev, y_test

# # Function for evaluationg the model
# def predict_and_eval(model, X, y):
#     y_pred = model.predict(X)
#     accuracy = (y_pred == y).mean()
#     #classification_rep = classification_report(y, y_pred)
#     #confusion_mat = confusion_matrix(y, y_pred)
#     return accuracy

# # Function for Hyper paramater tuning
# def tune_hyperparameters(train_data, train_labels, dev_data, dev_labels, param_combinations):
#     best_params = None
#     best_model = None
#     best_dev_accuracy = 0.0

#     for params in param_combinations:
#         # Train a model with the current set of hyperparameters
#         model = train_classifier(train_data, train_labels, params)
        
#         # Evaluate the model on the training data
#         train_accuracy = predict_and_eval(model, train_data, train_labels)
        
#         # Evaluate the model on the development data
#         dev_accuracy = predict_and_eval(model, dev_data, dev_labels)
        
#         # Check if this model's development accuracy is better than the current best
#         if dev_accuracy > best_dev_accuracy:
#             best_params = params
#             best_model = model
#             best_dev_accuracy = dev_accuracy
    
#     return train_accuracy, best_params, best_model, best_dev_accuracy

# Define a function for tuning hyperparameters
def tune_hyperparameters(X_train, y_train, X_dev, y_dev, param_combinations, model_type):
    if model_type == 'svm':
        model = SVC()
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier()
    
    grid_search = GridSearchCV(model, param_combinations, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_hparams = grid_search.best_params_
    best_model = grid_search.best_estimator_
    best_accuracy = grid_search.best_score_
    
    return grid_search.best_score_, best_hparams, best_model, best_accuracy

# Define a function to train a classifier
def train_classifier(X_train, y_train, best_hparams, model_type):
    if model_type == 'svm':
        model = SVC(**best_hparams)
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(**best_hparams)
    
    model.fit(X_train, y_train)
    return model

# Define a function to predict and evaluate the model
def predict_and_eval(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Define a function to preprocess the data (you may need to implement this function)
def preprocess(data):
    num_samples = len(data)
    data = data.reshape((num_samples, -1))
    return data


def create_hparam_combo(gamma_range, C_range):
    return [{'gamma': gamma, 'C': C} for gamma in gamma_range for C in C_range]