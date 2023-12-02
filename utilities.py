from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.exceptions import ConvergenceWarning

# Function to load and return the digits dataset
def load_digits_dataset():
    digits = datasets.load_digits()
    images = digits.images
    labels = digits.target 
    return images, labels


# Split the data into training and testing sets
def split_dataset(X, y, test_size=0.5, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


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


# # Define a function for tuning hyperparameters
# def tune_hyperparameters(X_train, y_train, X_dev, y_dev, param_combinations, model_type):
#     if model_type == 'logistic_regression':
#         model = LogisticRegression()
#     elif model_type == 'svm':
#         model = SVC()
#     elif model_type == 'decision_tree':
#         model = DecisionTreeClassifier()
    
    
#     grid_search = GridSearchCV(model, param_combinations, cv=3, n_jobs=-1, scoring='accuracy')
#     grid_search.fit(X_train, y_train)
    
#     best_hparams = grid_search.best_params_
#     best_model = grid_search.best_estimator_
#     best_accuracy = grid_search.best_score_
    
#     return grid_search.best_score_, best_hparams, best_model, best_accuracy

def tune_hyperparameters(X_train, y_train, X_dev, y_dev, param_combinations, model_type):
    if model_type == 'logistic_regression':
        # Set max_iter to a higher value and suppress the warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model = LogisticRegression(max_iter=1000)
        #model = LogisticRegression()
    elif model_type == 'svm':
        model = SVC()
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier()
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    grid_search = GridSearchCV(model, param_combinations, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_hparams = grid_search.best_params_
    best_model = grid_search.best_estimator_
    best_accuracy = grid_search.best_score_
    
    return best_accuracy, best_hparams, best_model, best_accuracy




# Define a function to train a classifier
def train_classifier(X_train, y_train, best_hparams, model_type):
    if model_type == 'logistic_regression':
        model = LogisticRegression()
    elif model_type == 'svm':
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

    scaler = StandardScaler() ## Applying Unit normalization as Q1.
    data_normalized = scaler.fit_transform(data)

    return data_normalized


def create_hparam_combo(gamma_range, C_range):
    return [{'gamma': gamma, 'C': C} for gamma in gamma_range for C in C_range]