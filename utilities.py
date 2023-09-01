from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

# Function to load and return the digits dataset
def load_digits_dataset():
    digits = datasets.load_digits()
    images = digits.images
    labels = digits.target 
    return images, labels

# Define a function to preprocess the data
def preprocess(data):
    num_samples = len(data)
    data = data.reshape((num_samples, -1))
    return data

# Split the data into training and testing sets
def split_dataset(X, y, test_size=0.5, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# Train a specified model on the given data
def train_classifier(X, y, model_params, model_type='svm'):
    if model_type == 'svm':
        classifier = svm.SVC(**model_params)
    classifier.fit(X, y)
    return classifier

# Function for splitting the data set inot train, test and dev set
def split_train_dev_test(X, y, test_size=0.2, dev_size=0.25, random_state=1):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=dev_size / (dev_size + test_size), random_state=random_state
    )
    return X_train, X_dev, X_test, y_train, y_dev, y_test

# Function for evaluationg the model
def predict_and_eval(model, X, y):
    y_pred = model.predict(X)
    accuracy = (y_pred == y).mean()
    classification_rep = classification_report(y, y_pred)
    confusion_mat = confusion_matrix(y, y_pred)
    return accuracy, classification_rep,confusion_mat