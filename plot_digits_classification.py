# The digits dataset consists of 8x8 pixel images of digits. The images attribute of the dataset stores 8x8 arrays of grayscale values for each image. We will use these arrays to visualize the first 4 images. The target attribute of the dataset stores the digit each image represents and this is included in the title of the 4 plots below.

# Import dataset, classifier and performance metrics from utilities file
from sklearn import metrics
from utilities import preprocess, train_classifier, split_dataset,load_digits_dataset,predict_and_eval,split_train_dev_test,tune_hyperparameters

# Data Loading
x,y = load_digits_dataset()

import itertools

# Define the ranges of development and test sizes
dev_size_options = [0.1, 0.2, 0.3]
test_size_options = [0.1, 0.2, 0.3]

# Generate combinations of development and test sizes
dev_test_combinations = [{'test_size': test, 'dev_size': dev} for test, dev in itertools.product(test_size_options, dev_size_options)]

for dict_size in dev_test_combinations:
    test_size_options = dict_size['test_size']
    dev_size_options = dict_size['dev_size']
    train_size = 1 - (dev_size_options+test_size_options)

    # Data splitting into train, test and dev set
    #X_train, X_test, y_train, y_test = split_dataset(x, y, test_size=0.3);
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(x, y, test_size=test_size_options, dev_size=dev_size_options);

    # Data Preprocessing
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)
    X_dev = preprocess(X_dev)

    # Defining the list hyoer-params for SVM Classifier:
    gamma_range = [0.001, 0.01, 0.1, 1.0, 10]
    C_range = [0.1, 1.0, 2, 5, 10]
    param_combinations = [{'gamma': gamma, 'C': C} for gamma, C in itertools.product(gamma_range, C_range)]

    #Tuning the Hyperparams:
    train_acc, best_hparams, best_model, best_accuracy = tune_hyperparameters(X_train, y_train, X_dev, y_dev, param_combinations)

    # Train the data
    model = train_classifier(X_train, y_train, {'gamma': 0.001}, model_type='svm')

    # Predict and evaluation of the model on the test subset
    accuracy_test = predict_and_eval(model,X_test, y_test)
    print("Accuracy on Test Set:", accuracy_test)
    #print("Classification Report on Test Set:\n", classification_rep_test)
    #print("Confusion Matrix on Test Set:\n", confusion_mat_test)

    #Print best params for each of the 9 combinations
    print(f'test_size={test_size_options}, dev_size={dev_size_options}, train_size={train_size}, train_acc:{train_acc:.2f} dev_acc:{best_accuracy:.2f} test_acc: {accuracy_test:.2f}')
    print(f' Best params:{best_hparams}')
