# The digits dataset consists of 8x8 pixel images of digits. The images attribute of the dataset stores 8x8 arrays of grayscale values for each image. We will use these arrays to visualize the first 4 images. The target attribute of the dataset stores the digit each image represents and this is included in the title of the 4 plots below.

# Import dataset, classifier and performance metrics from utilities file
from sklearn import metrics
from utilities import preprocess, train_classifier, split_dataset,load_digits_dataset,predict_and_eval,split_train_dev_test

# Data Loading
x,y = load_digits_dataset()

# Data splitting into train, test and dev set
#X_train, X_test, y_train, y_test = split_dataset(x, y, test_size=0.3);
X_train, X_dev, X_test, y_train, y_dev, y_test = split_train_dev_test(x, y, test_size=0.2, dev_size=0.25);

# Data Preprocessing
X_train = preprocess(X_train)
X_test = preprocess(X_test)

# Train the data
model = train_classifier(X_train, y_train, {'gamma': 0.001}, model_type='svm')

# Predict and evaluation of the model on the test subset
accuracy_test, classification_rep_test, confusion_mat_test = predict_and_eval(model,X_test, y_test)
print("Accuracy on Test Set:", accuracy_test)
print("Classification Report on Test Set:\n", classification_rep_test)
print("Confusion Matrix on Test Set:\n", confusion_mat_test)

