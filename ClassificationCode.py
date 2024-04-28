# Import necessary libraries for data handling, preprocessing, and machine learning.
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Loads a CSV file named 'heart_failure.csv' into a pandas DataFrame called data
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
# DataFrame.info() method is used to provide a DataFrame column's datatype and number of non-null entries. This gives an overview of a dataset
print(data.info())
# A Counter is a special type of dictionary provided by Python's collections module. It is used specifically for counting elements from an iterable (like a list, tuple, or string). Each element from the iterable becomes a key in the Counter dictionary, and the value for each key is the count of that element in the iterable.
# Here the instance of counter class is used how many times each unique value appears in the 'death_event' column.
print('Classes and number of values in the dataset', Counter(data['death_event']))
# Extracts the column "death_event" from the DataFrame data and sets it to target varaible "y"
y = data["death_event"]
# Extracts the relevant feature columns for the model and assign them to DataFrame x.
x = data[
    ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets',
     'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']]
# Convert categorical feature columns in x to one-hot encoded variables to prepare for model training.
x = pd.get_dummies(x)
# This splits the dataset into two parts: one for training the model (70% of the data) and one for testing the model (30% of the data).
# The function train_test_split randomly divides the rows of x and y into training sets (X_train, Y_train) and testing sets (X_test, Y_test) while maintaining the correspondence between x (features) and y (labels).
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)
ct = ColumnTransformer([("numeric", StandardScaler(),
                         ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine',
                          'serum_sodium', 'time'])])
# Apply ColumnTransformer to X_train
X_train = ct.fit_transform(X_train)
# Transform the test data using the already fitted ColumnTransformer to scale and normalize features.
X_test = ct.transform(X_test)
# Initialize a LabelEncoder to convert string labels to integers.
le = LabelEncoder()
# Fit and transform the training labels into integer labels.
Y_train = le.fit_transform(Y_train.astype(str))
# Transform the test labels to integers using the already fitted LabelEncoder.
Y_test = le.transform(Y_test.astype(str))
# Convert integer labels into a binary matrix format required for categorical crossentropy loss.
# This is necessary because the model's output layer will be using softmax activation, which requires input in this format.
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Initialize a Sequential model for a linear stack of layers.
# This type of model builds a neural network model layer by layer in a sequential manner.
model = Sequential()

# Input layer specifying the input shape to match the number of features in X_train.
# The input shape must match the shape of the data that will be fed to the model.
model.add(InputLayer(shape=(X_train.shape[1],)))

# A 'Dense' layer is a fully connected layer where each neuron receives input from all the neurons of the previous layer.
# 'ReLU' stands for Rectified Linear Unit and is a type of activation function that is commonly used in neural networks.
model.add(Dense(12, activation='relu'))

# A dense output layer with 2 output neurons and softmax activation function, for binary classification.
# The softmax function outputs a probability distribution over the 2 possible output classes.
model.add(Dense(2, activation='softmax'))

# Compile the model with the categorical crossentropy loss function, the Adam optimizer, and accuracy as a metric.
# 'categorical_crossentropy' is a loss function that is used when there are two or more label classes.
# The Adam optimizer is an algorithm for gradient-based optimization of stochastic objective functions.
# Accuracy is used as a metric for monitoring the training and testing steps. It measures the model's performance based on the percentage of correctly predicted data points.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# This trains the model with training data and labels, specifying the number of epochs and batch size, and display verbose output.
# An epoch is one complete presentation of the data set to be learned to a learning machine.
# The batch size is the number of training examples utilized in one iteration.
model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1)

# Evaluate the model on the test data and print the loss and accuracy.
loss, acc = model.evaluate(X_test, Y_test, verbose=0)

# Print the loss and accuracy to understand how well the model performed.
# 'Loss' represents how well the model did on training, with a lower loss indicating a better model.
# 'Accuracy' represents the percentage of correct predictions.
print("Loss", loss, "Accuracy:", acc)
# Predict the class probabilities for the test data.
y_estimate = model.predict(X_test, verbose=0)
# Output layer with softmax for binary classification; outputs class probabilities.
model.add(Dense(2, activation='softmax'))
# Convert predicted probabilities to class indices (0 or 1).
y_estimate = np.argmax(y_estimate, axis=1)
# Convert one-hot encoded test labels back to class indices for comparison.
y_true = np.argmax(Y_test, axis=1)
# Print the classification report to evaluate the model with precision, recall, and F1-score for each class.
print(classification_report(y_true, y_estimate))

# Initialize the ConfusionMatrixDisplay object with the confusion matrix 'cm'. This object will allow you to display matrix.
cm = confusion_matrix(y_true, y_estimate)
# Display the confusion matrix using the display object. The 'cmap=plt.cm.Blues' parameter colors the matrix in shades of blue.
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# Plot the confusion matrix using the display object
disp.plot(cmap=plt.cm.Blues)

# Set the title, x-label, and y-label
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()













