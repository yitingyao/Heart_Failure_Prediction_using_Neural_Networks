# Heart Failure Prediction using Neural Networks
## Project Objective
Cardiovascular diseases are the leading cause of death globally, and early detection and management through machine learning can significantly improve outcomes. The model developed in this project assists in predicting the likelihood of death due to heart failure from various clinical parameters. 

## Technical Requirements
  - Python 3.9 or higher (must be compatible with TensorFlow
  - Pands
  - NumPy
  - Matplotlib
  - scikit-learn
  - TensorFlow
    
## Technical Description
This project implements a neural network to predict the likelihood of mortality due to heart disease using a Kaggle dataset with 12 clinical predictors including age, anemia, and diabetes. Utilizing TensorFlow's Sequential modeling, the network comprises an input layer, a ReLU-activated hidden layer, and a softmax output layer tailored for classification tasks—outputting probabilities for mortality outcomes. The model was compiled with categorical crossentropy loss and the Adam optimizer, which together optimize the model's parameters to minimize prediction errors by efficiently navigating the problem's loss landscape and adjusting weights to reduce the discrepancy between predicted and actual outputs.

The model's performance is gauged through several key indicators:
* A confusion matrix reveals the counts of true positives (correctly predicted events), true negatives (correctly predicted non-events), false positives (incorrectly predicted events), and false negatives (missed actual events).
* Accuracy measures the proportion of total predictions that are correct.
* Precision assesses the accuracy of positive predictions.
* Recall, or sensitivity, calculates the rate at which actual positives are correctly identified.
* The F1-score is a harmonic mean of precision and recall, providing a single measure for test accuracy.

## Results
The model's performance has an overall accuracy of 82%, thus it correctly predicts a large majority of the outcomes. In particular, the model is better at predicting non-events (true negatives), with a precision of 85% and a recall of 90%. However, it struggles more with correctly identifying events (true positives), with a precision of 75% and a recall of 64%, as shown in the classification report. Further model training in the future, such as hyperparameter could potentially enhance its accuracy. 

## Data Sources

This project utilizes the Heart Failure Clinical Records Dataset available on Kaggle, which comprises medical records of heart failure patients with 12 clinical features.

- **Dataset Name**: Heart Failure Prediction Dataset
- **Access**: The dataset can be accessed at the following link: [Heart Failure Clinical Records Data](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data).

