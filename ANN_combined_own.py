# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:53:14 2020

@author: P. Akash Pattanaik
"""

# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:/Akash/Data Science/Deep Learning A-Z  ANN Udemy/P16-Artificial-Neural-Networks\
/Artificial_Neural_Networks/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # convert string category (Geography) to numeric category
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) # convert string category (Gender) to numeric category

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'),[1])], 
                        remainder='passthrough') 
# 1st arg = List of (name, transformer, column(s)) tuples specifying the transformer objects to be applied 
## name = 'one_hot_encoder'
## transformer = OneHotEncoder(categories='auto')
## columns = [1] ; 2nd column
# remainder = remaining columns should be 'drop' or 'passthrough'
X = ct.fit_transform(X).astype(float) # convert numerical category to dummy
X = X[:, 1:] # remove one dummy variable

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) # train the scaling model with train data
X_test = sc.transform(X_test) # fit the scaling model to test data


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential() # classifier is an object of class Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# units = output dimensions = (11+1)/2 = 6 nodes in the first hidden layer
# kernel_intializer = values from uniform distribution
# activation_function = rectification function
# input_dim = number of nodes in input layer

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# no need to provide input_dim as the 1st hidden layer will act as input to 2nd hidden layer

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# compiling the ANN means to apply Stochastic Gradient Desent to ANN to find the best weights
# optimizer = 'adam'; it is the best Stochastic Gradient Desent Algorithm
# The loss/objective function will be used by 'adam' algorithm to optimise the weights
# The choice of loss/objective function also depends upon the activation function.
# loss = 'binary_crossentropy' for binary classicification & 
#        'categorical_crossentropy' for more than two categories classification
# The metrics argument takes a list of metrics for evaluating the model
# metrics =['accuracy']; we have selected only one metrics 

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
# The first argument is the predictor/input variables for the ANN model
# The second argument is the actual output variable for the ANN model
# 'batch_size' defines the number of observations after which the weights are updated  
# 'epochs' number of epochs to train the model (default =1)
# An epoch is an iteration over the entire X and Y data provided


# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test) # gives the probabilty that a customer will leave the bank(exited = 1)
y_pred = (y_pred > 0.5) # 0.5 is the threshold for converting probability scores to hard outcomes
# if prob_scores > 0.5, the outcome = 1 (True) (exited = 1)
# if prob_scores < 0.5, the outcome = 0 (False) (exited = 0)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# TP = 132, TN = 1550 that implies correct predictions = 1550+132 = 1682 & FP = 45, FN = 273 
# Accuracy = 1682/2000 = 0.841 = 84 %

# Predicting a single new observation
'''
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
'''
new_data = np.array([[1, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
# we have to write two square brackets so that we create a 2d numpy array because a single bracket
# will crete a single numpy array vertically not horizontally. However, a 2d array with one row and no
# columns will create a 1d horizontal array.

# now we have to scale the data according to our training scale
new_data = sc.transform(new_data) # feature scaling

# predict whether the customer will leave the bank or not
new_pred = classifier.predict(new_data) # this will give the probability scores for customer leaving
new_pred = (new_pred>0.5) # this will give the hard outcomes


# Part 4 - Evaluating, Improving and Tuning the ANN

# "Evaluating the ANN"
# We can evaluate the performance of our model based on train dataset using k-fold cross validation
# We have to run Part-1 Data Preprocessing to obtain X_train, y_train, X_test, y_test
# The ANN model requires the keras and the k-fold cross validation requires the scikit-learn 
# Hence, we need to somehow combine these library. 
# A wrapper function is available in the keras library called KerasClassifier,
# that wraps the k-fold cross validation function from sci-kit learn.  
from keras.wrappers.scikit_learn import KerasClassifier
# Now we need the k-fold cross validation function from sci-kit learn
from sklearn.model_selection import cross_val_score

from keras.models import Sequential
from keras.layers import Dense

# Now we have to build our ANN architecture and put it inside a function 
def build_classifier():
    classifier = Sequential() # classifier is an object of class Sequential() and is local to this function
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# Now we have build a object from class KerasClassifier which will take a function as input argument
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100 )
# Above classifier variable is a global one

# Now the cross_val_score will be used which will return 10 accuries for 10-fold cross validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
print(accuracies)
# The estimator argument is nothing but the object from the wrapper functions KerasClassifier
# cv = 10 for 10-fold cross validation
# n_jobs = -1 allows for parallel computing (3-4 mins) but for this to work open spyder direclty and 
# check following:
# 1) Which graphics card do you have?  If you do not have a valid GPU you cannot run njobs=-1
# 2) To install Tensorflow 2.1.0 you have to "pip install tensorflow-gpu==2.1.0" and 
#    joblib the same "pip install joblib==0.14.1"
# 3) To see what you have installed type "conda list" at an anaconda prompt
# 4) There will be no ouput in the console for n_jobs = -1 
# n_jobs = 1 works but takes a lot of time to compute (10 mins) & there will be output in the console
mean = accuracies.mean() # mean = 83.78 %
standard_deviation = accuracies.std() # standard deviation = 1.2 % (no overfitting problem as sd is low)

# "Improving the ANN"
## Before starting this, run part-1 to obtain X_train, y_train, X_test, y_test
## # Dropout Regularization to reduce overfitting if needed 
from keras.layers import Dropout # Dropout will be applied to different layers
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer with dropout
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(rate = 0.1)) # rate = float between ) 0 and 1. fraction of input units to drop
# Always start with rate = 0.1 and then increase if overfitting still exists

# Adding the second hidden layer with dropout
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# "Tuning the ANN"
from keras.wrappers.scikit_learn import KerasClassifier
# Now we need the grid search cross validation function from sci-kit learn
from sklearn.model_selection import GridSearchCV
# grid search CV is similar to k-fold CV

from keras.models import Sequential # required to make our ANN
from keras.layers import Dense # required to make our ANN

# Now we have to build our ANN architecture and put it inside a function 
def build_classifier(optimizer): # we have to input different optimizer to the ANN architecture
    classifier = Sequential() # classifier is an object of class Sequential() and is local to this function
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# Now we have build a object from class KerasClassifier which will take a function as input argument
classifier = KerasClassifier(build_fn = build_classifier) 
# We will not supply the hyper parameters like batch size and number of epochs in above classifier
# We will find these hyper parameters after tuning

# We will create a dictionary, which will contains combinations of these hyper parameters
# Then we have to find the optimum hyper parameters
parameters = {'batch_size' : [25, 32], # common practice is to take power of the 2 as the batch size
              'epochs' : [100, 500], # we kept 100 because batch_size has changed
              'optimizer' : ['adam', 'rmsprop']} # both optimizer are based on gradient descent
# 'rmsprop is a better choice in case of RNN

# We will create a grid search object of class GridSearchCV
grid_search = GridSearchCV(estimator = classifier, # our ANN architecture
                           param_grid = parameters, # our parameters which are to be optimised
                           scoring = 'accuracy', # we will decide the best combinations by accuracy
                           cv = 10,
                           n_jobs = -1) # we will do 10-fold cross validation

# Now we will provide training data to this grid_search ovbject or fit the ANN to the data
grid_search = grid_search.fit(X = X_train, y = y_train) # 2*2*2 = 8 combinations will be tried out
# Above command will fit our ANN to the training set while running grid search to find the optimal
# hyper parameters that we are testing here

# GridSearchCV class contains some attribute to find best parameter & accuracy
best_parameter = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Homework: How to further improve accuracy
# 1. Change the architecture of neural network
# 2. Tray out different parameters combinations








 