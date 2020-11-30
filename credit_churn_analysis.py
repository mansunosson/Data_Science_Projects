# Here we compare an MLP Classifier with a GLM approach to determine credit card churners. Data is from https://www.kaggle.com/sakshigoyal7/credit-card-customers
# Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neural_network
import sklearn.linear_model
import sklearn.model_selection
import sklearn.metrics
import sklearn.preprocessing
import joblib
import keras
import os

# Data Loading and Cleaning

# Load credit card churn data
bankdata = pd.read_csv("BankChurners.csv")
# Convert categoricals to inidicators
indicatorvars = pd.get_dummies(bankdata)
# Get colnames
colnames = list(indicatorvars.columns)
# Subset colnames to construct cleaned data set
clean_data = indicatorvars.drop(["CLIENTNUM", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2", "Attrition_Flag_Existing Customer", "Gender_F", "Education_Level_Uneducated", "Marital_Status_Single", "Income_Category_Less than $40K", "Card_Category_Blue"], axis = 1)
colnames = list(clean_data.columns)

# Quick visualization of variables
for i in range(len(colnames)):
    plt.figure()
    plt.xlabel(colnames[i])
    plt.hist(clean_data[colnames[i]])   
    plt.show()
    

# Split data into training and testing sets
X  = clean_data.drop("Attrition_Flag_Attrited Customer", axis = 1)
Y  = clean_data.pop("Attrition_Flag_Attrited Customer")
    
# Split data set in train and test
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1, random_state=5, stratify=Y)


# Define wrapper function for MLP Classifier
def MLP_train(X_train, Y_train, X_test, Y_test):
    
    # Create MLP Classifier model object
    classifier = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(200, ), activation='relu', solver='adam', 
                                                 alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.005, power_t=0.5, 
                                                 max_iter=1000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, 
                                                 nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
                                                 n_iter_no_change=10)  
    # Fit classifier using the training data
    classifier.fit(X_train, Y_train)
    
    # Evaluate on training data
    print('\n-- Training data --')
    pred = classifier.predict(X_train)
    accuracy = sklearn.metrics.accuracy_score(Y_train, pred)
    print('Accuracy: {0:.2f}'.format(accuracy * 100.0))
    print('Classification Report:')
    print(sklearn.metrics.classification_report(Y_train, pred))
    print('Confusion Matrix:')
    print(sklearn.metrics.confusion_matrix(Y_train, pred))
    print('')
    # Evaluate on test data
    print('\n---- Test data ----')
    pred = classifier.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(Y_test, pred)
    print('Accuracy: {0:.2f}'.format(accuracy * 100.0))
    print('Classification Report:')
    print(sklearn.metrics.classification_report(Y_test, pred))
    print('Confusion Matrix:')
    print(sklearn.metrics.confusion_matrix(Y_test, pred))




# Define wrapper function for Logistic regression
def logreg_train(X_train, Y_train, X_test, Y_test):
    
    # Create logistic regression model object
    classifier = sklearn.linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
                                            fit_intercept=True, intercept_scaling=1, class_weight=None, 
                                            random_state=None, solver='lbfgs', max_iter=1000, multi_class='auto', 
                                            verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
    
    # Fit classifier using the training data
    classifier.fit(X_train, Y_train)
    # Evaluate on training data
    print('\n-- Training data --')
    pred = classifier.predict(X_train)
    accuracy = sklearn.metrics.accuracy_score(Y_train, pred)
    print('Accuracy: {0:.2f}'.format(accuracy * 100.0))
    print('Classification Report:')
    print(sklearn.metrics.classification_report(Y_train, pred))
    print('Confusion Matrix:')
    print(sklearn.metrics.confusion_matrix(Y_train, pred))
    print('')
    # Evaluate on test data
    print('\n---- Test data ----')
    pred = classifier.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(Y_test, pred)
    print('Accuracy: {0:.2f}'.format(accuracy * 100.0))
    print('Classification Report:')
    print(sklearn.metrics.classification_report(Y_test, pred))
    print('Confusion Matrix:')
    print(sklearn.metrics.confusion_matrix(Y_test, pred))
    print('Coefficients:')
    coefs = pd.DataFrame(data = classifier.coef_, columns = list(X_train.columns))
    print(coefs)


# Evaluate the two approaches on training and testing data
MLP_train(X_train, Y_train, X_test, Y_test)   
logreg_train(X_train, Y_train, X_test, Y_test)

# The two methods have similar prediction accuracy but GLM gives estimated coefficients which provide insight...



