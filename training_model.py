import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf


dftrain = pd.read_csv('train.csv') # training data
dfeval = pd.read_csv('eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function(): #return a lambda
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) #create tf.data.Dataset object with features and corresponding labels
    if shuffle:
      ds = ds.shuffle(1000) #randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs) #split dataset into batches of size batch_size, and repeat for number of epochs
    return ds #return batch
  return input_function # return function object

#create instances of input functions
train_input_fn = make_input_fn(dftrain, y_train) 
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

#create the model
#uses estimator module from tensorflow
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn) # train
result = linear_est.evaluate(eval_input_fn) #get model metrics by testing on testing data

#print(result)
#print(result['accuracy'])

# note that tensorflow models are built to make predictions on large datasets. They are not great at making predictions on one piece of datum
result = list(linear_est.predict(eval_input_fn))
clear_output()
print("--------------------------------------------------------------------------------------------------------------")
print('Data was collected on 625 passengers on the RMS Titanic. Give a number n to see information on the passenger, \ntheir likelihood of survival given this information, and whether or not they survived.')
print("--------------------------------------------------------------------------------------------------------------")
n = input("Enter an integer in the range [0,625):")
n_int = int(n)

print ("\nData collected for Passenger " + n + ":\n")
print(dfeval.loc[n_int])
print("\nChance of survival:")
#print(result[n]['probabilities'][1]) # chance of survival
chances = result[n_int]['probabilities'][1]
pr = str(chances*100)
print(pr + " %\n")
#print(y_eval.loc[n])  # labels for this data set; did this person actually survive?
if y_eval.loc[n_int]== 0:
  print("Passenger " + n + " did not survive.")
else:
  print("Passenger " + n + " survived")
# print(result[0]['probabilities'][0]) #chance of death

