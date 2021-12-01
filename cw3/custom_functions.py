
import numpy as np
import os
import tensorflow as tf

def loadHeartFailureDataset():    

    path = os.path.join("datasets","heart_failure_clinical_records_dataset.csv")
    data = np.genfromtxt(path, delimiter=",", names=True)

    data = data.view(np.float64).reshape((len(data), -1))
    x_data = data[:, 0:-1]
    y_data = data[:, -1]
    y_data = y_data.astype('int32')
    #no_data = y_data
    y_data = np.identity(2)[y_data] # one hot encoding

    # print(x_data)
    # print(y_data)
    return x_data, y_data

def oneHotEncoding(y_data):
    y_data = np.identity(2)[y_data]
    return y_data

def unOneHotEncoding(y_data, axis):
    y_data = np.argmax(y_data, axis)
    return y_data

def minMaxNorm(data):
    spacing = 5
    pct_min = np.percentile(data, spacing, axis=0)
    pct_max = np.percentile(data, 100 - spacing, axis=0)
    norm_data = np.zeros(data.shape)
    for i in range(len(data[0])):
        norm_data[:, i] = (data[:,i] - pct_min[i]) / (pct_max[i] - pct_min[i])

    norm_data[norm_data > 1] = 1
    norm_data[norm_data < 0] = 0

    return norm_data

def splitLabel(data, number_of_labels):
    number_of_columns = len(data[0])

    x_data = data[:, 0:-number_of_labels]
    y_data = data[:, number_of_columns - number_of_labels : number_of_columns]
    return x_data, y_data

def mergeLabel(x_data, y_data):
    return np.column_stack((x_data, y_data))

def validationSplit(data, percentage):
    portion = round(data.shape[0]*percentage)
    train = data[0:portion]
    test = data[portion:data.shape[0]]
    return train, test

def shuffleRow(data):
    np.random.seed(2)
    np.random.shuffle(data)
    return data

def f1Score(predicted, actual):
    TP = tf.count_nonzero(predicted * actual)
    TN = tf.count_nonzero((predicted - 1) * (actual - 1))
    FP = tf.count_nonzero(predicted * (actual - 1))
    FN = tf.count_nonzero((predicted - 1) * actual)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return f1