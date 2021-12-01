# %%
#import library stuff
# import warnings

# from numpy.lib.function_base import average
# warnings.filterwarnings('ignore', category=DeprecationWarning)
# warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
# tf.get_logger().setLevel('ERROR')

# print all available device
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# import math
# import logging
# logging.basicConfig(level=logging.ERROR)

# import matplotlib.pyplot as plt

import os 

# %%
#Network parameters
n_input = 12
n_hidden1 = 24
n_hidden2 = 12
n_hidden3 = 6
n_output = 2

#Learning parameters
learning_constant = 0.2 
number_epochs = 20000
batch_size = 1000

#Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

#DEFINING WEIGHTS AND BIASES
b1 = tf.Variable(tf.random_normal([n_hidden1]))
b2 = tf.Variable(tf.random_normal([n_hidden2]))
b3 = tf.Variable(tf.random_normal([n_hidden3]))
b4 = tf.Variable(tf.random_normal([n_output]))
w1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
w2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
w3 = tf.Variable(tf.random_normal([n_hidden2, n_hidden3]))
w4 = tf.Variable(tf.random_normal([n_hidden3, n_output]))

def multilayer_perceptron(input_d):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, w1), b1)) #f(X * W1 + b1)
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, w3), b3))
    out_layer = tf.nn.softmax(tf.add(tf.matmul(layer_3, w4),b4))
    return out_layer


# %%
def loadHeartFailureDataset():    #just get the data

    path = os.path.join("datasets","heart_failure_clinical_records_dataset.csv")
    data = np.genfromtxt(path, delimiter=",", names=True)

    #splitting data and label out
    data = data.view(np.float64).reshape((len(data), -1))
    x_data = data[:, 0:-1]
    y_data = data[:, -1]
    y_data = y_data.astype('int32')
    y_data = np.identity(2)[y_data] # one hot encoding

    # print(x_data)
    # print(y_data)
    return x_data, y_data

def minMaxNorm(data): #normalize the data
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

def validationSplit(data, percentage): #split the data between train and test
    portion = round(data.shape[0]*percentage)
    train = data[0:portion]
    test = data[portion:data.shape[0]]
    return train, test

def partitionIndex(instances_size, k): 
    partitionIndicies = np.zeros((k, 2))
    itemsPerFold = round(instances_size / k)
    for i in range(k): 
        if i == 0:
            partitionIndicies[i, :] = [(i)*itemsPerFold, (i + 1)*itemsPerFold - 1]
        else:
            partitionIndicies[i, :] = [(i)*itemsPerFold, (i + 1)*itemsPerFold - 1]
        
    partitionIndicies[k - 1,1] = instances_size - 1
    return partitionIndicies

def splitPartition(data, range_indices, partition_selection):
    range_list = np.arange(range_indices[partition_selection,0],range_indices[partition_selection,1] + 1, dtype = int)
    is_test_mat = np.zeros((data.shape[0], 1), dtype = int)
    
    for i in range_list:
        is_test_mat[i] = 1
        
    test_data = np.empty((0,data.shape[1]))
    train_data = np.empty((0,data.shape[1]))

    for i in range(data.shape[0]):
        if(is_test_mat[i] == 1):
            test_data = np.append(test_data, [data[i,:]], axis = 0)
        else:
            train_data = np.append(train_data, [data[i,:]], axis = 0)
       
    return test_data, train_data
    
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

def oneHotEncoding(y_data):
    y_data = np.identity(2)[y_data]
    return y_data

def unOneHotEncoding(y_data, axis):
    y_data = np.argmax(y_data, axis)
    return y_data

def maxCountOccur(data):
    new_list = [i for x in data for i in x]
    max_val = max(new_list, key=lambda x:new_list.count(x)) 
    return max_val

def findMean(data):
    meanValue = np.mean(data)
    return meanValue

# %%

#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
#Initializing the variables
init = tf.global_variables_initializer()

# Load dataset
x_data, y_data = loadHeartFailureDataset()
x_data = minMaxNorm(x_data)

number_of_labels = len(y_data[0]) #because of the output required 2
# train_x = x_data
# train_y = y_data

data = mergeLabel(x_data, y_data)
data = shuffleRow(data)
#train_set, test_set = validationSplit(data, 0.5)
#train_x, train_y = splitLabel(train_set, number_of_labels)
#test_x, test_y = splitLabel(test_set, number_of_labels)
# print("\nTrain X:\n", train_x)
# print("\nTrain Y:\n", train_y)
# print("\nTest X:\n", test_x)
# print("\nTest Y:\n", test_y)

with tf.Session() as sess:
    sess.run(init)

    k = 10
    inner_k = 5
    
    accuracies = np.zeros((1, k))
    f1scores = np.zeros((1, k))
    all_best_hyper_learning = np.zeros((k, inner_k))
    epochs = 200
    
    losses = np.zeros((50,200), dtype = int)
    incre = 0
    
    #setup outer partition
    outer_range_indices = partitionIndex(data.shape[0], k)
    
    for i in range(k): 
        #create outer partition
        outer_test_data, outer_train_data = splitPartition(data, outer_range_indices, i)
        
        outer_train_X, outer_train_Y = splitLabel(outer_train_data, number_of_labels)
        outer_test_X, outer_test_Y = splitLabel(outer_test_data, number_of_labels)
        
     
        #setup inner partition
        inner_range_indices = partitionIndex(outer_train_data.shape[0], inner_k)
       
        for j in range(inner_k):
            #create inner partition
            inner_test_data, inner_train_data = splitPartition(outer_train_data, inner_range_indices, j)
           
            bestF1Score = 0
            
            inner_train_X, inner_train_Y = splitLabel(inner_train_data, number_of_labels)
            inner_test_X, inner_test_Y = splitLabel(inner_test_data, number_of_labels)
    
            #grid search
            for hyper_learning in np.arange(0.01,0.06,0.01):

                #Create model
                neural_network = multilayer_perceptron(X)

                #Define loss and optimizer
                loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network,Y))
                
                optimizer = tf.train.GradientDescentOptimizer(hyper_learning).minimize(loss_op)
                
                for epoch in range(epochs):
                    sess.run(optimizer, feed_dict={X: inner_train_X, Y: inner_train_Y})
                    

                    ## need to mod this thing to gain performance
                    # pred = (neural_network)
                    # mse_loss_obj = tf.keras.losses.MSE(pred,Y)
                    # loss = mse_loss_obj.eval({X: inner_train_X, Y: inner_train_Y})
                    # losses[incre][epoch] = np.mean(loss) 
                    
                           
                output = neural_network.eval({X: inner_test_X})
                correct_prediction = tf.equal(tf.argmax(output,1),unOneHotEncoding(inner_test_Y,1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                actual = unOneHotEncoding(inner_test_Y,1)
                predicted = tf.argmax(output,1)
                f1 = f1Score(predicted,actual)
                value_f1 = tf.keras.backend.get_value(f1)
                
                incre = incre + 1
                
                if value_f1 > bestF1Score:
                    all_best_hyper_learning[i,j] = hyper_learning
                    bestF1Score = value_f1
                    
            
            print("\t Inner: %d Testsize: %d Trainsize: %d Accuracy: %f F1Score: %f BestLearning: %f"
                        %(j, inner_test_data.shape[0], inner_train_data.shape[0], tf.keras.backend.get_value(accuracy), bestF1Score,
                        all_best_hyper_learning[i,j]))
                 
        most_learning = maxCountOccur(all_best_hyper_learning[0:i + 1,:])
       
        
        optimizer = tf.train.GradientDescentOptimizer(most_learning).minimize(loss_op) 
        for epoch in range(epochs):
            sess.run(optimizer, feed_dict={X: outer_train_X, Y: outer_train_Y})
        
        output = neural_network.eval({X: outer_test_X})
        correct_prediction = tf.equal(tf.argmax(output,1),unOneHotEncoding(outer_test_Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracies[0][i] = tf.keras.backend.get_value(accuracy)
        actual = unOneHotEncoding(outer_test_Y,1)
        predicted = tf.argmax(output,1)
        f1 = f1Score(predicted,actual)
        f1scores[0][i] = tf.keras.backend.get_value(f1)
        print("Outer: %d Testsize: %d Trainsize: %d Accuracy: %f F1Score: %f MostBestLearning: %f"
                        %(i, outer_test_X.shape[0], outer_train_X.shape[0], accuracies[0][i], f1scores[0][i],
                        most_learning))

    mean_accuracy = findMean(accuracies)
    mean_f1score = findMean(f1scores)
    best_hyper_learning = maxCountOccur(all_best_hyper_learning)
    print("FinalAccuracy: %f FinalF1: %f FinalLearning: %f"
          %(mean_accuracy, mean_f1score, best_hyper_learning))
    # print(losses)
           


