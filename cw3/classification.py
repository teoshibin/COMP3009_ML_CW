# %%
import warnings

from numpy.lib.function_base import average
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import math
import logging
logging.basicConfig(level=logging.ERROR)

import matplotlib.pyplot as plt

import os 
# import pandas

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
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, w1), b1))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, w3), b3))
    out_layer = tf.nn.softmax(tf.add(tf.matmul(layer_3, w4),b4))
    return out_layer


# %%
def loadHeartFailureDataset():    

    path = os.path.join("datasets","heart_failure_clinical_records_dataset.csv")
    data = np.genfromtxt(path, delimiter=",", names=True)

    data = data.view(np.float64).reshape((len(data), -1))
    x_data = data[:, 0:-1]
    y_data = data[:, -1]
    y_data = y_data.astype('int32')
    y_data = np.identity(2)[y_data] # one hot encoding

    # print(x_data)
    # print(y_data)
    return x_data, y_data

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
    portion = round(len(data)*percentage)
    train = data[0:portion]
    test = data[portion:len(data)]
    return train, test

# %%

#Create model
neural_network = multilayer_perceptron(X)

#Define loss and optimizer
loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network,Y))
#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=neural_network,labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)

#Initializing the variables
init = tf.global_variables_initializer()

# Load dataset
x_data, y_data = loadHeartFailureDataset()
x_data = minMaxNorm(x_data)

number_of_labels = len(y_data[0])

# train_x = x_data
# train_y = y_data

data = mergeLabel(x_data, y_data)
train_set, test_set = validationSplit(data, 0.5)
train_x, train_y = splitLabel(train_set, number_of_labels)
test_x, test_y = splitLabel(test_set, number_of_labels)

# print("\nTrain X:\n", train_x)
# print("\nTrain Y:\n", train_y)
# print("\nTest X:\n", test_x)
# print("\nTest Y:\n", test_y)

with tf.Session() as sess:
    sess.run(init)


    #Training epoch
    average_losses = np.zeros(int(number_epochs / 500))
    for epoch in range(number_epochs):
        sess.run(optimizer, feed_dict={X: train_x, Y: train_y})
        
        #Display the epoch
        if epoch % 100 == 0:
            print("Epoch:", '%d' % (epoch))

        if epoch % 500 == 0:
            # Test model
            pred = (neural_network) # Apply softmax to logits
            mse_loss_obj = tf.keras.losses.MSE(pred,Y)
            loss = mse_loss_obj.eval({X: train_x, Y: train_y})
            average_losses[int(epoch / 500)] = np.mean(loss)
    
    # plot overfitting loss over epoch
    plt.plot(average_losses)
    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch / 500")
    plt.show()

    #tf.keras.evaluate(pred,batch_x)
    # output = neural_network.eval({X: test_x})
    # print("\nPrediction:\n", output[0:10])
    # print("\nActual:\n", test_y[0:10])

    # plot scatter label and prediction
    output = neural_network.eval({X: train_x})
    for i in range(len(train_y[0])):
        plt.figure(i)
        plt.plot(train_y[:, i], 'ro', output[:, i], 'bo', np.full((len(train_y), 1), 0.5))
        plt.ylabel(f"Label {i}")
        plt.xlabel('instances')
    plt.show()

    # estimated_class=tf.argmax(pred, 1)#+1e-50-1e-50
    # correct_prediction1 = tf.equal(tf.argmax(pred, 1),label)
    # accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
    
    # print(accuracy1.eval({X: batch_x}))


