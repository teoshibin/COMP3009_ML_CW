# %%
import warnings
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
number_epochs = 1000
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
number_of_labels = len(y_data[0])

train_x = x_data
train_y = y_data

# data = mergeLabel(x_data, y_data)
# train_set, test_set = validationSplit(data, 0.8)
# train_x, train_y = splitLabel(train_set, number_of_labels)
# test_x, test_y = splitLabel(test_set, number_of_labels)

print("\nTrain X:\n", train_x)
print("\nTrain Y:\n", train_y)
# print("\nTest X:\n", test_x)
# print("\nTest Y:\n", test_y)

with tf.Session() as sess:
    sess.run(init)
    #Training epoch
    for epoch in range(number_epochs):
        sess.run(optimizer, feed_dict={X: train_x, Y: train_y})
        #Display the epoch
        if epoch % 100 == 0:
            print("Epoch:", '%d' % (epoch))

    # Test model
    pred = tf.nn.softmax(neural_network) # Apply softmax to logits
    accuracy = tf.keras.losses.MSE(pred,Y)
    print("\nLoss:\n", accuracy.eval({X: train_x, Y: train_y}))
    
    #tf.keras.evaluate(pred,batch_x)
    print("\nPrediction:\n", pred.eval({X: train_x}))
    output=neural_network.eval({X: train_x})
    # plt.plot(train_y[0:10], 'ro', output[0:10], 'bo')
    # plt.ylabel('some numbers')
    # plt.show()

    # estimated_class=tf.argmax(pred, 1)#+1e-50-1e-50
    # correct_prediction1 = tf.equal(tf.argmax(pred, 1),label)
    # accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
    
    # print(accuracy1.eval({X: batch_x}))


