
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functions.neural_network import *
from functions.data_loading import *
from functions.data_preprocessing import *
from functions.data_splitting import *
from functions.metrics import *
from functions.math import *
import time

start = time.time()
tf.set_random_seed(69)

#Learning parameters
learning_constant = 0.01
max_epoch = 300

# ------------------------------- Load dataset ------------------------------- #

x_data, y_data = loadHeartFailureDataset()

# clean & split
x_data = minMaxNorm(x_data)

number_of_labels = len(y_data[0])

data = mergeLabel(x_data, y_data)

# # duplicate instances of label 1 one time to balance the dataset
# deathEventInstances = data[data[:,len(data[0]) - 1] == 1, :]

# data = np.append(data, deathEventInstances, axis=0)

data = shuffleRow(data)

train_set, test_set = validationSplit(data, 0.5)
train_x, train_y = splitLabel(train_set, number_of_labels)
test_x, test_y = splitLabel(test_set, number_of_labels)

# y_data is one hot encoded label data
# def calculateBinaryHist(y_data):
#     # for the first row calculate number of horizontal elements aka calculate total number of columns
#     end_index = len(y_data[0])
#     postives = len(y_data[y_data[:, end_index] == 1, :]

    # postives = y_data[y_data[:,len(y_data[0]) - 1] == 1, :] 

# ------------------------------- Create model ------------------------------- #

#Network parameters
n_input = 12
n_hidden1 = 24
n_hidden2 = 12
n_hidden3 = 6
n_output = 2

#Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

input_layer = one_layer_perceptron(X, n_input, n_hidden1, "sigmoid")
layer_1 = one_layer_perceptron(input_layer, n_hidden1, n_hidden2, "sigmoid")
layer_2 = one_layer_perceptron(layer_1, n_hidden2, n_hidden3, "sigmoid")
logits = one_layer_perceptron(layer_2, n_hidden3, n_output, "none")
# logits is https://stackoverflow.com/a/66804099/13405629

neural_network = tf.nn.softmax(logits)

#Define loss and optimizer

loss_op = tf.nn.weighted_cross_entropy_with_logits(labels=Y, logits=neural_network, pos_weight=2)
# loss_op = tf.reduce_mean(tf.math.squared_difference(neural_network,Y))
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y))
# loss_op = tf.keras.losses.mean_squared_logarithmic_error(neural_network, Y)
# loss_op = tf.keras.losses.categorical_crossentropy(Y, neural_network)
# loss_op = tf.keras.losses.binary_crossentropy(Y, neural_network)
# loss_op = tf.keras.losses.squared_hinge(Y, neural_network)
# loss_op = tf.keras.losses.kullback_leibler_divergence(Y, neural_network)

# optimizer = tf.train.GradientDescentOptimizer(learning_constant).minimize(loss_op)
optimizer = tf.train.AdamOptimizer(learning_constant).minimize(loss_op)

#Initializing the variables
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    # average_losses = np.zeros(int(max_epoch / 10))

    # termination stuff
    terminate_tolerence = -1
    current_tolerence = terminate_tolerence
    best_f1 = 0
    best_model = neural_network
    best_epoch = 0

    errors =np.zeros(int(max_epoch / 1))
    #Training epoch
    for epoch in range(max_epoch):
        sess.run(optimizer, feed_dict={X: train_x, Y: train_y})
        

        #Display the epoch
        if epoch % 1 == 0:

            output = neural_network.eval({X: test_x})
            actual = unOneHotEncoding(test_y,1)
            predicted = tf.argmax(output,1)
            f1 = round(tf.keras.backend.get_value(f1Score(predicted,actual)), 4)
            errors[int(epoch / 1)] = f1

            # termination criteria
            if f1 > best_f1:
                best_f1 = f1
                best_model = neural_network
                best_epoch = epoch
                current_tolerence = terminate_tolerence
            else:
                if current_tolerence != 0:
                    current_tolerence = current_tolerence - 1
                else:
                    break
                
            print("Epoch: %d F1: %f Tolerance: %d" % (epoch, f1, current_tolerence))

        # if epoch % 10 == 0:
        #     # Test model
        #     pred = (neural_network) # Apply softmax to logits
        #     mse_loss_obj = tf.keras.losses.MSE(pred,Y)
        #     loss = mse_loss_obj.eval({X: train_x, Y: train_y})
        #     average_losses[int(epoch / 10)] = np.mean(loss)


    # plot overfitting loss over epoch
    plt.plot(errors)
    plt.ylabel("Bin Cross Entropy Loss")
    plt.xlabel("Epoch / 1")
    plt.show()

    #tf.keras.evaluate(pred,batch_x)
    # output = neural_network.eval({X: test_x})
    # print("\nPrediction:\n", output[0:10])
    # print("\nActual:\n", test_y[0:10])

    # plot scatter label and prediction
        
    # for i in range(len(train_y[0])):
    #     plt.figure(i)
    #     plt.plot(train_y[:, i], 'ro', output[:, i], 'bo', np.full((len(train_y), 1), 0.5))
    #     plt.ylabel(f"Label {i}")
    #     plt.xlabel('instances')
    # plt.show()
   
    output = best_model.eval({X: test_x})
    # estimated_class=tf.argmax(pred, 1)#+1e-50-1e-50
   
    correct_prediction1 = tf.equal(tf.argmax(output,1),unOneHotEncoding(test_y,1))
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

    actual = unOneHotEncoding(test_y,1)
    predicted = tf.argmax(output,1)
    f1 = f1Score(predicted,actual)
 

    # print(unOneHotEncoding(test_y,1))
    # print(tf.keras.backend.get_value(tf.argmax(output,1)))
    print(tf.keras.backend.get_value(f1))
    print(tf.keras.backend.get_value(accuracy1))
   
    end = time.time()
    print("Total Time Elapsed: ", end - start)

    print(f"best_epoch: {best_epoch} best_f1: {best_f1}")
