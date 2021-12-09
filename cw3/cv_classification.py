# University of Nottingham COMP 3009 Machine Learning GROUP 18

# ANN
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold

# Custom Functions
from functions.neural_network import *
from functions.data_IO import *
from functions.data_preprocessing import *
# from functions.data_splitting import *
from functions.metrics import *
# from functions.math import *
from functions.plots import *

# Misc
import sys
import time

# ---------------------- STORE AND PRINT STANDARD OUTPUT --------------------- #

cdSubDir("cw3") # workflow code to ensure this script ran in the correct dir
sys.stdout = Logger("classification.log") # disable this to prevent print from generating .log files

# ----------------------------------- START ---------------------------------- #

start = time.time()
seed = 2021

# cross validation parameters
k = 10
stratifiedKF = StratifiedKFold(n_splits = k, random_state = seed, shuffle= True) # cv datasplitting obj

## only turning on either one of these learning rate settings
# hyper parameter tuning
learning_rates = np.around(np.arange(0.005, 0.051, 0.005),3)
# final selected model
# learning_rates = np.array([0.02])

max_epoch = 200
epoch_per_eval = 1 # change this to a larger value to improve performance while reducing plot details

# ----------------------- DATA LOADING & PREPROCESSING ----------------------- #

# Load dataset
x_data, y_data = loadHeartFailureDataset()
x_data = minMaxNorm(x_data)

# calculate weights for weighted loss later on
# w_j = n / k * n_j
# weight of j class = instances / num_classes * j_class_instances
# class with low instances will increase the weight
# class with high instances will decrease in weight
# class with balance instances weight = 1
unique_label = np.unique(y_data)
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=unique_label, y=y_data)

# --------------------------- TENSOR GRAPH RELATED --------------------------- #

def weighted_binary_cross_entropy( y_true, y_pred, weight1=1, weight0=1 ):
    """cross entropy that slightly improves performance for imbalance dataset

    Args:
        y_true (float): 2d array of dataset attributes
        y_pred (float): 2d array of one hot encoded dataset labels
        weight1 (double, optional): scalling weights for possitive class. Defaults to 1.
        weight0 (double, optional): scalling weight for negative class. Defaults to 1.

    Returns:
        tensor: loss function
    """
    
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0 )
    return K.mean( logloss, axis=-1)

def myModel():
    """construct MLP and define loss, optimizer

    Returns:
        X, Y, LR : input placeholders
        NN, LossFunction, Optimizer : components to do training
    """
    
    #Network parameters
    n_input = 12
    n_hidden1 = 24
    n_hidden2= 32
    n_hidden3 = 24
    n_output = 2

    #Defining the input and the output
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_output])

    # Define Network
    input_layer, _ = one_layer_perceptron(X, n_input, n_hidden1, "sigmoid")
    layer_1, _ = one_layer_perceptron(input_layer, n_hidden1, n_hidden2, "sigmoid")
    layer_2, _ = one_layer_perceptron(layer_1, n_hidden2, n_hidden3, "sigmoid")
    logits, _ = one_layer_perceptron(layer_2, n_hidden3, n_output, "none")
    neural_network = tf.nn.softmax(logits)

    # Define Loss to Optimize
    LR = tf.placeholder("float", [])
    loss_op = weighted_binary_cross_entropy(Y, neural_network, class_weights[0], class_weights[1])
    optimizer = tf.train.AdamOptimizer(LR).minimize(loss_op)

    return X, Y, LR, neural_network, loss_op, optimizer

# --------------------------------- TRAINING --------------------------------- #

# storage
all_test_f1 = np.zeros((len(learning_rates), k))
all_test_acc = np.zeros((len(learning_rates), k))
all_test_loss = np.ones((len(learning_rates), k)) * np.inf
all_train_loss = np.ones((len(learning_rates), k)) * np.inf

# grid search param 1
for lr_index in range(len(learning_rates)):

    print(f"Learning Rate: {learning_rates[lr_index]}")

    # k fold cross validation
    for k_index, (train_index, test_index) in enumerate(stratifiedKF.split(x_data, y_data)):
        
        print(f"Fold: {k_index + 1}")

        with tf.Session() as sess:

            # reconstruct model everytime before training
            tf.set_random_seed(seed)
            X, Y, LR, neural_network, loss_op, optimizer = myModel()
            init = tf.global_variables_initializer()
            sess.run(init)

            # split k fold dataset
            train_x, test_x = x_data[train_index], x_data[test_index]
            train_y, test_y = oneHotEncoding(y_data[train_index]), oneHotEncoding(y_data[test_index])
        
            # training
            for epoch in range(max_epoch):
                epoch_start = time.time()
                sess.run(optimizer, feed_dict={X: train_x, Y: train_y, LR: learning_rates[lr_index]})

                # evaluate every epoch_per_eval epochs
                actual_epoch = epoch + 1
                if actual_epoch % epoch_per_eval == 0:
                    
                    output = neural_network.eval({X: test_x})
                    
                    # calcuate all performance metrics and loss
                    f1 = f1Score(output,test_y)
                    correct_prediction = np.equal(np.argmax(output,1),np.argmax(test_y,1))
                    acc = np.mean(correct_prediction)
                    test_loss = np.mean(loss_op.eval({X: test_x, Y: test_y}))
                    train_loss = np.mean(loss_op.eval({X: train_x, Y: train_y}))

                    # early stopping (similar concept but instead of stopping we record down the performance)
                    if f1 > all_test_f1[lr_index][k_index]:
                        all_test_loss[lr_index][k_index] = test_loss
                        all_train_loss[lr_index][k_index] = train_loss
                        all_test_f1[lr_index][k_index] = f1
                        all_test_acc[lr_index][k_index] = acc

                    epoch_end = time.time()

                    print(
                        f"Epoch: {actual_epoch}\t"
                        f"Test Acc: {acc:.6f}\t"
                        f"Test F1: {f1:.6f}\t"
                        f"Test Loss: {test_loss:.6f}\t"
                        f"Train Loss: {train_loss:.6f}\t"
                        f"Time: {(epoch_end - epoch_start):.6f}\t"
                        )

        # reset constructed model or so called graphs
        tf.reset_default_graph()
        
# ------------------------------- PLOT FIGURES ------------------------------- #

# calculate total time taken
end = time.time()
print("Time Elapsed: ", end - start)

# only print the result of each fold if only one param config is specified
if len(learning_rates) == 1:
    print("F1 Scores:\n", all_test_f1)
    print("Loss:\n", all_test_loss)

# plot the range of loss and performance of all param configs
myBoxplot(all_test_loss, learning_rates, "Learning Rates 10-fold cv loss Distributions", "Learning Rates", "Losses")
# myBoxplot(all_train_loss, learning_rates, "Learning Rates k-fold cv loss Distributions", "Distributions", "Losses")
myBoxplot(all_test_f1, learning_rates, "Learning Rates 10-fold cv loss Distributions", "Learning Rates", "F1 Score")
# myBoxplot(all_test_acc, learning_rates, "Learning Rates k-fold cv loss Distributions", "Distributions", "Accuracy")
plt.show()
                