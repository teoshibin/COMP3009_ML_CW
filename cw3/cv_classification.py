
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold

from functions.neural_network import *
from functions.data_loading import *
from functions.data_preprocessing import *
# from functions.data_splitting import *
from functions.metrics import *
# from functions.math import *

import sys

# ---------------------- STORE AND PRINT STANDARD OUTPUT --------------------- #

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("classification.log", "w")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    

sys.stdout = Logger() # disable this to prevent print from generating .log files

# ----------------------------------- START ---------------------------------- #

import time
start = time.time()
seed = 69
tf.set_random_seed(seed)

k = 10
stratifiedKF = StratifiedKFold(n_splits = k, random_state = seed, shuffle= True)
## only turning on either one of these learning rate settings
# learning_rates = np.array([0.01, 0.005, 0.001]) # multiple models to see the behavior of lr
learning_rates = np.array([0.005]) # final selected model
max_epoch = 200
epoch_per_eval = 1 # change this to a larger value to improve performance while reducing plot details

# ----------------------- DATA LOADING & PREPROCESSING ----------------------- #

# Load dataset
x_data, y_data = loadHeartFailureDataset()
x_data = minMaxNorm(x_data)

# w_j = n / k * n_j
# weight of j class = instances / num_classes * j_class_instances
# class with low instances will increase the weight
# class with high instances will decrease in weight
# class with balance instances weight = 1
unique_label = np.unique(y_data)
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=unique_label, y=y_data)

# --------------------------- TENSOR GRAPH RELATED --------------------------- #

def weighted_binary_cross_entropy( y_true, y_pred, weight1=1, weight0=1 ) :
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight1 + (1 - y_true) * K.log(1 - y_pred) * weight0 )
    return K.mean( logloss, axis=-1)

#Network parameters
n_input = 12
n_hidden1 = 24
n_hidden2 = 12
n_hidden3 = 6
n_output = 2

#Defining the input and the output
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

# Define Network
input_layer = one_layer_perceptron(X, n_input, n_hidden1, "sigmoid")
layer_1 = one_layer_perceptron(input_layer, n_hidden1, n_hidden2, "sigmoid")
layer_2 = one_layer_perceptron(layer_1, n_hidden2, n_hidden3, "sigmoid")
logits = one_layer_perceptron(layer_2, n_hidden3, n_output, "none")
neural_network = tf.nn.softmax(logits)

# Define Loss to Optimize
loss_op = weighted_binary_cross_entropy(Y, neural_network, class_weights[0], class_weights[1])
optimizer = []
for lr in learning_rates:
    optimizer.append(tf.train.AdamOptimizer(lr).minimize(loss_op))

#Initializing the variables
init = tf.global_variables_initializer()

# --------------------------------- TRAINING --------------------------------- #

# storage
all_f1 = np.zeros((len(learning_rates), k, int(max_epoch / epoch_per_eval)))
all_acc = np.zeros((len(learning_rates), k, int(max_epoch / epoch_per_eval)))
all_test_loss = np.zeros((len(learning_rates), k, int(max_epoch / epoch_per_eval)))
all_train_loss = np.zeros((len(learning_rates), k, int(max_epoch / epoch_per_eval)))

for lr_index in range(len(learning_rates)):

    print(f"Learning Rate: {learning_rates[lr_index]}")

    for k_index, (train_index, test_index) in enumerate(stratifiedKF.split(x_data, y_data)):
        
        print(f"Fold: {k_index + 1}")

        with tf.Session() as sess:
            sess.run(init)

            train_x, test_x = x_data[train_index], x_data[test_index]
            train_y, test_y = oneHotEncoding(y_data[train_index]), oneHotEncoding(y_data[test_index])
        
            for epoch in range(max_epoch):
                sess.run(optimizer[lr_index], feed_dict={X: train_x, Y: train_y})

                #Display the epoch
                actual_epoch = epoch + 1
                if actual_epoch % epoch_per_eval == 0:

                    modIndex = int(epoch / epoch_per_eval)

                    output = neural_network.eval({X: test_x})

                    actual = tf.argmax(test_y,1)
                    predicted = tf.argmax(output,1)
                    f1 = round(tf.keras.backend.get_value(f1Score(predicted,actual)), 6)
                    f1 = 0 if np.isnan(f1) else f1
                    all_f1[lr_index][k_index][modIndex] = f1

                    correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(test_y,1))
                    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    acc = round(tf.keras.backend.get_value(acc), 6)
                    all_acc[lr_index][k_index][modIndex] = acc

                    test_loss = np.mean(tf.keras.backend.get_value(loss_op.eval({X: test_x, Y: test_y})))
                    all_test_loss[lr_index][k_index][modIndex] = test_loss

                    train_loss = np.mean(loss_op.eval({X: train_x, Y: train_y}))
                    all_train_loss[lr_index][k_index][modIndex] = train_loss
                
                    print(
                        f"Epoch: {actual_epoch}\t"
                        f"Test Acc: {acc:.6f}\t"
                        f"Test F1: {f1:.6f}\t"
                        f"Test Loss: {test_loss:.6f}\t"
                        f"Train Loss: {train_loss:.6f}\t"
                        )
                    
            sess.close()
        
# ------------------------------- PLOT FIGURES ------------------------------- #

plt.figure("fig1")
for i in range(len(learning_rates)):

    # mean results of 10 fold
    plt_test_loss = np.mean(all_test_loss[i], axis=0)
    plt_train_loss = np.mean(all_train_loss[i], axis=0)

    # plot overfitting loss over epoch
    plt.plot(plt_test_loss)
    plt.plot(plt_train_loss)

plt.xlabel(f"Epoch / {epoch_per_eval}")
plt.ylabel("Binary Cross Entropy Loss")
plt.legend(["Test Loss 1", "Train Loss 1","Test Loss 2", "Train Loss 2","Test Loss 3", "Train Loss 3"])

plt.figure("fig2")
for i in range(len(learning_rates)):

    # mean results of 10 fold
    plt_f1 = np.mean(all_f1[i], axis=0) 
    plt_acc = np.mean(all_acc[i], axis=0)

    # plot metric score over epoch
    plt.plot(plt_f1)
    plt.plot(plt_acc)
    
plt.xlabel(f"Epoch / {epoch_per_eval}")
plt.ylabel("Metric Score")    
plt.legend(["F1 Score 1", "Accuracy 1","F1 Score 2", "Accuracy 2","F1 Score 3", "Accuracy 3"])

end = time.time()
print("Time Elapsed: ", end - start)

plt.show()
                