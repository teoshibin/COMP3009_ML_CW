
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices())

from functions.neural_network import *
from functions.data_IO import *
from functions.data_preprocessing import *
# from functions.data_splitting import *
from functions.metrics import *
# from functions.math import *
from functions.plots import *

import sys
import time

# ---------------------- STORE AND PRINT STANDARD OUTPUT --------------------- #

cdSubDir("cw3")
sys.stdout = Logger("regression.log") # disable this to prevent print from generating .log files

# ----------------------------------- START ---------------------------------- #

start = time.time()
seed = 2021

k = 10
KF = KFold(n_splits = k, random_state = seed, shuffle= True)

## only turning on either one of these learning rate settings
## Full Run of these configs requires 123s * 5 * 5 time = 51.25mins
learning_rates = np.around(np.arange(0.005, 0.05, 0.01),3)
weight_decays = np.around(np.arange(0.005, 0.05, 0.01),3)

## Full Run of one cofiguration require 123 seconds
#learning_rates = np.around(np.array([0.035]),3)
#weight_decays = np.around(np.array([0.025]),3)

max_epoch = 5000
epoch_per_eval = 1 # change this to a larger value to improve performance while reducing plot details
eval_per_print = 100 # print every x eval, to reduce printed logs

# ----------------------- DATA LOADING & PREPROCESSING ----------------------- #

# Load dataset
x_data, y_data = loadConcreteDataset()
x_data = minMaxNorm(x_data)

# --------------------------- TENSOR GRAPH RELATED --------------------------- #

def myModel(weight_decay = 0.01):
    #Network parameters
    n_input = 8
    n_hidden1 = 48
    n_hidden2 = 32
    n_hidden3 = 16
    n_hidden4 = 32
    n_hidden5 = 8
    n_output = 1

    #Defining the input and the output
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_output])

    # Define Network
    input_layer, w1 = one_layer_perceptron(X, n_input, n_hidden1, "relu")
    layer_1, w2 = one_layer_perceptron(input_layer, n_hidden1, n_hidden2, "relu")
    layer_2, w3= one_layer_perceptron(layer_1, n_hidden2, n_hidden3, "relu")
    layer_3, w4= one_layer_perceptron(layer_2, n_hidden3, n_hidden4, "relu")
    layer_4, w5= one_layer_perceptron(layer_3, n_hidden4, n_hidden5, "relu")
    neural_network, w6 = one_layer_perceptron(layer_4, n_hidden5, n_output, "none")

    # Define Loss to Optimize
    LR = tf.placeholder("float", [])
    regularizer = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4) + tf.nn.l2_loss(w5)  + tf.nn.l2_loss(w6)
    loss_op = tf.math.sqrt(tf.reduce_mean(tf.math.squared_difference(neural_network,Y)) + weight_decay * regularizer)
    optimizer = tf.train.AdamOptimizer(LR, 0.799, 0.999).minimize(loss_op)

    # #Network parameters
    # structure = np.array([
    #     (8, ""),
    #     (48, "relu"),
    #     (32, "relu"),
    #     (16, "relu"),
    #     (32, "relu"),
    #     (8, "relu"),
    #     (1, "none"),
    #     ])

    # # Define Network
    # neural_network, regularizer, X, Y = multi_layer_perceptron(structure)

    # # Define Loss to Optimize
    # LR = tf.placeholder("float", [])
    # loss_op = tf.math.sqrt(tf.reduce_mean(tf.math.squared_difference(neural_network,Y)) + weight_decay * regularizer)
    # optimizer = tf.train.AdamOptimizer(LR).minimize(loss_op)

    return X, Y, LR, neural_network, loss_op, optimizer

# --------------------------------- TRAINING --------------------------------- #

# storage for best settings in each combination of hyper-parameter.
all_test_rmse = np.ones((len(weight_decays),len(learning_rates), k)) * np.inf
all_test_loss = np.ones((len(weight_decays),len(learning_rates), k)) * np.inf
all_train_loss = np.ones((len(weight_decays),len(learning_rates), k)) * np.inf

for weight_index in range(len(weight_decays)):
    for lr_index in range(len(learning_rates)):
    
        print(f"Learning Rate: {learning_rates[lr_index]}")
    
        for k_index, (train_index, test_index) in enumerate(KF.split(x_data, y_data)):
            
            print(f"Fold: {k_index + 1}")
    
            # config=tf.ConfigProto(log_device_placement=True) # place this into tf.session to show u're using gpu or cpu
            with tf.Session() as sess:
    
                # reset model
                tf.set_random_seed(seed)
                X, Y, LR, neural_network, loss_op, optimizer = myModel(weight_decays[weight_index])
                init = tf.global_variables_initializer()
                sess.run(init)
    
                train_x, test_x = x_data[train_index], x_data[test_index]
                train_y, test_y = np.reshape(y_data[train_index],(-1,1)), np.reshape(y_data[test_index],(-1,1))
                
                # train_losses = []
                # test_losses = []
                # rmses =[]
                
                epoch_start = time.time()
                for epoch in range(max_epoch):
                    sess.run(optimizer, feed_dict={X: train_x, Y: train_y, LR: learning_rates[lr_index]})
    
                    # eval every epoch_per_eval epochs
                    actual_epoch = epoch + 1
                    if actual_epoch % epoch_per_eval == 0:
    
                        output = neural_network.eval({X: test_x})
                        
                        # calcuate all metrics
    
                        rmse = rmseScore(output,test_y)
                        # rmses.append(rmse)
    
                        test_loss = np.mean(loss_op.eval({X: test_x, Y: test_y}))
                        # test_losses.append(test_loss)
        
                        train_loss = np.mean(loss_op.eval({X: train_x, Y: train_y}))
                        # train_losses.append(train_loss)
                        
                        if rmse < all_test_rmse[weight_index][lr_index][k_index]:
                            all_test_loss[weight_index][lr_index][k_index] = test_loss
                            all_train_loss[weight_index][lr_index][k_index] = train_loss
                            all_test_rmse[weight_index][lr_index][k_index] = rmse
        
                        # only print the result of eval every eval_per_print evals
                        if actual_epoch % (epoch_per_eval * eval_per_print) == 0:
                            epoch_end = time.time()
                            print(
                                f"Epoch: {actual_epoch}\t"
                                f"Test RMSE: {rmse:.6f}\t"
                                f"Test Loss: {test_loss:.6f}\t"
                                f"Train Loss: {train_loss:.6f}\t"
                                f"Time: {(epoch_end - epoch_start):.6f}\t"
                                )
                            epoch_start = time.time()
                # # Trainloss and test loss compare graph to check overfitting and underfitting.
                # if k_index == 0:
                #     plt.plot(train_losses[max_epoch*0.1:])
                #     plt.plot(test_losses[1000:])
                #     plt.title("Weight Decay:"+ str(weight_decays[weight_index])+ "learning rate:" + str(learning_rates[lr_index]) +" Best RMSE:" + str(all_test_rmse[weight_index][lr_index][k_index]))
                #     plt.legend(['train', 'validation'], loc='upper left')
                #     plt.figure()
    
            # reset model
            tf.reset_default_graph()
        
# ------------------------------- PLOT FIGURES ------------------------------- #

end = time.time()
print("Time Elapsed: ", end - start)

if len(learning_rates) == 1 and len(weight_decays) == 1:
    print("RMSE:\n", all_test_rmse)
    print("Loss:\n", all_test_loss)

for weight_index in range(len(weight_decays)):
    myBoxplot(all_test_loss[weight_index], learning_rates, 
              "Learning Rates 10-fold cv loss Distributions, weight decay = "+str(weight_decays[weight_index]), 
              "Learning Rates", "Losses")
    # plt.savefig(os.path.join('results', 'regression', f'Loss_wd_{weight_decays[weight_index]}.png'))
    # myBoxplot(all_train_loss, learning_rates, "Learning Rates k-fold cv loss Distributions", "Distributions", "Losses")
    myBoxplot(all_test_rmse[weight_index], learning_rates, 
              "Learning Rates 10-fold cv loss Distributions, weight decay = "+str(weight_decays[weight_index]), 
              "Learning Rates", "RMSE Score")
    # plt.savefig(os.path.join('results', 'regression', f'RMSE_wd_{weight_decays[weight_index]}.png'))
    # myBoxplot(all_test_acc, learning_rates, "Learning Rates k-fold cv loss Distributions", "Distributions", "Accuracy")
plt.show()