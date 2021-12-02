# Result
# CAUTION THIS TOOK 11H 30M TO COMPLETE
# FinalAccuracy: 0.799195  FinalF1: 0.700898  FinalLearning: 0.01

import os
import numpy as np
import tensorflow as tf
from functions.neural_network import *
from functions.data_loading import *
from functions.data_preprocessing import *
from functions.data_splitting import *
from functions.metrics import *
from functions.math import *
tf.set_random_seed(69)

# --------------------------- TENSOR GRAPH RELATED --------------------------- #

#Defining the input and the output
X = tf.placeholder("float", [None, 12])
Y = tf.placeholder("float", [None, 2])

def multilayer_perceptron(input_x):

    input_layer = one_layer_perceptron(input_x, 12, 24, "sigmoid")
    layer_1 = one_layer_perceptron(input_layer, 24, 12, "sigmoid")
    layer_2 = one_layer_perceptron(layer_1, 12, 6, "sigmoid")
    out_layer = one_layer_perceptron(layer_2, 6, 2, "softmax")
    return out_layer

#Create model
neural_network = multilayer_perceptron(X)

#Define loss and optimize
loss_op = tf.keras.losses.binary_crossentropy(Y,neural_network)

#Define optimizer
optimizer = []
learning_rates = np.arange(0.01, 0.06, 0.01)
for lr in learning_rates:
    optimizer.append(tf.train.AdamOptimizer(lr).minimize(loss_op))

#Initializing the variables
init = tf.global_variables_initializer()

# ----------------------- DATA LOADING & PREPROCESSING ----------------------- #

# Load dataset
x_data, y_data = loadHeartFailureDataset()
x_data = minMaxNorm(x_data)

number_of_labels = len(y_data[0]) #because of the output required 2

data = mergeLabel(x_data, y_data)
data = shuffleRow(data)

# ----------------------------- CROSS VALIDATION ----------------------------- #


# parameter
k = 10
inner_k = 5
max_epochs = 200
dispEpochEveryN = 5 # evaluation every N epoch
terminate_tolerence = 15 # termination stuff

# storage
accuracies = np.zeros((1, k))
f1scores = np.zeros((1, k))
all_best_hyper_learning = np.zeros((k, inner_k), dtype = int)

# setup outer partition
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
        for hyper_learning_loop in range(5):
               
            with tf.Session() as sess:
                sess.run(init)
                
                # termination stuff
                current_tolerence = terminate_tolerence
                best_f1 = 0
                best_model = neural_network
                best_epoch = 0
                # allf1 =np.zeros(int(max_epochs / dispEpochEveryN))

                for epoch in range(max_epochs):
                    sess.run(optimizer[hyper_learning_loop], feed_dict={X: inner_train_X, Y: inner_train_Y})
                    
                    # need to mod this thing to gain performance
                    # pred = (neural_network)
                    # mse_loss_obj = tf.keras.losses.MSE(pred,Y)
                    # loss = mse_loss_obj.eval({X: inner_train_X, Y: inner_train_Y})
                    # losses[incre][epoch] = np.mean(loss) 
                    
                    #Display the epoch
                    if epoch % dispEpochEveryN == 0:

                        output = neural_network.eval({X: inner_test_X})
                        actual = unOneHotEncoding(inner_test_Y,1)
                        predicted = tf.argmax(output,1)
                        f1 = round(tf.keras.backend.get_value(f1Score(predicted,actual)), 6)
                        f1 = 0 if np.isnan(f1) else f1
                        # allf1[int(epoch / dispEpochEveryN)] = f1

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
                        
                        print(f"\tEpoch: {epoch}\t" 
                              f"F1: {f1:.6f}\t"
                              f"Tolerance: {current_tolerence} "
                               ) 
                        
                print(f"\tSelected Epoch: {best_epoch}\t" 
                        f"F1: {best_f1:.6f} "
                        )
                
                output = best_model.eval({X: inner_test_X})
                correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(inner_test_Y,1))
                f1 = tf.keras.backend.get_value(f1Score(tf.argmax(output,1), tf.argmax(inner_test_Y,1)))
                f1 = 0 if np.isnan(f1) else f1
                accuracy = tf.keras.backend.get_value(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
                
                # incre = incre + 1
                
                if f1 > bestF1Score:
                    all_best_hyper_learning[i,j] = hyper_learning_loop
                    bestF1Score = f1
                        
                print(f"\tInner: {j}\t"      
                        f"Current Lr: {learning_rates[hyper_learning_loop]}\t" 
                        f"Testsize: {inner_test_data.shape[0]}\t" 
                        f"Trainsize: {inner_train_data.shape[0]}\t" 
                        f"Acc: {accuracy:.6f}\t" 
                        f"F1: {bestF1Score:.6f}\t" 
                        f"Best Lr: {learning_rates[all_best_hyper_learning[i,j]]} "
                        )
                
                sess.close()
                    
    with tf.Session() as sess:
        sess.run(init)
        most_learning = maxCountOccur(all_best_hyper_learning[0:i + 1,:])
        
        for epoch in range(max_epochs):
            sess.run(optimizer[most_learning], feed_dict={X: outer_train_X, Y: outer_train_Y})
        
        output = neural_network.eval({X: outer_test_X})
        correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(outer_test_Y,1))
        accuracies[0][i] = tf.keras.backend.get_value(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
        f1scores[0][i] = tf.keras.backend.get_value(f1Score(tf.argmax(output,1),tf.argmax(outer_test_Y,1)))
        
        print(f"Outer: {i} "
              f" Testsize: {outer_test_X.shape[0]} "
              f" Trainsize: {outer_train_X.shape[0]} "
              f" Accuracy: {accuracies[0][i]:.6f} "
              f" F1Score: {f1scores[0][i]:.6f} "
              f" MostBestLearning: {learning_rates[most_learning]} "
               )
    
mean_accuracy = findMean(accuracies)
mean_f1score = findMean(f1scores)
best_hyper_learning = maxCountOccur(all_best_hyper_learning)
print(f"FinalAccuracy: {mean_accuracy:.6f} "
      f" FinalF1: {mean_f1score:.6f} " 
      f" FinalLearning: {learning_rates[best_hyper_learning]} "
       )

# Result
# CAUTION THIS TOOK 11H 30M TO COMPLETE
# FinalAccuracy: 0.799195  FinalF1: 0.700898  FinalLearning: 0.01