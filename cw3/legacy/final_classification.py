
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

learning_rate = 0.006
max_epoch = 200
epoch_per_eval = 1
terminate_tolerence = -1

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

# Define Loss to Optimize
loss_op = tf.keras.losses.binary_crossentropy(Y,neural_network)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

#Initializing the variables
init = tf.global_variables_initializer()

# ----------------------- DATA LOADING & PREPROCESSING ----------------------- #

# Load dataset
x_data, y_data = loadHeartFailureDataset()
x_data = minMaxNorm(x_data)

number_of_labels = len(y_data[0])

data = mergeLabel(x_data, y_data)
data = shuffleRow(data)

train_set, test_set = validationSplit(data, 0.8) # train test split
train_x, train_y = splitLabel(train_set, number_of_labels)
test_x, test_y = splitLabel(test_set, number_of_labels)
# train_x, train_y = splitLabel(data, number_of_labels)

# --------------------------------- TRAINING --------------------------------- #

with tf.Session() as sess:
    sess.run(init)

    saver=tf.train.Saver()
    save_path='death_event_model/'

    # average_losses = np.zeros(int(max_epoch / 10))

    # termination stuff
    current_tolerence = terminate_tolerence
    best_loss = np.Inf
    best_model = neural_network
    best_epoch = 0

    # storage
    all_f1 =np.zeros(int(max_epoch / epoch_per_eval))
    all_acc =np.zeros(int(max_epoch / epoch_per_eval))
    test_loss =np.zeros(int(max_epoch / epoch_per_eval))
    train_loss =np.zeros(int(max_epoch / epoch_per_eval))

    #Training epoch
    for epoch in range(max_epoch):
        sess.run(optimizer, feed_dict={X: train_x, Y: train_y})
        
        #Display the epoch
        if epoch % epoch_per_eval == 0:

            modIndex = int(epoch / epoch_per_eval)

            output = neural_network.eval({X: test_x})
            actual = tf.argmax(test_y,1)
            predicted = tf.argmax(output,1)
            f1 = round(tf.keras.backend.get_value(f1Score(predicted,actual)), 6)
            f1 = 0 if np.isnan(f1) else f1
            all_f1[modIndex] = f1

            correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(test_y,1))
            all_acc[modIndex] = tf.keras.backend.get_value(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))

            loss1 = np.mean(tf.keras.backend.get_value(loss_op.eval({X: test_x, Y: test_y})))
            test_loss[modIndex] = loss1
            loss2 = loss_op.eval({X: train_x, Y: train_y})
            train_loss[modIndex] = np.mean(loss2)

            # termination criteria
            if loss1 < best_loss:
                best_loss = loss1
                best_epoch = epoch
                saver.save(sess=sess,save_path=save_path)
                current_tolerence = terminate_tolerence
            else:
                if current_tolerence != 0:
                    current_tolerence = current_tolerence - 1
                else:
                    break
                
            print(f"Epoch: {epoch}\t"
                  f"F1: {f1:.6f}\t"
                  f"Loss: {test_loss[int(epoch / epoch_per_eval)]:.6f}\t"
                  f"Tolerance: {current_tolerence}")
   

    print(f"Best Epoch: {best_epoch}\t"
          f"Best Lost: {best_loss}")

    saver.restore(sess=sess,save_path=save_path)
    output = neural_network.eval({X: test_x})
    correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(test_y,1))
    accuracy = tf.keras.backend.get_value(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))

    actual = tf.argmax(test_y,1)
    predicted = tf.argmax(output,1)
    f1 = tf.keras.backend.get_value(f1Score(predicted,actual))
 
    print(f"Acc: {accuracy:.6f}\t"
          f"F1: {f1:.6f}\t"
          f"Loss: {test_loss[int(best_epoch / epoch_per_eval)]:.6f}")
   
    end = time.time()
    print("Total Time Elapsed: ", end - start)

    # print(tf.keras.backend.get_value(test_loss))

    # plot overfitting loss over epoch
    plt.figure("fig1")
    plt.plot(test_loss)
    plt.plot(train_loss)
    plt.axvline(int(best_epoch / epoch_per_eval), lineStyle='--', color='r')
    plt.xlabel(f"Epoch / {epoch_per_eval}")
    plt.ylabel("Binary Cross Entropy Loss")
    plt.legend(["Test Loss", "Train Loss", "Selected Model"])

    plt.figure("fig2")
    plt.plot(all_f1)
    plt.plot(all_acc)
    plt.axvline(int(best_epoch / epoch_per_eval), lineStyle='--', color='r')
    plt.xlabel(f"Epoch / {epoch_per_eval}")
    plt.ylabel("Metric Score")    
    plt.legend(["F1 Score", "Accuracy", "Selected Model"])

    plt.show()