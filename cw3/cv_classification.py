import os
import numpy as np
import tensorflow as tf
tf.set_random_seed(69)

#Defining the input and the output
X = tf.placeholder("float", [None, 12])
Y = tf.placeholder("float", [None, 2])

#DEFINING WEIGHTS AND BIASES
def one_layer_perceptron(input_x, input_size, output_size, activation_type):

    b = tf.Variable(tf.random_normal([output_size]))
    w = tf.Variable(tf.random_normal([input_size, output_size]))

    if activation_type == "sigmoid":
        layer = tf.nn.sigmoid(tf.add(tf.matmul(input_x, w), b))

    elif activation_type == "relu":
        layer = tf.nn.relu(tf.add(tf.matmul(input_x, w), b))

    elif activation_type == "softmax":
        layer = tf.nn.softmax(tf.add(tf.matmul(input_x, w), b))

    return layer

def multilayer_perceptron(input_x):

    input_layer = one_layer_perceptron(input_x, 12, 24, "sigmoid")
    layer_1 = one_layer_perceptron(input_layer, 24, 12, "sigmoid")
    layer_2 = one_layer_perceptron(layer_1, 12, 6, "sigmoid")
    out_layer = one_layer_perceptron(layer_2, 6, 2, "softmax")
    return out_layer

# #Defining the input and the output
# X = tf.placeholder("float", [None, 12])
# Y = tf.placeholder("float", [None, 2])

# #DEFINING WEIGHTS AND BIASES
# b1 = tf.Variable(tf.random_normal([24]))
# b2 = tf.Variable(tf.random_normal([12]))
# b3 = tf.Variable(tf.random_normal([6]))
# b4 = tf.Variable(tf.random_normal([2]))
# w1 = tf.Variable(tf.random_normal([12, 24]))
# w2 = tf.Variable(tf.random_normal([24, 12]))
# w3 = tf.Variable(tf.random_normal([12, 6]))
# w4 = tf.Variable(tf.random_normal([6, 2]))

# def multilayer_perceptron(input_d):
#     layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_d, w1), b1)) #f(X * W1 + b1)
#     layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w2), b2))
#     layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, w3), b3))
#     out_layer = tf.nn.softmax(tf.add(tf.matmul(layer_3, w4),b4))
#     return out_layer

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


#Create model
neural_network = multilayer_perceptron(X)

#Define loss and optimize
loss_op = tf.keras.losses.binary_crossentropy(Y,neural_network)

#Define optimizer
optimizer = []
learning_rate = 0.01
for i in range(5):
    optimizer.append(tf.train.AdamOptimizer(learning_rate).minimize(loss_op))
    learning_rate = learning_rate + 0.01

#Initializing the variables
init = tf.global_variables_initializer()

# Load dataset
x_data, y_data = loadHeartFailureDataset()
x_data = minMaxNorm(x_data)

number_of_labels = len(y_data[0]) #because of the output required 2

data = mergeLabel(x_data, y_data)
data = shuffleRow(data)

#parameter
k = 10
inner_k = 5
max_epochs = 200
terminate_tolerence = 15 # termination stuff

accuracies = np.zeros((1, k))
f1scores = np.zeros((1, k))
all_best_hyper_learning = np.zeros((k, inner_k), dtype = int)

#losses = np.zeros((50,200), dtype = int)
#incre = 0

dispEpochEveryN = 5

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
                        f1 = round(tf.keras.backend.get_value(f1Score(predicted,actual)), 4)
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
                        
                        print(f"\t\tEpoch: {epoch} " 
                              f" F1: {f1:.6f} "
                              f" Tolerance: {current_tolerence} "
                               ) 
                        
                print(f"\t Epoch: {best_epoch}" 
                      f" Best_F1: {best_f1:.6f} "
                       )
                
                output = best_model.eval({X: inner_test_X})
                correct_prediction = tf.equal(tf.argmax(output,1),tf.argmax(inner_test_Y,1))
                f1 = tf.keras.backend.get_value(f1Score(tf.argmax(output,1), tf.argmax(inner_test_Y,1)))
                accuracy = tf.keras.backend.get_value(tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))
                
                # incre = incre + 1
                
                if f1 > bestF1Score:
                    all_best_hyper_learning[i,j] = hyper_learning_loop
                    bestF1Score = f1
                        
                print(f"\t Inner: {j} "      
                      f" Testsize: {inner_test_data.shape[0]} " 
                      f" Trainsize: {inner_train_data.shape[0]} " 
                      f" Accuracy: {accuracy:.6f} " 
                      f" F1Score: {bestF1Score:.6f} " 
                      f" BestLearning: {all_best_hyper_learning[i,j]} "
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
              f" MostBestLearning: {most_learning} "
               )
    
mean_accuracy = findMean(accuracies)
mean_f1score = findMean(f1scores)
best_hyper_learning = maxCountOccur(all_best_hyper_learning)
print(f"FinalAccuracy: {mean_accuracy:.6f} "
      f" FinalF1: {mean_f1score:.6f} " 
      f" FinalLearning: {best_hyper_learning} "
       )
         
