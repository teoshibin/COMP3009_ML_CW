import numpy as np

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