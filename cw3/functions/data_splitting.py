import numpy as np

def splitLabel(data, number_of_labels):
    """split attributes and labels

    Args:
        data (2d floats): dataset
        number_of_labels (int): number of column labels at the end of dataset

    Returns:
        (2d floats, 2d floats): attributes columns and label columns
    """
    
    number_of_columns = len(data[0])
    
    x_data = data[:, 0:-number_of_labels]
    y_data = data[:, number_of_columns - number_of_labels : number_of_columns]
    return x_data, y_data

def mergeLabel(x_data, y_data):
    """merge attributes and labels

    Args:
        x_data (2d floats): attributes
        y_data (2d floats): labels

    Returns:
        2d floats: dataset
    """
    
    return np.column_stack((x_data, y_data))

def validationSplit(data, percentage): #split the data between train and test
    """train test split using percentage

    Args:
        data (2d floats): dataset
        percentage (floats): proportion of training set

    Returns:
        (2d floats, 2d floats): training set and testing set
    """
    
    portion = round(data.shape[0]*percentage)
    train = data[0:portion]
    test = data[portion:data.shape[0]]
    return train, test

def partitionIndex(instances_size, k): 
    """partition dataset and return its index

    Args:
        instances_size (int): number of instances
        k (int): k folds

    Returns:
        2d int: range of index for each folds
    """
    
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
    """partition the dataset acording to k folds with specified index

    Args:
        data (floats): dataset
        range_indices (2d int): range indices from partition index function
        partition_selection (int): currently selected fold

    Returns:
        (2d floats, 2d floats): test dataset and train dataset
    """
    
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