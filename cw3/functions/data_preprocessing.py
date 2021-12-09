import numpy as np


def minMaxNorm(data): #normalize the data
    """normalize data with 5 and 95 percentile, sqeezing them into range of [0,1]

    Args:
        data (2d floats): dataset where rows are instances columns are attributes

    Returns:
        2d floats: normalized data
    """
    spacing = 5
    pct_min = np.percentile(data, spacing, axis=0)
    pct_max = np.percentile(data, 100 - spacing, axis=0)
    norm_data = np.zeros(data.shape)
    for i in range(len(data[0])):
        norm_data[:, i] = (data[:,i] - pct_min[i]) / (pct_max[i] - pct_min[i])

    norm_data[norm_data > 1] = 1
    norm_data[norm_data < 0] = 0

    return norm_data

    
def shuffleRow(data):
    """custom shuffle rows

    Args:
        data (2d floats): dataset

    Returns:
        2d floats: shuffled dataset
    """
    
    np.random.seed(2)
    np.random.shuffle(data)
    return data

def oneHotEncoding(y_data, num_class=2):
    """one hot encoding for multi class

    Args:
        y_data (1d vertical floats): dataset labels 

    Returns:
        2d floats: one hot encoded labels
    """
    
    y_data = np.identity(num_class)[y_data]
    return y_data

def unOneHotEncoding(y_data, axis):
    """decode one hot encoded label

    Args:
        y_data (2d floats): dataset labels
        axis (int): axis to decode from

    Returns:
        2d floats: decoded dataset labels
    """
    
    y_data = np.argmax(y_data, axis)
    return y_data