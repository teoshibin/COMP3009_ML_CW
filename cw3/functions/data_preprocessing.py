import numpy as np


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

    
def shuffleRow(data):
    np.random.seed(2)
    np.random.shuffle(data)
    return data

def oneHotEncoding(y_data):
    y_data = np.identity(2)[y_data]
    return y_data

def unOneHotEncoding(y_data, axis):
    y_data = np.argmax(y_data, axis)
    return y_data