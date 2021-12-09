import numpy as np

def maxCountOccur(data):
    """return value that occured the most in this input

    Args:
        data (1d floats): input array

    Returns:
        floats: value that occured the most
    """
    
    new_list = [i for x in data for i in x]
    max_val = max(new_list, key=lambda x:new_list.count(x)) 
    return max_val

def findMean(data): 
    """calculate mean

    Args:
        data (2d floats): input array

    Returns:
        float: mean
    """
       
    meanValue = np.mean(data)
    return meanValue