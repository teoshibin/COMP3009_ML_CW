import numpy as np

def maxCountOccur(data):
    new_list = [i for x in data for i in x]
    max_val = max(new_list, key=lambda x:new_list.count(x)) 
    return max_val

def findMean(data):
    meanValue = np.mean(data)
    return meanValue