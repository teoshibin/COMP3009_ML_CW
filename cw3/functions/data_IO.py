import numpy as np
import pandas as pd
import os
import sys

def loadHeartFailureDataset():    #just get the data
    """load csv and split data into x and y

    Returns:
        (2d floats, 2d floats): attributes and labels
    """

    path = os.path.join("datasets","heart_failure_clinical_records_dataset.csv")
    data = np.genfromtxt(path, delimiter=",", names=True)

    #splitting data and label out
    data = data.view(np.float64).reshape((len(data), -1))
    x_data = data[:, 0:-1]
    y_data = data[:, -1]
    y_data = y_data.astype('int32')
    #y_data = np.identity(2)[y_data] # one hot encoding

    return x_data, y_data

def loadConcreteDataset(): 
    """load excel and split data into x and y

    Returns:
        (2d floats, 2d floats): attributes and labels
    """

    path = os.path.join("datasets","Concrete_Data.xls")
    df = pd.read_excel(path)
    data = df.to_numpy()

    #splitting data and label out
    x_data = data[:, 0:-1]
    y_data = data[:, -1]

    return x_data, y_data

def cdSubDir(subdir):
    """change directory into this sub directory if current directory isn't in this subdir

    Args:
        subdir (String): folder name
    """
    
    current_dir = os.getcwd()
    folder_name = current_dir.split(sep=os.path.sep)[-1]
    if not (folder_name == subdir):
        os.chdir(os.path.join(current_dir, subdir))


class Logger(object):
    """Logger class for logging printed output into .log files

    Args:
        object (object): this
    """
    
    def __init__(self, filename="out.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass   