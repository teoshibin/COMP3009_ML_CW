import numpy as np
import os

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