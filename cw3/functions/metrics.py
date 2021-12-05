import tensorflow as tf
import numpy as np
def f1Score(output, test_y):
    actual =np.argmax(test_y,1).astype(dtype='bool')
    predicted = np.argmax(tf.keras.backend.get_value(output),1).astype(dtype='bool')

    not_predicted = np.logical_not(predicted)
    not_actual = np.logical_not(actual)

    TP = np.count_nonzero(np.logical_and(predicted, actual))
    # TN = np.count_nonzero(np.logical_and(not_predicted, not_actual))
    FP = np.count_nonzero(np.logical_and(predicted, not_actual))
    FN = np.count_nonzero(np.logical_and(not_predicted, actual))
    
    p_denominator = TP + FP
    if p_denominator:
        precision = TP / p_denominator
    else:
        precision = 0
    
    rc_denominator = TP + FN
    if rc_denominator:
        recall = TP / rc_denominator
    else:
        recall = 0
    
    f1_denominator = precision + recall
    if f1_denominator:
        f1 = 2 * precision * recall / f1_denominator
    else:
        f1 = 0
    return f1

def rmseScore(output, test_y):
    rmse = np.sqrt(np.mean(np.square(output - test_y)))

    return rmse

def recall(predicted, actual):
    TP = np.count_nonzero(predicted * actual)
    FN = np.count_nonzero((predicted - 1) * actual)
    return TP / (TP + FN)

def precision(predicted, actual):
    TP = np.count_nonzero(predicted * actual)
    FP = np.count_nonzero(predicted * (actual - 1))
    return TP / (TP + FP)