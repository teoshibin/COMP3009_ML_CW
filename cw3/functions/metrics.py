import tensorflow as tf

def f1Score(predicted, actual):
    TP = tf.count_nonzero(predicted * actual)
    TN = tf.count_nonzero((predicted - 1) * (actual - 1))
    FP = tf.count_nonzero(predicted * (actual - 1))
    FN = tf.count_nonzero((predicted - 1) * actual)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return f1