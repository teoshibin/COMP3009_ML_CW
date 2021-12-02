import tensorflow as tf

#DEFINING WEIGHTS AND BIASES
def one_layer_perceptron(input_x, input_size, output_size, activation_type):

    b = tf.Variable(tf.random_normal([output_size]))
    w = tf.Variable(tf.random_normal([input_size, output_size]))

    if activation_type == "sigmoid":
        layer = tf.nn.sigmoid(tf.add(tf.matmul(input_x, w), b))

    elif activation_type == "relu":
        layer = tf.nn.relu(tf.add(tf.matmul(input_x, w), b))

    elif activation_type == "softmax":
        layer = tf.nn.softmax(tf.add(tf.matmul(input_x, w), b))

    return layer