import tensorflow as tf

#DEFINING WEIGHTS AND BIASES
def one_layer_perceptron(input_x, input_size, output_size, activation_type="none"):

    b = tf.Variable(tf.random_normal([output_size]))
    w = tf.Variable(tf.random_normal([input_size, output_size]))

    logits = tf.add(tf.matmul(input_x, w), b)

    if activation_type == "sigmoid":
        layer = tf.nn.sigmoid(logits)

    elif activation_type == "relu":
        layer = tf.nn.relu(logits)

    elif activation_type == "softmax":
        layer = tf.nn.softmax(logits)
    
    elif activation_type == "none":
        layer = logits

    return layer, w