import tensorflow as tf

#DEFINING WEIGHTS AND BIASES
def one_layer_perceptron(input_x, input_size, output_size, activation_type="none"):
    """define weights, biases and activation function one single layer

    Args:
        input_x (tensor): previous ANN layer with the output size of current input size
        input_size (int): number of input nodes
        output_size (int): number of output nodes
        activation_type (str, optional): name of activation function. Defaults to "none".

    Returns:
        tensor: a layer of ANN that wrap apon previous layers with the size of input_size and output_size
    """

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

def multi_layer_perceptron(structure):
    """contruct multi layers perceptron with specified structure
        
        e.g.
        structure = np.array([
        (8, ""),
        (48, "relu"),
        (32, "relu"),
        (16, "relu"),
        (32, "relu"),
        (8, "relu"),
        (1, "none"),
        ])

    Args:
        structure ([(int, string)]): array of tuples specifying number of nodes and activation function

    Returns:
        (tensor, tensor, placeholder, placeholder): complete MLP ANN, regularizer, X training input, Y actual training ouput
    """
        
    #Defining the input and the output
    X = tf.placeholder("float", [None, structure[0][0]])
    Y = tf.placeholder("float", [None, structure[-1][0]])

    # Loop Generate Network
    output_layer, w = one_layer_perceptron(X, int(structure[0][0]), int(structure[1][0]), structure[1][1])
    regularizer = tf.nn.l2_loss(w)
    for i in range(1, len(structure) - 1):
        output_layer, w = one_layer_perceptron(
            output_layer, int(structure[i][0]), int(structure[i + 1][0]), structure[i + 1][1])    
        regularizer = regularizer + tf.nn.l2_loss(w)
        
    return output_layer, regularizer, X, Y