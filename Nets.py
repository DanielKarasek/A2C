import tensorflow as tf
import tensorflow.contrib.layers as layers

registered = {}


def register(name):
    def inner(func):
        if name not in registered:
            registered[name] = func
        return func
    return inner


@register("fully_connected_net")
def dnn_build(hiddens=(256,),
              batch_norm=True,
              reuse=tf.AUTO_REUSE,
              **fully_kwargs):
    """
    Function creating function to build DNN with givent architecture and
    relu activations in hidden layers + linear activation for output layer.
    args:
        net: integer - inputs to neural network
        batch_norm: Boolean - Whether to normalize batch between each layer
        hiddens: Iterable object of scalar numbers where each number represents
               number of units in given layer e.g. 2 hidden layers with 10 and 15 neurons == [10,15])

    returns: Handle to output layer
    """
    def build_net(X):
        with tf.variable_scope("fully_connected", reuse=reuse):
            net = X
            for idx, hidden in enumerate(hiddens):
                hidden = int(hidden)
                net = layers.fully_connected(inputs=net,
                                             num_outputs=hidden,
                                             activation_fn=None,
                                             **fully_kwargs
                                             )
                if batch_norm:
                    net = layers.layer_norm(net, trainable=True, center=True, scale=True)

                net = tf.nn.relu(net)
            return net
    return build_net


@register("conv_net")
def build_cnn(layers_params=((16, 8, 4), (32, 4, 2)), scale=False, reuse=tf.AUTO_REUSE, **conv_kwargs):
    """
    Creates build function for CNN, with given architecture

    layers_params:
        Iterable of n triplets, where n is number of layers.
        Each triplet contains of (number of filters, filter size, stride)
    conv_kwargs:
        Kwargs for convolution layers
    """
    def build_net(X):
        if scale:
            net = (tf.cast(X, tf.float32)-127.5)/127.5
        else:
            net = X
        with tf.variable_scope("convnet", reuse=reuse):
            for filt, kernel_size, stride in layers_params:
                net = layers.conv2d(inputs=net,
                                    num_outputs=filt,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    activation_fn=tf.nn.relu,
                                    **conv_kwargs
                                    )
        net = layers.flatten(net)
        return net

    return build_net


@register("conv2fully")
def build_conv2fully(batch_norm=False, scale=True, reuse=tf.AUTO_REUSE, **kwargs):
    """
        Creates function to build CNN followed by DNN with given architecture
        batch_norm: Boolean - Whether to use batch normalization in DNN
        Kwargs: Kwargs for both CNN and DNN
    """
    def build_net(X):
        with tf.variable_scope("conv2fully", reuse=reuse):
            net = get_network_builder("conv_net")(scale=scale, **kwargs)(X)
            net = get_network_builder("fully_connected_net")(batch_norm=batch_norm,
                                                             **kwargs)(net)
        return net
    return build_net


def get_network_builder(name):
    """
        Finds function to create type of given NN if exists
        args:
            name: String - name of NN type
        returns: Function to create given NN type
    """
    if name in registered:
        return registered[name]
    else:
        raise ValueError("unknown network")


@register("deepQ")
def build_Q_func(net_name="conv2fully", dueling=True, **net_kwargs):
    """
    Creates Deep Q network builder with given architecture
    net_name: String - which type of NN to use
    dueling: Boolean - Whether to use dueling architecture
    net_kwargs: Kwargs for NN
    returns: Deep Q network builder
    """
    def build_net(inputs, action_size, scope="deepQ", reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            net = get_network_builder(net_name)(**net_kwargs)(inputs)

            with tf.variable_scope("Q_values_calculation"):
                if dueling:
                    adv = layers.fully_connected(inputs=net,
                                                 num_outputs=action_size,
                                                 activation_fn=None,
                                                 )
                    val = layers.fully_connected(inputs=net,
                                                 num_outputs=1,
                                                 activation_fn=None
                                                 )
                    q_values = adv + val

                else:
                    q_values = layers.fully_connected(inputs=net,
                                                      num_outputs=action_size,
                                                      activation_fn=None,
                                                      )
            return q_values

    return build_net
