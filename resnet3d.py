import sys
import time


import tensorflow as tf

from utils3d import residual_block, downsample_block, upsample_block
from resnetcore import resnetcore

# Declaring exception names:
class ConfigurationException(Exception): pass
class IncompleteFeedDict(Exception): pass



# Main class
class resnet3d(resnetcore):
    '''Define a network model and run training

    U resnet implementation
    '''
    def __init__(self, params):
        '''initialization

        Requires a list of parameters as python dictionary

        Arguments:
            params {dict} -- Network parameters

        Raises:
            ConfigurationException -- Missing a required parameter
        '''
        required_params =[
            'MINIBATCH_SIZE',
            'SAVE_ITERATION',
            'NUM_LABELS',
            'N_INITIAL_FILTERS',
            'NETWORK_DEPTH',
            'RESIDUAL_BLOCKS_PER_LAYER',
            'LOGDIR',
            'BASE_LEARNING_RATE',
            'TRAINING',
            'RESTORE',
            'ITERATIONS',
        ]

        for param in required_params:
            if param not in params:
                raise ConfigurationException("Missing paragmeter "+ str(param))

        self._params = params
        super(resnet3d, self).__init__(params)

    def _build_network(self, input_placeholder, label_dims):

        x = input_placeholder

        # We break up the intial filters into parallel U ResNets
        # The filters are concatenated at some point, and progress together

        verbose = True

        if verbose:
            print "Initial shape: " + str(x.get_shape())

        x = tf.layers.conv3d(x, self._params['N_INITIAL_FILTERS'],
                                kernel_size=[7, 7, 7],
                                strides=[2, 2, 2],
                                padding='same',
                                use_bias=False,
                                trainable=self._params['TRAINING'],
                                name="Conv2DInitial",
                                reuse=None)

        # ReLU:
        x = tf.nn.relu(x)

        if verbose:
            print "After initial convolution: " + str(x.get_shape())





        # Begin the process of residual blocks and downsampling:
        for i in xrange(self._params['NETWORK_DEPTH']):

            for j in xrange(self._params['RESIDUAL_BLOCKS_PER_LAYER']):
                x = residual_block(x, self._params['TRAINING'],
                                   batch_norm=True,
                                   name="resblock_down{0}_{1}".format(i, j))

                x = downsample_block(x, self._params['TRAINING'],
                                        batch_norm=True,
                                        name="downsample{0}".format(i))
                if verbose:
                    print "Layer {i}: x.get_shape(): {s}".format(
                        i=i, s=x.get_shape())


        final_convolutional_layer = x

        # Here, split into different classifiers for each final classifier:
        logits = dict()
        for label_name in label_dims:

            this_x = residual_block(final_convolutional_layer, self._params['TRAINING'],
                    batch_norm=True, name="resblock_{0}".format(label_name))

            # Apply a bottle neck to get the right shape:
            num_labels = label_dims[label_name][-1]
            this_x = tf.layers.conv3d(this_x,
                             num_labels,
                             kernel_size=[1,1,1],
                             strides=[1,1,1],
                             padding='same',
                             activation=None,
                             use_bias=False,
                             trainable=self._params['TRAINING'],
                             name="BottleneckConv2D_{0}".format(label_name))

            if verbose:
                print "Shape after {0} bottleneck: ".format(label_name) + str(this_x.get_shape())

            # Apply global average pooling to get the right final shape:
            shape = (this_x.shape[1], this_x.shape[2], this_x.shape[3])
            this_x = tf.nn.pool(this_x,
                       window_shape=shape,
                       pooling_type="AVG",
                       padding="VALID",
                       dilation_rate=None,
                       strides=None,
                       name="GlobalAveragePool_{0}".format(label_name),
                       data_format=None)

            if verbose:
                print "Shape after {0} pooling: ".format(label_name) + str(this_x.get_shape())


            # Reshape to the right shape for logits:
            this_x = tf.reshape(this_x, [tf.shape(this_x)[0], num_labels],
                     name="global_pooling_reshape_{0}".format(label_name))

            if verbose:
                print "Final {0} shape: ".format(label_name) + str(this_x.get_shape())


            # Add this label to the logits:
            logits.update({label_name : this_x})


        return logits

