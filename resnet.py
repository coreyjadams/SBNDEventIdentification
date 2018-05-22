import sys
import time


import tensorflow as tf

from utils import residual_block, downsample_block, upsample_block

from resnetcore import resnetcore

# Declaring exception names:
class ConfigurationException(Exception): pass
class IncompleteFeedDict(Exception): pass



# Main class
class resnet(resnetcore):
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
        super(resnet, self).__init__(params)

    def _build_network(self, input_placeholder, label_dims):

        x = input_placeholder

        # We break up the intial filters into parallel U ResNets
        # The filters are concatenated at some point, and progress together

        verbose = False

        if verbose:
            print "Initial shape: " + str(x.get_shape())
        n_planes = self._params['NPLANES']

        x = tf.split(x, n_planes*[1], -1)
        if verbose:
            for p in range(len(x)):
                print "Plane {0} initial shape:".format(p) + str(x[p].get_shape())

        # Initial convolution to get to the correct number of filters:
        for p in range(len(x)):
            x[p] = tf.layers.conv2d(x[p], self._params['N_INITIAL_FILTERS'],
                                    kernel_size=[7, 7],
                                    strides=[2, 2],
                                    padding='same',
                                    use_bias=False,
                                    trainable=self._params['TRAINING'],
                                    name="Conv2DInitial_plane{0}".format(p),
                                    reuse=None)

            # ReLU:
            x[p] = tf.nn.relu(x[p])

        if verbose:
            print "After initial convolution: "

            for p in range(len(x)):
                print "Plane {0}".format(p) + str(x[p].get_shape())





        # Begin the process of residual blocks and downsampling:
        for p in xrange(len(x)):
            for i in xrange(self._params['NETWORK_DEPTH_PRE_MERGE']):

                for j in xrange(self._params['RESIDUAL_BLOCKS_PER_LAYER']):
                    x[p] = residual_block(x[p], self._params['TRAINING'],
                                          batch_norm=True,
                                          name="resblock_down_plane{0}_{1}_{2}".format(p, i, j))

                x[p] = downsample_block(x[p], self._params['TRAINING'],
                                        batch_norm=True,
                                        name="downsample_plane{0}_{1}".format(p,i))
                if verbose:
                    print "Plane {p}, layer {i}: x[{p}].get_shape(): {s}".format(
                        p=p, i=i, s=x[p].get_shape())

        # print "Reached the deepest layer."

        # Here, concatenate all the planes together before the residual block:
        x = tf.concat(x, axis=-1)

        if verbose:
            print "Shape after concatenation: " + str(x.get_shape())

        # At the bottom, do another residual block:
        for i in xrange(self._params['NETWORK_DEPTH_POST_MERGE']):
            for j in xrange(self._params['RESIDUAL_BLOCKS_PER_LAYER']):
                x = residual_block(x, self._params['TRAINING'],
                    batch_norm=True, name="resblock_postmerge_{0}_{1}".format(i, j))

            x = downsample_block(x, self._params['TRAINING'],
                                 batch_norm=True,
                                 name="downsample_postmerge{0}".format(i))

        if verbose:
            print "Shape after final block: " + str(x.get_shape())


        final_convolutional_layer = x

        # At this point, we split into several bottlenecks to produce
        # The final output logits for each label.
        # Each splits off from the final convolutional layer with a
        # residual block and a bottle neck
        logits = dict()
        for label_name in label_dims:

            this_x = residual_block(final_convolutional_layer, self._params['TRAINING'],
                    batch_norm=True, name="resblock_{0}".format(label_name))

            # Apply a bottle neck to get the right shape:
            num_labels = label_dims[label_name][-1]
            this_x = tf.layers.conv2d(this_x,
                             num_labels,
                             kernel_size=[1,1],
                             strides=[1, 1],
                             padding='same',
                             activation=None,
                             use_bias=False,
                             trainable=self._params['TRAINING'],
                             name="BottleneckConv2D_{0}".format(label_name))

            if verbose:
                print "Shape after {0} bottleneck: ".format(label_name) + str(this_x.get_shape())

            # Apply global average pooling to get the right final shape:
            shape = (this_x.shape[1], this_x.shape[2])
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
