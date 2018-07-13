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
    def __init__(self):
        '''initialization

        Requires a list of parameters as python dictionary

        Arguments:
            params {dict} -- Network parameters

        Raises:
            ConfigurationException -- Missing a required parameter
        '''

        # Call the base class to initialize _core_network_params:
        super(resnet, self).__init__()

        # Extend the parameters to include the needed ones:

        self._core_network_params += [
            'N_INITIAL_FILTERS',
            'RESIDUAL_BLOCKS_PER_LAYER',
            'NETWORK_DEPTH_PRE_MERGE',
            'NETWORK_DEPTH_POST_MERGE',
            'SHARE_WEIGHTS',
            'NPLANES',
        ]


        return




    def _build_network(self, inputs, verbosity=0):

        ''' verbosity 0 = no printouts
            verbosity 1 = sparse information
            verbosity 2 = debug
        '''

        # Spin off the input image(s):

        x = inputs['image']

        if verbosity > 1:
            print "Initial input shape: " + str(x.get_shape())


        # We break up the intial filters into parallel U ResNets
        # The filters are concatenated at some point, and progress together


        if verbosity > 1:
            print "Initial shape: " + str(x.get_shape())

        n_planes = self._params['NPLANES']

        if verbosity > 1:
            print "Attempting to split into {} planes".format(n_planes)

        x = tf.split(x, n_planes*[1], -1)

        if verbosity > 1:
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

        if verbosity > 1:
            print "After initial convolution: "

            for p in range(len(x)):
                print "Plane {0}: ".format(p) + str(x[p].get_shape())



        if verbosity > 0:
            print "Begining downsampling"

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
                if verbosity > 1:
                    print "Plane {p}, layer {i}: x[{p}].get_shape(): {s}".format(
                        p=p, i=i, s=x[p].get_shape())

        # print "Reached the deepest layer."

        if verbosity > 0:
            print "Concatenating planes together"

        # Here, concatenate all the planes together before the residual block:
        x = tf.concat(x, axis=-1)



        if verbosity > 1:
            print "Shape after concatenation: " + str(x.get_shape())

        # At the bottom, do another residual block:
        for i in xrange(self._params['NETWORK_DEPTH_POST_MERGE']):
            for j in xrange(self._params['RESIDUAL_BLOCKS_PER_LAYER']):
                x = residual_block(x, self._params['TRAINING'],
                    batch_norm=True, name="resblock_postmerge_{0}_{1}".format(i, j))

            x = downsample_block(x, self._params['TRAINING'],
                                 batch_norm=True,
                                 name="downsample_postmerge{0}".format(i))

        if verbosity > 1:
            print "Shape after final block: " + str(x.get_shape())


        final_convolutional_layer = x

        # At this point, we split into several bottlenecks to produce
        # The final output logits for each label.
        # Each splits off from the final convolutional layer with a
        # residual block and a bottle neck

        # if there is only one output class, just do one more residual block and return.
        # infer the number of labels from the inputs tensors

        if isinstance(inputs['label'], dict):
            logits = dict()
            for label_name in inputs['label'].keys():
                this_x = residual_block(final_convolutional_layer, self._params['TRAINING'],
                    batch_norm=True, name="resblock_{0}".format(label_name))

                num_labels = inputs['label'][label_name].get_shape().as_list()[-1]
                # Bottle neck layer converts to proper output size:
                this_x = tf.layers.conv2d(this_x,
                                          num_labels,
                                          kernel_size=[1,1],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=None,
                                          use_bias=False,
                                          trainable=self._params['TRAINING'],
                                          name="BottleneckConv2D_{0}".format(label_name))

                if verbosity > 1:
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

                if verbosity > 1:
                    print "Shape after {0} pooling: ".format(label_name) + str(this_x.get_shape())


                # Reshape to the right shape for logits:
                this_x = tf.reshape(this_x, [tf.shape(this_x)[0], num_labels],
                         name="global_pooling_reshape_{0}".format(label_name))

                if verbosity > 1:
                    print "Final {0} shape: ".format(label_name) + str(this_x.get_shape())


                # Add this label to the logits:
                logits.update({label_name : this_x})

        else:
            this_x = residual_block(final_convolutional_layer, self._params['TRAINING'],
                                    batch_norm=True, name="resblock_final")

            num_labels = inputs['label'].get_shape().as_list()[-1]
            # Bottle neck layer converts to proper output size:
            this_x = tf.layers.conv2d(this_x,
                                      num_labels,
                                      kernel_size=[1,1],
                                      strides=[1, 1],
                                      padding='same',
                                      activation=None,
                                      use_bias=False,
                                      trainable=self._params['TRAINING'],
                                      name="BottleneckConv2D")

            if verbosity > 1:
                print "Shape after bottleneck: " + str(this_x.get_shape())

            # Apply global average pooling to get the right final shape:
            shape = (this_x.shape[1], this_x.shape[2])
            this_x = tf.nn.pool(this_x,
                       window_shape=shape,
                       pooling_type="AVG",
                       padding="VALID",
                       dilation_rate=None,
                       strides=None,
                       name="GlobalAveragePool",
                       data_format=None)

            if verbosity > 1:
                print "Shape after pooling : " + str(this_x.get_shape())


            # Reshape to the right shape for logits:
            this_x = tf.reshape(this_x, [tf.shape(this_x)[0], num_labels],
                     name="global_pooling_reshape_{0}".format(label_name))

            if verbosity > 1:
                print "Final shape: " + str(this_x.get_shape())

            logits = this_x

        return logits
