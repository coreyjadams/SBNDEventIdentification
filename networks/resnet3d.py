import sys
import time


import tensorflow as tf

from utils3d import residual_block, downsample_block, convolutional_block

from resnetcore import resnetcore


# Main class
class resnet3d(resnetcore):
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
        super(resnet3d, self).__init__()

        self._core_network_params += [
            'N_INITIAL_FILTERS',
            'RESIDUAL_BLOCKS_PER_LAYER',
            'NETWORK_DEPTH',
        ]




    def _build_network(self, inputs, verbosity=2):

        x = inputs['image']

        # We break up the intial filters into parallel U ResNets
        # The filters are concatenated at some point, and progress together


        if verbosity > 1:
            print "Initial shape: " + str(x.get_shape())

        x = convolutional_block(x,
                        is_training=self._params['TRAINING'],
                        name="initial_convolution",
                        batch_norm=True,
                        dropout=False,
                        kernel_size=[5,5,5],
                        strides=[4,4,4],
                        n_filters=self._params['N_INITIAL_FILTERS'],
                        reuse=False)

        if verbosity > 1:
            print "After initial convolution: " + str(x.get_shape())


        # x = convolutional_block(x,
        #                 is_training=self._params['TRAINING'],
        #                 name="initial_convolution2",
        #                 batch_norm=True,
        #                 dropout=False,
        #                 kernel_size=[3,3,3],
        #                 strides=[2,2,2],
        #                 n_filters=self._params['N_INITIAL_FILTERS'],
        #                 reuse=False)

        if verbosity > 1:
            print "After initial convolution: " + str(x.get_shape())

        if verbosity > 0:
            print "Begin downsampling {} times".format(self._params['NETWORK_DEPTH'])

        # Begin the process of residual blocks and downsampling:
        for i in xrange(self._params['NETWORK_DEPTH']):

            for j in xrange(self._params['RESIDUAL_BLOCKS_PER_LAYER']):
                x = residual_block(x, self._params['TRAINING'],
                                   batch_norm=True,
                                   name="resblock_down{0}_{1}".format(i, j))

            x = downsample_block(x, self._params['TRAINING'],
                                    batch_norm=True,
                                    name="downsample{0}".format(i))
            if verbosity > 1:
                print "Layer {i}: x.get_shape(): {s}".format(
                    i=i, s=x.get_shape())


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

                # Apply a bottle neck to get the right shape:
                num_labels = inputs['label'][label_name].get_shape().as_list()[-1]
                this_x = tf.layers.conv3d(this_x,
                                          num_labels,
                                          kernel_size=[1,1,1],
                                          strides=[1,1,1],
                                          padding='same',
                                          activation=None,
                                          use_bias=False,
                                          trainable=self._params['TRAINING'],
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                          name="BottleneckConv3D_{0}".format(label_name))

                if verbosity > 1:
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
                batch_norm=True, name="resblock")

            # Apply a bottle neck to get the right shape:
            num_labels = inputs['label'].get_shape().as_list()[-1]
            this_x = tf.layers.conv3d(this_x,
                                      num_labels,
                                      kernel_size=[1,1,1],
                                      strides=[1,1,1],
                                      padding='same',
                                      activation=None,
                                      use_bias=False,
                                      trainable=self._params['TRAINING'],
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
                                      name="BottleneckConv3D")

            if verbosity > 1:
                print "Shape after bottleneck: " + str(this_x.get_shape())


                # Apply global average pooling to get the right final shape:
                shape = (this_x.shape[1], this_x.shape[2], this_x.shape[3])
                this_x = tf.nn.pool(this_x,
                           window_shape=shape,
                           pooling_type="AVG",
                           padding="VALID",
                           dilation_rate=None,
                           strides=None,
                           name="GlobalAveragePool",
                           data_format=None)

                if verbosity > 1:
                    print "Shape after pooling: " + str(this_x.get_shape())


                # Reshape to the right shape for logits:
                this_x = tf.reshape(this_x, [tf.shape(this_x)[0], num_labels],
                         name="global_pooling_reshape")

                if verbosity > 1:
                    print "Final shape: " + str(this_x.get_shape())


                # Add this label to the logits:
                logits.update({label_name : this_x})


        return logits













