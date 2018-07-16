import sys
import time


import tensorflow as tf

from utils import residual_block, downsample_block, upsample_block


from uresnetcore import uresnetcore

class uresnet(uresnetcore):

    '''
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
        super(uresnet, self).__init__()

        # Extend the parameters to include the needed ones:

        self._core_network_params += [
            'N_INITIAL_FILTERS',
            'RESIDUAL_BLOCKS_PER_LAYER',
            'RESIDUAL_BLOCKS_DEEPEST_LAYER',
            'NETWORK_DEPTH',
            'SHARE_WEIGHTS',
            'NPLANES',
            'NUM_LABELS',
            'BATCH_NORM',
        ]


        return



    def _create_softmax(self, logits):
        '''Must return a dict type

        [description]

        Arguments:
            logits {[type]} -- [description]

        Raises:
            NotImplementedError -- [description]
        '''

        # For the logits, we compute the softmax and the predicted label

        # Since this is the 2D network, the logits is a list of len == NPLANES.
        # Compute the softmax and prediction over each pixel on each plane,
        # but maintain the list structure.

        output = dict()
        output['softmax'] = []
        output['prediction'] = []

        for p in range(self._params['NPLANES']):
            this_logits = logits[p]
            output['softmax'].append(tf.nn.softmax(this_logits, axis=-1))
            output['prediction'].append(tf.argmax(this_logits, axis=-1))

        print output

        return output





    def _calculate_loss(self, inputs, outputs):
        ''' Calculate the loss.

        returns a single scalar for the optimizer to use.
        '''

        # To calculate the loss, we have to split the inputs
        # again since the logits for each plane come split.


        with tf.name_scope('cross_entropy'):
            n_planes = self._params['NPLANES']
            labels = tf.split(inputs['label'], n_planes*[1], -1)

            self._loss_by_plane = [ [] for i in range(self._params['NPLANES']) ]

            for p in xrange(n_planes):

                # Unreduced loss, shape [BATCH, L, W]
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels[p],
                                            logits=outputs[p])

                print losses
                print inputs['weight'][p]

                if self._params['BALANCE_LOSS']:
                    losses = tf.multiply(losses, weights[p])

                self._loss_by_plane[p] = tf.reduce_sum(tf.reduce_sum(losses))

                # Add the loss to the summary:
                tf.summary.scalar("Total_Loss_plane{0}".format(p), self._loss_by_plane[p])


            self._loss = tf.reduce_sum(self._loss_by_plane)

            tf.summary.scalar("Total_Loss", self._loss)



        print 'here'


        # with tf.name_scope('cross_entropy'):

        #     else:
        #         #otherwise, just one set of logits, against one label:
        #         loss = tf.reduce_mean(
        #             tf.nn.softmax_cross_entropy_with_logits(labels=inputs['label'],
        #                                                     logits=outputs))



        #     # If desired, add weight regularization loss:
        #     if 'REGULARIZE_WEIGHTS' in self._params:
        #         reg_loss = tf.losses.get_regularization_loss()
        #         loss += reg_loss


        #     # Total summary:
        #     tf.summary.scalar("Total Loss",loss)

        #     return loss



    # def _calculate_accuracy(self, inputs, outputs):
    #     ''' Calculate the accuracy.

    #     '''

    #     # Compare how often the input label and the output prediction agree:

    #     with tf.name_scope('accuracy'):

    #         if isinstance(outputs['prediction'], dict):
    #             accuracy = dict()

    #             for key in outputs['prediction'].keys():
    #                 correct_prediction = tf.equal(tf.argmax(inputs['label'][key], -1),
    #                                               outputs['prediction'][key])
    #                 accuracy.update({
    #                     key : tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #                 })


    #                 # Add the accuracies to the summary:
    #                 tf.summary.scalar("{0}_Accuracy".format(key), accuracy[key])

    #         else:
    #             correct_prediction = tf.equal(tf.argmax(inputs['label'], -1),
    #                                           outputs['prediction'])
    #             accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #             tf.summary.scalar("Accuracy", accuracy)

    #     return accuracy

    def _build_network(self, inputs, verbosity = 0):


        if verbosity > 1:
            print inputs

        x = inputs['image']

        # We break up the intial filters into parallel U ResNets
        # The filters are concatenated at the deepest level
        # And then they are split again into the parallel chains


        # print x.get_shape()
        n_planes = self._params['NPLANES']

        if self._params['SHARE_WEIGHTS']:
            sharing = True
        else:
            sharing = False

        x = tf.split(x, n_planes*[1], -1)

        if verbosity > 1:
            for p in range(len(x)):
                print "Plane {0} shape: ".format(p) + str(x[p].get_shape())

        # Initial convolution to get to the correct number of filters:
        for p in range(len(x)):
            name = "Conv2DInitial"
            reuse = False
            if not sharing:
                name += "_plane{0}".format(p)
            if sharing and p != 0:
                reuse = True

            if verbosity > 1:
                print "Name: {0} + reuse: {1}".format(name, reuse)

            x[p] = tf.layers.conv2d(x[p], self._params['N_INITIAL_FILTERS'],
                                    kernel_size=[5, 5],
                                    strides=[1, 1],
                                    padding='same',
                                    use_bias=False,
                                    trainable=self._params['TRAINING'],
                                    name=name,
                                    reuse=reuse)

            # ReLU:
            x[p] = tf.nn.relu(x[p])

        if verbosity > 1:
            for p in range(len(x)):
                print x[p].get_shape()



        # Need to keep track of the outputs of the residual blocks before downsampling, to feed
        # On the upsampling side

        network_filters = [[] for p in range(len(x))]

        # Begin the process of residual blocks and downsampling:
        for p in xrange(len(x)):
            for i in xrange(self._params['NETWORK_DEPTH']):

                for j in xrange(self._params['RESIDUAL_BLOCKS_PER_LAYER']):
                    name = "resblock_down"
                    reuse = False
                    if not sharing:
                        name += "_plane{0}".format(p)
                    if sharing and p != 0:
                        reuse = True

                    name += "_{0}_{1}".format(i, j)

                    if verbosity > 1:
                        print "Name: {0} + reuse: {1}".format(name, reuse)


                    x[p] = residual_block(x[p], self._params['TRAINING'],
                                          batch_norm=self._params['BATCH_NORM'],
                                          name=name,
                                          reuse=reuse)

                name = "downsample"
                reuse = False
                if not sharing:
                    name += "_plane{0}".format(p)
                if sharing and p != 0:
                    reuse = True

                name += "_{0}".format(i)

                if verbosity > 1:
                    print "Name: {0} + reuse: {1}".format(name, reuse)

                network_filters[p].append(x[p])
                x[p] = downsample_block(x[p], self._params['TRAINING'],
                                        batch_norm=self._params['BATCH_NORM'],
                                        name=name,
                                        reuse=reuse)

                if verbosity > 1:
                    print "Plane {p}, layer {i}: x[{p}].get_shape(): {s}".format(
                        p=p, i=i, s=x[p].get_shape())

        if verbosity > 0:
            print "Reached the deepest layer."

        # Here, concatenate all the planes together before the residual block:
        x = tf.concat(x, axis=-1)
        if verbosity > 0:
            print "Shape after concat: " + str(x.get_shape())

        # At the bottom, do another residual block:
        for j in xrange(self._params['RESIDUAL_BLOCKS_DEEPEST_LAYER']):
            x = residual_block(x, self._params['TRAINING'],
                batch_norm=self._params['BATCH_NORM'], name="deepest_block_{0}".format(j))

        # print "Shape after deepest block: " + str(x.get_shape())

        # Need to split the network back into n_planes
        # The deepest block doesn't change the shape, so
        # it's easy to split:
        x = tf.split(x, n_planes, -1)

        # for p in range(len(x)):
        #     print x[p].get_shape()

        # print "Upsampling now."


        # Come back up the network:
        for p in xrange(len(x)):
            for i in xrange(self._params['NETWORK_DEPTH']-1, -1, -1):

                # print "Up start, Plane {p}, layer {i}: x[{p}].get_shape(): {s}".format(
                #     p=p, i=i, s=x[p].get_shape())

                # How many filters to return from upsampling?
                n_filters = network_filters[p][-1].get_shape().as_list()[-1]


                name = "upsample"
                reuse = False
                if not sharing:
                    name += "_plane{0}".format(p)
                if sharing and p != 0:
                    reuse = True

                name += "_{0}".format(i)
                if verbosity > 1:
                    print "Name: {0} + reuse: {1}".format(name, reuse)

                # Upsample:
                x[p] = upsample_block(x[p],
                                      self._params['TRAINING'],
                                      batch_norm=self._params['BATCH_NORM'],
                                      n_output_filters=n_filters,
                                      name=name,
                                      reuse=reuse)


                x[p] = tf.concat([x[p], network_filters[p][-1]],
                                  axis=-1, name='up_concat_plane{0}_{1}'.format(p,i))

                # Remove the recently concated filters:
                network_filters[p].pop()
                # with tf.variable_scope("bottleneck_plane{0}_{1}".format(p,i)):

                name = "BottleneckUpsample"
                reuse = False
                if not sharing:
                    name += "_plane{0}".format(p)
                if sharing and p != 0:
                    reuse = True

                name += "_{0}".format(i)

                if verbosity > 1:
                    print "Name: {0} + reuse: {1}".format(name, reuse)


                # Include a bottleneck to reduce the number of filters after upsampling:
                x[p] = tf.layers.conv2d(x[p],
                                        n_filters,
                                        kernel_size=[1,1],
                                        strides=[1,1],
                                        padding='same',
                                        activation=None,
                                        use_bias=False,
                                        reuse=reuse,
                                        trainable=self._params['TRAINING'],
                                        name=name)

                x[p] = tf.nn.relu(x[p])

                # Residual
                for j in xrange(self._params['RESIDUAL_BLOCKS_PER_LAYER']):
                    name = "resblock_up"
                    reuse = False
                    if not sharing:
                        name += "_plane{0}".format(p)
                    if sharing and p != 0:
                        reuse = True

                    name += "_{0}_{1}".format(i, j)

                    if verbosity > 1:
                        print "Name: {0} + reuse: {1}".format(name, reuse)


                    x[p] = residual_block(x[p], self._params['TRAINING'],
                                          batch_norm=self._params['BATCH_NORM'],
                                          reuse=reuse,
                                          name=name)

                # print "Up end, Plane {p}, layer {i}: x[{p}].get_shape(): {s}".format(
                #     p=p, i=i, s=x[p].get_shape())

        # Split here for segmentation labeling and vertex finding.

        presplit_filters = [ layer for layer in x ]

        for p in xrange(len(x)):
            name = "FinalResidualBlock"
            reuse = False
            if not sharing:
                name += "_plane{0}".format(p)
            if sharing and p != 0:
                reuse = True

            if verbosity > 1:
                print "Name: {0} + reuse: {1}".format(name, reuse)


            x[p] = residual_block(x[p],
                    self._params['TRAINING'],
                    batch_norm=self._params['BATCH_NORM'],
                    reuse=reuse,
                    name=name)

            name = "BottleneckConv2D"
            reuse = False
            if not sharing:
                name += "_plane{0}".format(p)
            if sharing and p != 0:
                reuse = True

            if verbosity > 1:
                print "Name: {0} + reuse: {1}".format(name, reuse)


            # At this point, we ought to have a network that has the same shape as the initial input, but with more filters.
            # We can use a bottleneck to map it onto the right dimensions:
            x[p] = tf.layers.conv2d(x[p],
                                 self._params['NUM_LABELS'],
                                 kernel_size=[7,7],
                                 strides=[1, 1],
                                 padding='same',
                                 activation=None,
                                 use_bias=False,
                                 trainable=self._params['TRAINING'],
                                 reuse=reuse,
                                 name=name)

        seg_logits = x

        if verbosity > 0:
            for p in range(self._params['NPLANES']):
                print "Final output shape, plane {}: ".format(p) + str(seg_logits[p].get_shape())

            # print x[p].get_shape()
        # The final activation is softmax across the pixels.  It gets applied in the loss function
#         x = tf.nn.softmax(x)


        return seg_logits