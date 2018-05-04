import sys
import time


import tensorflow as tf

from utils import residual_block, downsample_block, upsample_block

# Declaring exception names:
class ConfigurationException(Exception): pass
class IncompleteFeedDict(Exception): pass



# Main class
class resnet(object):
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
            'NPLANES',
            'N_INITIAL_FILTERS',
            'NETWORK_DEPTH_PRE_MERGE',
            'NETWORK_DEPTH_POST_MERGE',
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

    def construct_network(self, dims):
        '''Build the network model

        Initializes the tensorflow model according to the parameters
        '''

        tf.reset_default_graph()


        start = time.time()
        # Initialize the input layers:
        self._input_image  = tf.placeholder(tf.float32, dims, name="input_image")
        self._input_labels = tf.placeholder(tf.int64,
                                            [dims[0], self._params['NUM_LABELS']],
                                            name="input_labels")


        sys.stdout.write(" - Finished input placeholders [{0:.2}s]\n".format(time.time() - start))
        start = time.time()
        logits = self._build_network(self._input_image)

        sys.stdout.write(" - Finished Network graph [{0:.2}s]\n".format(time.time() - start))
        start = time.time()
        # for p in xrange(len(logits_by_plane)):
        #     print "logits_by_plane[{0}].get_shape(): ".format(p) + str(logits_by_plane[p].get_shape())
        self._softmax = tf.nn.softmax(logits)
        self._predicted_labels = tf.argmax(logits, axis=-1)

        print self._input_labels.get_shape()
        print logits.get_shape()

        # for p in xrange(len(self._softmax)):
        #     print "self._softmax[{0}].get_shape(): ".format(p) + str(self._softmax[p].get_shape())
        # for p in xrange(len(self._predicted_labels)):
        #     print "self._predicted_labels[{0}].get_shape(): ".format(p) + str(self._predicted_labels[p].get_shape())


        # Keep a list of trainable variables for minibatching:
        with tf.variable_scope('gradient_accumulation'):
            self._accum_vars = [tf.Variable(tv.initialized_value(),
                                trainable=False) for tv in tf.trainable_variables()]

        sys.stdout.write(" - Finished gradient accumulation [{0:.2}s]\n".format(time.time() - start))
        start = time.time()



        # Accuracy calculations:
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self._input_labels, -1),
                                          tf.argmax(self._predicted_labels, -1))
            self._total_accuracy   = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


            # Add the accuracies to the summary:
            tf.summary.scalar("Total_Accuracy", self._total_accuracy)

        sys.stdout.write(" - Finished accuracy [{0:.2}s]\n".format(time.time() - start))
        start = time.time()

        # Loss calculations:
        with tf.name_scope('cross_entropy'):
            self._loss = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(labels=self._input_labels,
                                                        logits=logits))

            tf.summary.scalar("Total_Loss", self._loss)

        sys.stdout.write(" - Finished cross entropy [{0:.2}s]\n".format(time.time() - start))
        start = time.time()

        # Optimizer:
        if self._params['TRAINING']:
            with tf.name_scope("training"):
                self._global_step = tf.Variable(0, dtype=tf.int32,
                    trainable=False, name='global_step')
                if self._params['BASE_LEARNING_RATE'] <= 0:
                    opt = tf.train.AdamOptimizer()
                else:
                    opt = tf.train.AdamOptimizer(self._params['BASE_LEARNING_RATE'])

                # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # with tf.control_dependencies(update_ops):

                # Variables for minibatching:
                self._zero_gradients =  [tv.assign(tf.zeros_like(tv)) for tv in self._accum_vars]
                self._accum_gradients = [self._accum_vars[i].assign_add(gv[0]) for
                                         i, gv in enumerate(opt.compute_gradients(self._loss))]
                self._apply_gradients = opt.apply_gradients(zip(self._accum_vars, tf.trainable_variables()),
                    global_step = self._global_step)

        sys.stdout.write(" - Finished optimizer [{0:.2}s]\n".format(time.time() - start))
        start = time.time()


        # Merge the summaries:
        self._merged_summary = tf.summary.merge_all()
        sys.stdout.write(" - Finished snapshotting [{0:.2}s]\n".format(time.time() - start))


    def apply_gradients(self,sess):

        return sess.run( [self._apply_gradients], feed_dict = {})


    def feed_dict(self, images, labels):
        '''Build the feed dict

        Take input images, labels and match
        to the correct feed dict tensorrs

        Arguments:
            images {numpy.ndarray} -- Image array, [BATCH, L, W, F]
            labels {numpy.ndarray} -- Label array, [BATCH, L, W, F]

        Returns:
            [dict] -- Feed dictionary for a tf session run call

        '''
        fd = dict()
        fd.update({self._input_image : images})
        if labels is not None:
            fd.update({self._input_labels : labels})

        return fd

    def losses():
        pass

    def make_summary(self, sess, input_data, input_label):
        fd = self.feed_dict(images  = input_data,
                            labels  = input_label)
        return sess.run(self._merged_summary, feed_dict=fd)

    def zero_gradients(self, sess):
        sess.run(self._zero_gradients)

    def accum_gradients(self, sess, input_data, input_label):

        feed_dict = self.feed_dict(images  = input_data,
                                   labels  = input_label)

        ops = [self._accum_gradients]
        doc = ['']
        # classification
        ops += [self._loss, self._total_accuracy]
        doc += ['loss', 'acc.']

        return sess.run(ops, feed_dict = feed_dict ), doc


    def run_test(self,sess, input_data, input_label):
        feed_dict = self.feed_dict(images   = input_data,
                                   labels   = input_label)

        ops = [self._loss, self._total_accuracy]
        doc = ['loss', 'acc.']

        return sess.run(ops, feed_dict = feed_dict ), doc

    def inference(self,sess,input_data,input_label=None):

        feed_dict = self.feed_dict(images=input_data, labels=input_label)

        ops = [self._softmax]
        if input_label is not None:
          ops.append(self._total_accuracy)

        return sess.run( ops, feed_dict = feed_dict )

    def global_step(self, sess):
        return sess.run(self._global_step)

    def _build_network(self, input_placeholder):

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


        # At this point, we ought to have a network that has the same shape as the initial input, but with more filters.
        # We can use a bottleneck to map it onto the right dimensions:
        x = tf.layers.conv2d(x,
                             self._params['NUM_LABELS'],
                             kernel_size=[7,7],
                             strides=[1, 1],
                             padding='same',
                             activation=None,
                             use_bias=False,
                             trainable=self._params['TRAINING'],
                             name="BottleneckConv2D")
        if verbose:
            print "Shape after bottleneck: " + str(x.get_shape())

        # And lastly, apply global average pooling to get to the correct final shape
        # For global average pooling, need to get the shape of the input:
        shape = (x.shape[1], x.shape[2])

        x = tf.nn.pool(x,
                       window_shape=shape,
                       pooling_type="AVG",
                       padding="VALID",
                       dilation_rate=None,
                       strides=None,
                       name="GlobalAveragePool",
                       data_format=None)
        if verbose:
            print "Shape after pooling: " + str(x.get_shape())

        # Reshape to remove empty dimensions:
        x = tf.reshape(x, [tf.shape(x)[0], self._params['NUM_LABELS']],
                     name="global_pooling_reshape")
        if verbose:
            print "Finalshape: " + str(x.get_shape())


        return x
