import sys
import time


import tensorflow as tf


# Declaring exception names:
class ConfigurationException(Exception): pass
class IncompleteFeedDict(Exception): pass



# Main class
class resnetcore(object):
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

    def construct_network(self, dims, label_dims):
        '''Build the network model

        Initializes the tensorflow model according to the parameters
        '''

        tf.reset_default_graph()


        start = time.time()
        # Initialize the input layers:
        self._input_image  = tf.placeholder(tf.float32, dims, name="input_image")

        self._input_labels = dict()

        for label_name in label_dims.keys():
            y = tf.placeholder(tf.int64,
                               label_dims[label_name],
                               name="label_{0}".format(label_name))
            self._input_labels.update({label_name : y })


        sys.stdout.write(" - Finished input placeholders [{0:.2}s]\n".format(time.time() - start))
        start = time.time()
        logits = self._build_network(self._input_image, label_dims)

        sys.stdout.write(" - Finished Network graph [{0:.2}s]\n".format(time.time() - start))
        start = time.time()
        # for p in xrange(len(logits_by_plane)):
        #     print "logits_by_plane[{0}].get_shape(): ".format(p) + str(logits_by_plane[p].get_shape())
        self._softmax = dict()
        self._predicted_labels = dict()
        for label_name in logits.keys():
            self._softmax.update(
                {label_name : tf.nn.softmax(logits[label_name]) }
                )

            self._predicted_labels.update(
                {label_name : tf.argmax(logits[label_name], axis=-1 ) }
                )


        # Keep a list of trainable variables for minibatching:
        with tf.variable_scope('gradient_accumulation'):
            self._accum_vars = [tf.Variable(tv.initialized_value(),
                                trainable=False) for tv in tf.trainable_variables()]

        sys.stdout.write(" - Finished gradient accumulation [{0:.2}s]\n".format(time.time() - start))
        start = time.time()



        # Accuracy calculations:
        with tf.name_scope('accuracy'):
            self._accuracy = dict()
            for label_name in logits.keys():

                correct_prediction = tf.equal(tf.argmax(self._input_labels[label_name], -1),
                                              tf.argmax(self._predicted_labels[label_name], -1))
                self._accuracy.update({
                    label_name : tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    })


                # Add the accuracies to the summary:
                tf.summary.scalar("{0}_Accuracy".format(label_name), self._accuracy[label_name])

        sys.stdout.write(" - Finished accuracy [{0:.2}s]\n".format(time.time() - start))
        start = time.time()

        # Loss calculations:
        with tf.name_scope('cross_entropy'):
            self._losses = dict()
            for label_name in logits.keys():
                _loss = tf.reduce_sum(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self._input_labels[label_name],
                                                            logits=logits[label_name]))

                tf.summary.scalar("{0}_Loss".format(label_name), _loss)
                self._losses.update({label_name : _loss})

            self._loss = sum(self._losses.itervalues())

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
            for label_name in labels.keys():
                fd.update({self._input_labels[label_name] : labels[label_name]})

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
        ops += [self._loss]
        doc += ['loss']

        for label_name in self._accuracy.keys():
            ops += [self._accuracy[label_name]]
            doc += ["acc. {0}".format(label_name)]

        return sess.run(ops, feed_dict = feed_dict ), doc


    def run_test(self,sess, input_data, input_label):
        feed_dict = self.feed_dict(images   = input_data,
                                   labels   = input_label)

        ops = [self._loss]
        doc = ['loss']
        for label_name in self._accuracy.keys():
            ops += [self._accuracy[label_name]]
            doc += ["acc. {0}".format(label_name)]

        return sess.run(ops, feed_dict = feed_dict ), doc

    def inference(self,sess,input_data,input_label=None):

        feed_dict = self.feed_dict(images=input_data, labels=input_label)

        ops = [self._softmax]
        if input_label is not None:
            for label_name in self._accuracy.keys():
                ops += [self._accuracy[label_name]]

        return sess.run( ops, feed_dict = feed_dict )

    def global_step(self, sess):
        return sess.run(self._global_step)

    def _build_network(self, input_placeholder, label_dims):
        raise NotImplementedError("_build_network must be implemented by the child class")
