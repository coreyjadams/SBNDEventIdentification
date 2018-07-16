
    def _create_softmax(self, logits):
        '''Must return a dict type

        [description]

        Arguments:
            logits {[type]} -- [description]

        Raises:
            NotImplementedError -- [description]
        '''

        # For the logits, we compute the softmax and the predicted label


        output = dict()

        output['softmax'] = tf.nn.softmax(logits)
        output['prediction'] = tf.argmax(logits, axis=-1)

        return output





    def _calculate_loss(self, inputs, outputs):
        ''' Calculate the loss.

        returns a single scalar for the optimizer to use.
        '''


        with tf.name_scope('cross_entropy'):

            else:
                #otherwise, just one set of logits, against one label:
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=inputs['label'],
                                                            logits=outputs))



            # If desired, add weight regularization loss:
            if 'REGULARIZE_WEIGHTS' in self._params:
                reg_loss = tf.losses.get_regularization_loss()
                loss += reg_loss


            # Total summary:
            tf.summary.scalar("Total Loss",loss)

            return loss

    def _calculate_accuracy(self, inputs, outputs):
        ''' Calculate the accuracy.

        '''

        # Compare how often the input label and the output prediction agree:

        with tf.name_scope('accuracy'):

            if isinstance(outputs['prediction'], dict):
                accuracy = dict()

                for key in outputs['prediction'].keys():
                    correct_prediction = tf.equal(tf.argmax(inputs['label'][key], -1),
                                                  outputs['prediction'][key])
                    accuracy.update({
                        key : tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    })


                    # Add the accuracies to the summary:
                    tf.summary.scalar("{0}_Accuracy".format(key), accuracy[key])

            else:
                correct_prediction = tf.equal(tf.argmax(inputs['label'], -1),
                                              outputs['prediction'])
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar("Accuracy", accuracy)

        return accuracy
