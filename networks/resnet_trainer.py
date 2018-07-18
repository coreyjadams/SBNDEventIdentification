import os
import sys
import time

import numpy

import tensorflow as tf

import resnet, resnet3d
import trainercore


class resnet_trainer(trainercore.trainercore):

    def __init__(self, config):
        super(resnet_trainer, self).__init__(config)

        if not self.check_params():
            raise Exception("Parameter check failed.")

        if '3d' in config['NAME']:
            net = resnet3d.resnet3d()
        else:
            net = resnet.resnet()

        net.set_params(config['NETWORK'])

        self.set_network_object(net)


    def fetch_minibatch_data(self, mode):
        # Return a dictionary object with keys 'image', 'label', and others as needed
        #            minibatch_data   = self._dataloaders['train'].fetch_data(
        #        self._config['TRAIN_CONFIG']['KEYWORD_DATA']).data()

        this_data = dict()
        this_data['image'] = self._dataloaders[mode].fetch_data(
            self._config['IO'][mode]['KEYWORD_DATA']).data()
#
        # Figure out if the label is a dict or not:
        if isinstance(self._config['IO'][mode]['KEYWORD_LABEL'], str):
            this_data['label'] = self._dataloaders[mode].fetch_data(
                self._config['IO'][mode]['KEYWORD_LABEL']).data()
        elif isinstance(self._config['IO'][mode]['KEYWORD_LABEL'], list):
            this_data['label'] = dict()
            for key in self._config['IO'][mode]['KEYWORD_LABEL']:
                hash_key = self.long_key_to_short_key(key)
                this_data['label'][hash_key] = self._dataloaders[mode].fetch_data(key).data()

        # code to flatten labels into one-hot-coded vector
        if self._config['ONE_HOT']:

            dims = self.fetch_minibatch_dims(mode)
            unpacked_dims = numpy.asarray([ dims['unpacked_label'][dim] for dim in dims['unpacked_label']])
            dim_lengths = unpacked_dims[:,-1]

            onehot_label_holder = numpy.zeros(shape=dims['label'])

            for batch in range(self._config['MINIBATCH_SIZE']):
                unpacked_data = [ numpy.argmax(this_data['label'][dim][batch]) for dim in this_data['label'] ]

                global_index = numpy.ravel_multi_index(unpacked_data, dim_lengths)

                onehot_label_holder[batch, global_index] = 1

            # Persist the original data:
            # this_data['unpacked_label'] = this_data['label']
            this_data['label'] = onehot_label_holder

        return this_data

    def fetch_minibatch_dims(self, mode):
        # Return a dictionary object with keys 'image', 'label', and others as needed
        # self._dataloaders['train'].fetch_data(keyword_label).dim() as an example
        this_dims = dict()
        this_dims['image'] = self._dataloaders[mode].fetch_data(
            self._config['IO'][mode]['KEYWORD_DATA']).dim()

        # Figure out if the label is a dict or not:
        if isinstance(self._config['IO'][mode]['KEYWORD_LABEL'], str):
            this_dims['label'] = self._dataloaders[mode].fetch_data(
                self._config['IO'][mode]['KEYWORD_LABEL']).dim()
        elif isinstance(self._config['IO'][mode]['KEYWORD_LABEL'], list):
            this_dims['label'] = dict()
            for key in self._config['IO'][mode]['KEYWORD_LABEL']:
                hash_key = self.long_key_to_short_key(key)
                this_dims['label'][hash_key] = self._dataloaders[mode].fetch_data(key).dim()

        # Here could be added code to flatten the dims into one long label dims

        if self._config['ONE_HOT'] and isinstance(this_dims['label'], dict):
            this_dims['unpacked_label'] = this_dims['label']
            dims = numpy.asarray([ this_dims['label'][dim] for dim in this_dims['label']])
            total_len = numpy.prod(dims[:,-1])
            this_dims['label'] = numpy.array([self._config['MINIBATCH_SIZE'], total_len])

        return this_dims


    def long_key_to_short_key(self, key):
        # This function only exists to convert things like so:
        # main_neutrino_label -> neutrino
        # val_chrpion_label -> chrpion
        # etc...

        return key.split('_')[1]

