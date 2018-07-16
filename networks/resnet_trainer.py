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

        # Figure out if the label is a dict or not:
        if isinstance(self._config['IO'][mode]['KEYWORD_LABEL'], str):
            this_data['label'] = self._dataloaders[mode].fetch_data(
                self._config['IO'][mode]['KEYWORD_LABEL']).data()
        elif isinstance(self._config['IO'][mode]['KEYWORD_LABEL'], list):
            this_data['label'] = dict()
            for key in self._config['IO'][mode]['KEYWORD_LABEL']:
                hash_key = self.long_key_to_short_key(key)
                this_data['label'][hash_key] = self._dataloaders[mode].fetch_data(key).data()

        # Here could be added code to flatten the labels into one long label

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

        return this_dims


    def long_key_to_short_key(self, key):
        # This function only exists to convert things like so:
        # main_neutrino_label -> neutrino
        # val_chrpion_label -> chrpion
        # etc...

        return key.split('_')[1]

    def unpack_labels(self, long_labels):
        pass

    def pack_labels(self, neutrino_label, proton_label, chrpion_label, ntrpion_label):

        # Roll up the labels into one long label:
        pass