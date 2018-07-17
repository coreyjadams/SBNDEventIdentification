import os
import sys
import time

import numpy

import tensorflow as tf

import uresnet
# import uresnet, uresnet3d
import trainercore


class uresnet_trainer(trainercore.trainercore):

    def __init__(self, config):
        super(uresnet_trainer, self).__init__(config)

        if not self.check_params():
            raise Exception("Parameter check failed.")

        if '3d' in config['NAME']:
            net = uresnet3d.uresnet3d()
        else:
            net = uresnet.uresnet()

        net.set_params(config['NETWORK'])

        self.set_network_object(net)


    def fetch_minibatch_data(self, mode):
        # Return a dictionary object with keys 'image', 'label', and others as needed
        #            minibatch_data   = self._dataloaders['train'].fetch_data(
        #        self._config['TRAIN_CONFIG']['KEYWORD_DATA']).data()

        this_data = dict()
        this_data['image'] = self._dataloaders[mode].fetch_data(
            self._config['IO'][mode]['KEYWORD_DATA']).data()

        this_data['label'] = self._dataloaders[mode].fetch_data(
            self._config['IO'][mode]['KEYWORD_LABEL']).data()


        # If the weights for each pixel are to be normalized, compute the weights too:
        if self._config['NETWORK']['BALANCE_LOSS']:
            this_data['weight'] = self.compute_weights(this_data['image'])


        return this_data

    def fetch_minibatch_dims(self, mode):
        # Return a dictionary object with keys 'image', 'label', and others as needed
        # self._dataloaders['train'].fetch_data(keyword_label).dim() as an example
        this_dims = dict()
        this_dims['image'] = self._dataloaders[mode].fetch_data(
            self._config['IO'][mode]['KEYWORD_DATA']).dim()

        this_dims['label'] = self._dataloaders[mode].fetch_data(
            self._config['IO'][mode]['KEYWORD_LABEL']).dim()

        # If the weights for each pixel are to be normalized, compute the weights too:
        if self._config['NETWORK']['BALANCE_LOSS']:
            this_dims['weight'] = this_dims['image']

        return this_dims



    def compute_weights(self, labels, boost_labels = None):
        # Take the labels, and compute the per-label weight

        # Prepare output weights:
        weights = numpy.zeros(labels.shape)

        i = 0
        for batch in labels:
            # First, figure out what the labels are and how many of each:
            values, counts = numpy.unique(batch, return_counts=True)

            n_pixels = numpy.sum(counts)
            for value, count in zip(values, counts):
                weight = 1.0*(n_pixels - count) / n_pixels
                if boost_labels is not None and value in boost_labels.keys():
                    weight *= boost_labels[value]
                mask = labels[i] == value
                weights[i, mask] += weight

            # Normalize the weights to sum to 1 for each event:
            weights[i] *= 1. / numpy.sum(weights[i])
            i += 1


        return weights

