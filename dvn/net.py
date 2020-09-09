from __future__ import print_function
import numpy as np
np.random.seed(123)  # for reproducibility
from keras.models import Model
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K
import keras
import os
from random import shuffle
from keras import layers as KL
import pickle as pickle
from .layers import objects

for k in objects:
    setattr(KL, k, objects[k])

class Network(object):
    def __init__(self, layers, input_shapes={}, input_tensors={}, models={}, customLayers={}, **kwargs):
        self.layers = layers
        self.outputs = {}
        self.models = {}
        self.modelParams = models

        for k in input_shapes:
             sh = input_shapes[k]
             self.outputs[k] = KL.Input(shape=sh['shape'], dtype=sh['dtype'], name=k)

        for k in input_tensors:
            sh = input_tensors[k]
            self.outputs[k] = input_tensors[k]

        self.built = False

        for k in customLayers:
            setattr(KL, k, customLayers[k])

        self.params = {'layers':layers, 'input_shapes': input_shapes, 'models': models}
        self.params.update(**kwargs)
        self.compiled = False

    def build(self, **kwargs):
        if self.built:
            return

        for layer in self.layers:
            lay = layer['layer']
            if isinstance(layer['inputs'], list):
                inputs = []
                for i in layer['inputs']:
                    inputs.append(self.outputs[i])
            else:
                inputs = self.outputs[layer['inputs']]

            if isinstance(lay, str):
                lay =  getattr(KL, lay)

            params = layer['params']
            cur_input = lay(**params)(inputs)
            self.outputs[params['name']] = cur_input

        for k in self.modelParams:
            p = self.modelParams[k]
            inputs = []
            outputs = []
            if isinstance(p['inputs'], list):
                for inp in p['inputs']:
                    inputs.append(self.outputs[inp])
            else:
                inputs = self.outputs[p['inputs']]

            if isinstance(p['outputs'], list):
                for out in p['outputs']:
                    outputs.append(self.outputs[out])
            else:
                outputs = self.outputs[p['outputs']]
            self.models[k] = Model(inputs, outputs)

        self.built = True

    def add_model(self, models, update=False, **kwargs):
        if not self.built:
            self.build()

        for k in models:
            p = models[k]

            if update:
                self.params['models'][k] = p

            inputs = []
            outputs = []
            if isinstance(p['inputs'], list):
                for inp in p['inputs']:
                    inputs.append(self.outputs[inp])
            else:
                inputs = self.outputs[p['inputs']]

            if isinstance(p['outputs'], list):
                for out in p['outputs']:
                    outputs.append(self.outputs[out])
            else:
                outputs = self.outputs[p['outputs']]
            self.models[k] = Model(inputs, outputs)

        self.built = True

    def compile(self, models, **kwargs):
        self._compile(models, **kwargs)

    def _compile(self, models, **kwargs):
        if not self.built:
            self.build()

        for k in models:
            if k in self.models:
                params = models[k]
                self.models[k].compile(**params)

        self.compiled = True

    def _fit(self, x, y, model, **kwargs):
        return self.models[model].fit(x, y, **kwargs)

    def fit(self, x, y, model, **kwargs):
        kwargs['x'] = x
        kwargs['y'] = y
        kwargs['model'] = model
        return self._fit(**kwargs)

    def _fit_generator(self, model, **kwargs):
        return self.models[model].fit_generator(**kwargs)

    def fit_generator(self, model, **kwargs):
        kwargs['model'] = model
        return self._fit_generator(**kwargs)

    def _predict_generator(self, model, **kwargs):
        return self.models[model].predict_generator(**kwargs)

    def predict_generator(self, model, **kwargs):
        kwargs['model'] = model
        return self._fit_generator(**kwargs)

    def _predict(self, x, model, **kwargs):
        return self.models[model].predict(x, **kwargs)

    def predict(self, x, model, **kwargs):
        kwargs['x'] = x
        kwargs['model'] = model
        return self._predict(**kwargs)

    def _evaluate(self, x, y, model, **kwargs):
        return self.models[model].evaluate(x, y, **kwargs)

    def evaluate(self, x, y, model, **kwargs):
        kwargs['x'] = x
        kwargs['y'] = y
        kwargs['model'] = model
        return self._evaluate(**kwargs)

    def save(self, filename):
        weights = {}
        for k in self.models:
            model = self.models[k]
            for layer in model.layers:
                weights[layer.name] = layer.get_weights()

        try:
            data = {'params': self.params, 'weights': weights}
            s = open(filename, 'wb')
            pickle.dump(data, s)
            print('Models successfully saved')
        except:
            print('Unable to save model')

    def size(self):
        total = 0.
        for k in self.models:
            model = self.models[k]
            for layer in model.layers:
                for w in layer.get_weights():
                    total = total + np.product(w.shape)
        return total

    @classmethod
    def load(cls, filename, customLayers={}, input_tensors={}):
        try:
            s = open(filename, 'rb')
            data = pickle.load(s, encoding='latin1')
            params = data['params']
            weights = data['weights']
            params['customLayers'] = customLayers
            params['input_tensors'] = input_tensors
            params['loading'] = True

            net = cls(**params)
            net.build()

            for k in net.models:
                model = net.models[k]
                for layer in model.layers:
                    if layer.name in weights:
                        layer.set_weights(weights[layer.name])

            print('Models successfully loaded')
            return net
        except:
            print('Unable to load model')
            return None
        
    @classmethod
    def convert_model(source, destination):
        content = ''
        outsize = 0
        with open(source, 'rb') as infile:
            content = infile.read()
            
        with open(destination, 'wb') as output:
            for line in content.splitlines():
                outsize += len(line) + 1
                output.write(line + str.encode('\n'))

        print("Done. Saved %s bytes." % (len(content)-outsize))

    @classmethod
    def size_from_file(fname):
        s = open(fname)
        data = pickle.load(s)
        keys = data['weights'].keys()
        total = 0.
        for k in keys:
            for w in data['weights'][k]:
                total = total + np.product(w.shape)
        s.close()
        return total


if __name__ == '__main__':
    pass
