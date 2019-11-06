from __future__ import print_function
from .net import Network
from keras import regularizers as reg
from keras import optimizers as opt
from keras import backend as K
import numpy as np
from . import losses as ls
from . import metrics as mt
from . import misc as ms

class UNET(Network):
    "Implementation of UNET (2D/3D) with or without cross hair filters"
    def __init__(self, nchannels=1, nlabels=2, nlevels=4, nfeats=32, cross_hair=False, dim=3, activation='tanh', **kwargs):
        """ Builds the network structure  based on the parameters provided
        parameters:
            nchannels (int) : the number of input channels to the network.
            nlabels (int)   : the number of labels to be predicted.
            nlevels (int)   : the number of levels of the VNET, represents how deep the network is.
            nfeats (int)    : the number of fetures generated at the first level of the network next levels will be twice the previous level
            cross_hair (bool): whether to use cross hair filters or classical full convolutions (Read Tetteh et al.)
            dim (int)       : the dimension of the network whether its 2D or 3D options are [2, 3] for 2D and 3D versions respectively
            activation (keras supported activation)      : the activation function used in the layers of the network
        """
        if 'loading' in kwargs:
            super(UNET, self).__init__(**kwargs)
            return

        inputs = {'main_input': {'shape': (nchannels,) +(None,)*dim, 'dtype': 'float32'}}
        cfeats = nfeats
        layers = []

        conv = 'Conv3DCH' if cross_hair else 'Conv3D'

        if dim == 2:
            conv = 'Conv2DCH' if cross_hair else 'Conv2D'

        kernel = (5,)*dim
        curinputs = 'main_input'

        for level in  range(nlevels):
            steps = 2
            for step in range(steps):
                layers.append(
                    {
                        'layer': conv,
                        'sort':(nlevels-level) * 10 + 9 - step,
                        'inputs': curinputs,
                        'params': {
                            'name': 'encoder_'+str(level)+str(step),
                            'filters': cfeats,
                            'kernel_size': kernel,
                            'strides': (1,)*dim,
                            'padding': 'same',
                            'activation': 'linear',
                        }
                    }
                )
                layers.append({
                    'layer': 'BatchNormalization',
                    'sort': (nlevels-level) * 10 + 9 - (step+0.2),
                    'inputs': 'encoder_'+str(level)+str(step),
                    'params': {
                        'name': 'encoder_'+str(level)+str(step)+'_bn',
                        'axis': 1
                    }
                })
                layers.append({
                    'layer': 'Activation',
                    'sort': (nlevels-level) * 10 + 9 - (step+0.4),
                    'inputs': 'encoder_'+str(level)+str(step)+'_bn',
                    'params': {
                        'name': 'encoder_'+str(level)+str(step)+'_act',
                        'activation': activation
                    }
                })
                curinputs = 'encoder_'+str(level)+str(step)+'_act'

                if step == 0:
                    cfeats = cfeats * 2

            levelskip = curinputs

            if level < nlevels - 1:
                layers.append({
                    'layer': conv,
                    'sort': (nlevels-level)*10 + 9 - (steps + 1),
                    'inputs': curinputs,
                    'params': {
                        'name': 'encoder_'+str(level)+'_subsample',
                        'filters': cfeats,
                        'kernel_size': kernel,
                        'strides': (2,)*dim,
                        'padding': 'same',
                        'activation': 'linear'
                    }
                })

                layers.append({
                    'layer': 'Concatenate',
                    'sort': -((nlevels-level)*10),
                    'inputs': [levelskip, 'decoder_'+str(level+1)+'_subsample'],
                    'params': {
                        'name': 'decoder_'+str(level)+'_concat',
                        'axis': 1
                    }
                })
                curinputs = 'decoder_'+str(level)+'_concat'

                for step in range(steps):
                    layers.append({
                        'layer': conv,
                        'sort': -((nlevels-level)*10+1+(step)),
                        'inputs': curinputs,
                        'params': {
                            'name': 'decoder_'+str(level)+str(step),
                            'filters': cfeats,
                            'kernel_size': kernel,
                            'strides': (1,)*dim,
                            'padding': 'same',
                            'activation': 'linear'
                        }
                    })
                    layers.append({
                        'layer': 'BatchNormalization',
                        'sort': -((nlevels-level) * 10 + 1 + (step+0.2)),
                        'inputs': 'decoder_'+str(level)+str(step),
                        'params': {
                            'name': 'decoder_'+str(level)+str(step)+'_bn',
                            'axis': 1,
                        }
                    })
                    layers.append({
                        'layer': 'Activation',
                        'sort': -((nlevels-level) * 10 + 1 + (step+0.4)),
                        'inputs': 'decoder_'+str(level)+str(step)+'_bn',
                        'params': {
                            'name': 'decoder_'+str(level)+str(step)+'_act',
                            'activation': activation
                        }
                    })
                    curinputs = 'decoder_'+str(level)+str(step)+'_act'

            if level > 0:
                layers.append({
                    'layer': 'Conv3DTranspose' if dim==3 else 'Conv2DTranspose',
                    'inputs': curinputs,
                    'sort': -((nlevels - level) * 10 + 1 + (steps)),
                    'params': {
                        'name': 'decoder_'+str(level)+'_subsample',
                        'filters': cfeats if level < nlevels-1 else cfeats * 2,
                        'kernel_size': kernel,
                        'strides': (2,)*dim,
                        'padding': 'same',
                        'activation': activation,
                    }
                })
            else:
                layers.append({
                    'layer': 'Conv3D' if dim==3 else 'Conv2D',
                    'inputs': curinputs,
                    'sort': -((nlevels - level) * 10 + 1 + (steps) + 1),
                    'params': {
                        'name': 'presoftmax',
                        'filters': nlabels,
                        'kernel_size': (1,)*dim,
                        'strides': (1,)*dim,
                        'padding': 'same',
                        'activation': 'linear',
                    }
                })
                layers.append({
                    'layer': 'Softmax',
                    'inputs': 'presoftmax',
                    'sort': -((nlevels - level) * 10 + 1 + (steps) + 2),
                    'params': {
                        'name': 'output',
                        'axis': 1
                    }
                })

            curinputs = 'encoder_'+str(level)+'_subsample'

        layers = sorted(layers, key=lambda i: i['sort'], reverse=True)
        models = {'default': {'inputs': 'main_input', 'outputs': 'output'}}
        kwargs['layers'] = layers
        kwargs['models'] = models
        kwargs['input_shapes'] = inputs
        super(UNET, self).__init__(**kwargs)

    def compile(self, loss=ls.categorical_crossentropy(1), optimizer='sgd', metrics=['acc'], **kwargs):
        super(UNET, self).compile(models={'default': {'loss': loss, 'optimizer': optimizer, 'metrics': metrics}})

    def fit(self, **kwargs):
        return super(UNET, self).fit(model='default', **kwargs)

    def fit_generator(self, **kwargs):
        return super(UNET, self).fit_generator(model='default', **kwargs)

    def predict_generator(self, **kwargs):
        return super(UNET, self).predict_generator(model='default', **kwargs)

    def predict(self, **kwargs):
        return super(UNET, self).predict(model='default', **kwargs)

    def evaluate(self, **kwargs):
        return super(UNET, self).evaluate(model='default', **kwargs)


if __name__ == '__main__':
    dim = 2
    net = UNET(cross_hair=True,dim=dim)
    net.compile()
    N = (10, 1) + (64, )*dim
    X = np.random.random(N)
    Y = np.random.randint(2, size=N)
    Y = np.squeeze(Y)
    Y = ms.to_one_hot(Y)
    Y = np.transpose(Y, axes=[0,dim+1] + list(range(1,dim+1)))
    print('Testing UNET Network')
    print('Data Information => ', 'volume size:', X.shape, ' labels:',np.unique(Y))
    net.fit(x=X, y=Y, epochs=10, batch_size=2, shuffle=True)
