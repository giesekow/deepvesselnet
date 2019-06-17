from net import Network
from keras import regularizers as reg
from keras import optimizers as opt
from keras import backend as K
import numpy as np
import losses as ls
import metrics as mt
import misc as ms

class FCN(Network):
    def __init__(self, nchannels=1, nlabels=2, cross_hair=False, activation='tanh', **kwargs):
        inputs = {'main_input': {'shape': (nchannels, None, None, None), 'dtype': 'float32'}}
        layers = []
        levels = [
            {'filters': 5, 'kernel': 3},
            {'filters': 10, 'kernel': 5},
            {'filters': 20, 'kernel': 5},
            {'filters': 50, 'kernel': 3},
        ]

        conv = 'Conv3DCH' if cross_hair else 'Conv3D'
        curinputs = 'main_input'
        cnt = 0

        for level in levels:
            layers.append({
                'layer': conv,
                'inputs': curinputs,
                'sort': -cnt,
                'params': {
                    'name': 'level_'+str(cnt),
                    'filters': level['filters'],
                    'kernel_size': (level['kernel'],)*3,
                    'strides': (1, 1, 1),
                    'padding': 'same',
                    'activation': activation
                }
            })
            curinputs = 'level_'+str(cnt)
            cnt += 1

        layers.append({
            'layer': 'Conv3D',
            'inputs': curinputs,
            'sort': -(cnt+1),
            'params': {
                'name': 'presoftmax',
                'filters': nlabels,
                'kernel_size': (1, 1, 1),
                'strides': (1, 1, 1),
                'padding': 'same',
                'activation': 'linear',
                }
        })
        layers.append({
            'layer': 'Softmax',
            'inputs': 'presoftmax',
            'sort': -(cnt + 2),
            'params': {
                'name': 'output',
                'axis': 1
            }
        })
        layers = sorted(layers, key=lambda i: i['sort'], reverse=True)

        models = {'default': {'inputs': 'main_input', 'outputs': 'output'}}
        super(FCN, self).__init__(layers=layers, input_shapes=inputs, models=models, **kwargs)

    def compile(self, loss=ls.categorical_crossentropy(1), optimizer='sgd', metrics=['acc'], **kwargs):
        super(FCN, self).compile(models={'default': {'loss': loss, 'optimizer': optimizer, 'metrics': metrics}})

    def fit(self, **kwargs):
        return super(FCN, self).fit(model='default', **kwargs)

    def fit_generator(self, **kwargs):
        return super(FCN, self).fit_generator(model='default', **kwargs)

    def predict_generator(self, **kwargs):
        return super(FCN, self).predict_generator(model='default', **kwargs)

    def predict(self, **kwargs):
        return super(FCN, self).predict(model='default', **kwargs)

    def evaluate(self, **kwargs):
        return super(FCN, self).evaluate(model='default', **kwargs)


if __name__ == '__main__':
    net = FCN(cross_hair=True)
    net.compile(loss=ls.weighted_categorical_crossentropy_with_fpr())
    N = (10, 1, 64, 64, 64)
    X = np.random.random(N)
    Y = np.random.randint(2, size=N)
    Y = np.squeeze(Y)
    Y = ms.to_one_hot(Y)
    Y = np.transpose(Y, axes=[0,4,1,2,3])

    print 'Testing FCN Network'
    print 'Data Information => ', 'volume size:', X.shape, ' labels:',np.unique(Y)
    net.fit(x=X, y=Y, epochs=30, batch_size=2, shuffle=True)
