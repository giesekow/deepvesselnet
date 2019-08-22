from keras import backend as K
from keras import layers as KL

class DimShuffle():
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.name = kwargs['name']

    def __call__(self,inputs, **kwargs):
        def layer(x):
            ndim = K.ndim(x)
            axes = []
            cnt = 0
            c = x
            for i in self.kwargs['permutation']:
                if i == 'x':
                    c = K.expand_dims(c,-1)
                    axes.append(ndim + cnt)
                    cnt += 1
                else:
                    axes.append(i)
            return K.permute_dimensions(c, axes)

        return KL.Lambda(layer)(inputs)

class Softmax():
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.name = kwargs['name']
        self.axis = kwargs['axis']

    def __call__(self,inputs, **kwargs):
        def layer(x):
            y = K.exp(x - K.max(x, self.axis, keepdims=True))
            return y / K.sum(y, self.axis, keepdims=True)

        return KL.Lambda(layer)(inputs)

class Convolution3DCH():
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.name = kwargs['name']
        self.filters = kwargs['filters']
        self.kernel = kwargs['kernel_size']
        self.activation = 'linear'
        if 'activation' in kwargs:
            self.activation = kwargs['activation']
            kwargs['activation'] = 'linear'
        if isinstance(self.kernel, int):
            ks = self.kernel
            self.kernel = (ks, ks, ks)
        ks = self.kernel
        kwargs['kernel_size'] = (1, ks[1], ks[2])
        kwargs['name'] = self.name + '_x_axis'
        kwargs['data_format'] = 'channels_first'
        self.convx = KL.Convolution3D(**kwargs)
        kwargs['kernel_size'] = (ks[0], 1, ks[2])
        kwargs['name'] = self.name + '_y_axis'
        kwargs['data_format'] = 'channels_first'

        self.convy = KL.Convolution3D(**kwargs)
        kwargs['kernel_size'] = (ks[0], ks[1], 1)
        kwargs['name'] = self.name + '_z_axis'
        kwargs['data_format'] = 'channels_first'

        self.convz = KL.Convolution3D(**kwargs)
        self.addLayer = KL.Add(name=self.name + '_add')

    def __call__(self,inputs, **kwargs):
        x = self.convx(inputs)
        y = self.convy(inputs)
        z = self.convz(inputs)
        out = self.addLayer([x, y, z])
        if self.activation != 'linear':
            out = KL.Activation(self.activation)(out)
        return out

    def get_weights(self, **kwargs):
        weights = {
            'x_weights': self.convx.get_weights(),
            'y_weights': self.convy.get_weights(),
            'z_weights': self.convz.get_weights()
        }
        return weights

    def set_weights(self, weights, **kwargs):
        if 'x_weights' in weights:
            self.convx.set_weights(weights['x_weights'])
        if 'y_weights' in weights:
            self.convy.set_weights(weights['y_weights'])
        if 'z_weights' in weights:
            self.convz.set_weights(weights['z_weights'])

class Convolution2DCH():
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.name = kwargs['name']
        self.filters = kwargs['filters']
        self.kernel = kwargs['kernel_size']
        self.activation = 'linear'

        if 'activation' in kwargs:
            self.activation = kwargs['activation']
            kwargs['activation'] = 'linear'

        if isinstance(self.kernel, int):
            ks = self.kernel
            self.kernel = (ks, ks)

        ks = self.kernel

        kwargs['kernel_size'] = (1, ks[1])
        kwargs['name'] = self.name + '_x_axis'
        kwargs['data_format'] = 'channels_first'
        self.convx = KL.Convolution2D(**kwargs)
        kwargs['kernel_size'] = (ks[0], 1)
        kwargs['name'] = self.name + '_y_axis'
        kwargs['data_format'] = 'channels_first'
        self.convy = KL.Convolution2D(**kwargs)
        self.addLayer = KL.Add(name=self.name + '_add')

    def __call__(self,inputs, **kwargs):
        x = self.convx(inputs)
        y = self.convy(inputs)
        out = self.addLayer([x, y])
        if self.activation != 'linear':
            out = KL.Activation(self.activation)(out)
        return out

    def get_weights(self, **kwargs):
        weights = {
            'x_weights': self.convx.get_weights(),
            'y_weights': self.convy.get_weights()
        }
        return weights

    def set_weights(self, weights, **kwargs):
        if 'x_weights' in weights:
            self.convx.set_weights(weights['x_weights'])
        if 'y_weights' in weights:
            self.convy.set_weights(weights['y_weights'])

objects = {
    'Dimshuffle': DimShuffle,
    'Softmax': Softmax,
    'Convolution3DCH': Convolution3DCH,
    'Convolution2DCH': Convolution2DCH,
    'Conv3D': KL.Convolution3D,
    'Conv2D': KL.Convolution2D,
    'Conv3DCH': Convolution3DCH,
    'Conv2DCH': Convolution2DCH
}

if __name__ == '__main__':
    pass
