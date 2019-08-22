from dvn import FCN
import numpy as np
import dvn.misc as ms
import dvn.losses as ls


dim = 2
net = FCN(cross_hair=True, dim=dim)
net.compile(loss=ls.weighted_categorical_crossentropy_with_fpr())
N = (10, 1,) +(64,)*dim
X = np.random.random(N)
Y = np.random.randint(2, size=N)
Y = np.squeeze(Y)
Y = ms.to_one_hot(Y)
Y = np.transpose(Y, axes=[0,dim+1] + list(range(1,dim+1)))
print('Testing FCN Network')
print('Data Information => ', 'volume size:', X.shape, ' labels:',np.unique(Y))
net.fit(x=X, y=Y, epochs=30, batch_size=2, shuffle=True)