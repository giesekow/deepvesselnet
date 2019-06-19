#Dependencies
1. keras
2. tensorflow / theano as backend for keras (tensoflow recommended)
3. sklearn
4. numpy


#Installation
`python setup.py install`

#Usage
```
from dvn import FCN, VNET, UNET  # import libraries

net = FCN()                                 # create the network object (You can replace FCN with VNET or UNET)
net.compile()                               # compile the network (supports keras compile parameters)
net.fit(x=X, y=Y, epochs=10, batch_size=10) # train the network (supports keras fit parameters)
preds = net.predict(x=X)                    # predict (supports keras predict parameters)
net.save(filename='model.dat')              # save network params
net = FCN.load(filename='model.dat')        # Load network params  (You can replace FCN with VNET or UNET as used above)
```
