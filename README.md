#Dependencies
1. keras
2. tensorflow / theano as backend for keras (tensoflow recommended)


#Installation
`python setup.py install`

#Usage

`from dvn import FCN, VNET, UNET  # import libraries`

`net = FCN()                                 # create the network object (You can replace FCN with VNET or UNET)`<br>
`net.compile()                               # compile the network`<br>
`net.fit(x=X, y=Y, epochs=10, batch_size=10) # train the network`<br>
`preds = net.predict(x=X)                    # predict`<br>
`net.save(filename='model.dat')              # save network params`<br>
`net = FCN.load(filename='model.dat')        # Load network params  (You can replace FCN with VNET or UNET as used above)`<br>
