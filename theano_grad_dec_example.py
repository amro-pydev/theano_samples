import theano
from theano import tensor as T
import numpy as np

import matplotlib.pyplot as plt


trX = np.linspace(-1,1, 101)
trY = 2.0 * trX + np.random.randn(*trX.shape) * 0.33

X = T.scalar()
Y = T.scalar()

def model(X, w):
    return X * w

w = theano.shared(np.asarray(0.0, dtype=theano.config.floatX))
y = model(X, w)


cost = T.mean(T.sqr(y - Y))
gradient = T.grad(cost=cost, wrt=w)
updates = [[w, w - gradient * 0.01]]

train = theano.function(inputs=[X,Y], outputs=cost, updates=updates, allow_input_downcast=True)

plt.ion()

for i in range(100):
    plt.clf()
    plt.plot(trX,trY,'o')
    plt.plot(trX,trX * w.get_value(),"--",linewidth=2)
    plt.title('Current Slope = %s' % str(w.get_value()))
    plt.draw()
    plt.pause(0.0001)
    print("i:", i)
    
    for x, y in zip(trX, trY):
        train(x, y)

