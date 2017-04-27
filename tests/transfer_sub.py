
import theano
import numpy as np
s = theano.shared(np.ones([3, 3], dtype='float32'))
sub = s[1:3]
f = theano.function([], s)
g = theano.function([], sub)
h = theano.function([], sub.transfer(None))
print(type(f()))
print(type(g()))
print(type(h()))

