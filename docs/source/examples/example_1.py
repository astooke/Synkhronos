import numpy as np
import theano
import theano.tensor as T
import synkhronos as synk

synk.fork()
s_init = np.ones(3, dtype='float32')
x = T.matrix('x')
s = theano.shared(s_init, name='s')
f = synk.function([x], updates=[(s, T.sum(x * s, axis=0))])
synk.distribute()
x_dat = synk.data(np.array([[1, 1],
                            [2, 2],
                            [3, 3],
                            [4, 4]]).astype('float32'))
print("\ns initial:\n", s.get_value())

f.as_theano(x_dat.data)
print("\ns after Theano call:\n", s.get_value())

s.set_value(s_init)
f(x_dat)
print("\nlocal s after reset and Synkhronos call:\n", s.get_value())

gathered_s = synk.gather(s, nd_up=1)
print("\ngathered s:\n", gathered_s)

synk.reduce(s, op="sum")
print("\nlocal s after in-place reduce:\n", s.get_value())

gathered_s = synk.gather(s, nd_up=1)
print("\ngathered s after reduce:\n", gathered_s)

synk.broadcast(s, s_init)
f(x_dat)
synk.all_reduce(s, op="sum")
gathered_s = synk.gather(s, nd_up=1)
print("\ngathered s after local reset, broadcast, Synkhronos call, "
    "and all-reduce:\n", gathered_s)
