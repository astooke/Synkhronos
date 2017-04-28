
import synkhronos as synk
import numpy as np
import theano

synk.fork()


s = theano.shared(np.ones([5, 5], dtype='float32'), name="shared_var")
s2 = theano.shared(np.ones([4, 4], dtype='float32'), name="shared_var_2")

f = synk.function([], [s.dot(s), s2.dot(s2)])

synk.distribute()

# print(f())

# print(synk.get_value(1, s))

# d = 2 * np.ones([5, 5], dtype='float32')

# synk.set_value(1, s, d)

d55 = np.array(list(range(5 * 5)), dtype='float32').reshape(5, 5)
d64 = np.array(list(range(6 * 4)), dtype='float32').reshape(6, 4)
