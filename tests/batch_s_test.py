

import theano
import synkhronos as synk
import numpy as np

synk.fork()

s = theano.shared(np.zeros([100, 2], dtype='float32'), name='shared_var')
# s = theano.shared(np.array(list(range(100 * 2)), dtype='float32').reshape(100, 2))

f = synk.function([], outputs=(s, "gather"), sliceable_shareds=[s])


synk.distribute()


d = np.ones([200, 2], dtype='float32')
for i, row in enumerate(d):
    row *= i

sd = synk.data(value=d)

synk.scatter(s, sd)

print(f())
print("\n")
print(f(num_slices=3))
print("\n")
print(f(batch_s=[0, 1, 2, 3, 4, 5]))
print("\n")
print(f(batch_s=[0, 1, 2, 3, 4, 5], num_slices=2))
print("\n")
print(f(batch_s=[49, 23, 1, 7, 23]))
print("\n")
print(f(batch_s=[[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]]))
print("\n")
print(f(batch_s=slice(3, 17)))
