
RUN_BOTH = False

import theano
import theano.tensor as T
import numpy as np

if RUN_BOTH:
    import synkhronos as synk
    synk.fork()
else:
    import theano.gpuarray
    theano.gpuarray.use("cuda")

x = T.matrix('x')
y = T.matrix('y')
v = T.vector('v')
s = theano.shared(np.ones([1, 5], dtype='float32'), name='s')

z = T.sum(x.dot(y), axis=0)

if RUN_BOTH:
    f_synk = synk.function([x, y], z, broadcast_inputs=[y])
    g_synk = synk.function([v], updates={s: s + v}, broadcast_inputs=[v])
    synk.distribute()

f_theano = theano.function([x, y], z)
g_theano = theano.function([v], updates={s: s + v})

x_dat = 0.01 * np.ones([1000, 10], dtype='float32')
x_dat1 = x_dat[:400]
x_dat2 = x_dat[400:]
y_dat = np.ones([10, 5], dtype='float32')


r_theano = f_theano(x_dat, y_dat)
print("result of f_theano: ", r_theano)
r_t_1 = f_theano(x_dat1, y_dat)
r_t_2 = f_theano(x_dat2, y_dat)
assert np.allclose(r_theano, r_t_1 + r_t_2)
print("\nbare theano function f: ")
theano.printing.debugprint(f_theano)
if RUN_BOTH:
    r_as_theano = f_synk.as_theano(x_dat, y_dat)
    r_synk = f_synk(x_dat, y_dat)
    r_1 = f_synk.as_theano(x_dat1, y_dat)
    r_2 = f_synk.as_theano(x_dat2, y_dat)
    assert np.allclose(r_theano, r_as_theano)
    assert np.allclose(r_synk, r_theano)
    assert np.allclose(r_1 + r_2, r_theano)
    print("\nsynk-wrapped theano function f: ")
    theano.printing.debugprint(f_synk.theano_function)

print("\nAll tests on functions 'f' passed.\n")

print("s before: ", s.get_value())
g_theano(r_theano)
s_after = s.get_value()
print("s after: ", s_after)

if RUN_BOTH:
    g_synk(r_theano)
    print("s local after synk: ", s.get_value())
    s_gather = synk.gather(shared_vars=s, nd_up=1)
    print("s gathered: ", s_gather)


