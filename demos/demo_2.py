"""
Demonstrate interactions with synkhronos.Data objects
"""

import numpy as np
import theano
# import theano.tensor as T
import synkhronos as synk
synk.fork()  # (making data: after fork(), before and/or after distribute())

# Generate some random data sets
x_dat_0 = np.random.randn(5, 4).astype(theano.config.floatX)
x_dat_1 = np.random.randn(5, 4).astype(theano.config.floatX)
x_dat_2 = np.random.randn(2000, 10).astype(theano.config.floatX)
x_dat_3 = np.random.randn(2100, 10).astype(theano.config.floatX)
x_dat_4 = np.random.randn(100, 8).astype(theano.config.floatX)
x_dat_5 = np.random.randn(100, 8).astype("float64")

# Create a Synkhronos data object to be used as data input to functions.
x = synk.data(x_dat_0)

print("\nSome information about x_data...")
print("object: ", x)
print("values:\n", x.data)  # x.data: numpy array, underlying memory: OS-shared
print("\nshape: ", x.shape)
print("length: ", len(x))
print("allocation size (items): ", x.alloc_size)
print("type(x.data): ", type(x.data))
print("\ndir(x): \n", dir(x))

# Reading and writing like a numpy array
rows = x[2:4]
print("x[2:4]:\n", rows)
x[3] = 0
print("\nzero'd a row, new values:\n", x.data)

# Writing new values
x[:] = x_dat_1
print("\nnew values (copied from x_dat_1):\n", x.data)

# Setting a new value, larger shape, same ndim; re-allocates underlying memory
x.set_value(x_dat_2, oversize=1.1)
print("\nnew size (from setting value to x_dat_2): ", x.size)
print("new allocation size (used oversize=1.1): ", x.alloc_size)

# Setting a new value that fits within allocation; no new memory allocation
x.set_value(x_dat_3)
print("\nnew size (from setting value to x_dat_3): ", x.size)
print("new length: ", len(x))
print("same allocation size: ", x.alloc_size)

# Write smaller data and fill the array, without changing memory allocation.
x.set_length(len(x_dat_2))  # (or x.set_shape(..), or x.set_value(..))
x[:] = x_dat_2  # (now easy to assign, e.g. straight from function output)
print("\nnew length (back to x_dat_2):", len(x))

# Shrink the array but see the large memory allocation still exists.
x.set_value(x_dat_4)
print("\nnew shape (x_dat_4): ", x.shape)
print("allocation size unchanged: ", x.alloc_size)

# To shrink the allocation: free_memory() (must use to free in workers)
x.free_memory()
x.set_value(x_dat_4)  # must re-write data to new allocation
print("\nnew allocation size (x_dat_4 after free_memory): ", x.alloc_size)

# Force the right data type.
x.set_value(x_dat_5, force_cast=True)
print("\nx_dat_5 values dtype forced from: ", x_dat_5.dtype, " to: ", x.dtype)
