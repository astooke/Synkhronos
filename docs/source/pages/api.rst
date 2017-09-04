
API Reference
=============

.. automodule:: synkhronos
   :members: fork, close, function, distribute

.. autoclass:: synkhronos.function_module.Function
   :members: __call__, build_inputs, as_theano, output_modes, update_modes, name

.. automodule:: synkhronos
   :members: data

.. autoclass:: synkhronos.data_module.Data
   :members: set_value, data, __getitem__, __setitem__, __len__, set_length, set_shape, condition_data, free_memory, alloc_size, dtype, ndim, shape, size, minibatch, name

.. automodule:: synkhronos
   :members: broadcast, scatter, gather, all_gather, reduce, all_reduce, set_value, get_value, get_lengths, get_shapes
