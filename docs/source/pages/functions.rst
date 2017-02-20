
Function Reference
==================

.. automodule:: synkhronos
   :members: fork, distribute, close, function, data, broadcast, gather, reduce, all_reduce, all_gather, scatter, make_slices

.. autoclass:: synkhronos.master.Function
   :special-members:
   :members: as_theano, __call__, build_inputs, theano_function, collect_modes, reduce_ops, output_to_cpu, name

.. autoclass:: synkhronos.master.SynkData
   :members: set_value, get_value, set_length, check_input_type, free_memory, dtype, ndim, length, alloc_size
