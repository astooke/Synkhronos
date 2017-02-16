
Function Reference
==================

.. automodule:: synkhronos
   :members: fork, distribute, close, function, get_shmems, set_shmems, free_shmems, broadcast, gather, reduce, all_reduce, all_gather, scatter, make_slices

.. autoclass:: synkhronos.master.Function
   :special-members:
   :members: get_input_shmems, set_input_shmems, as_theano, __call__, theano_function, collect_modes, reduce_ops, output_to_cpu, name
