
Function Reference
==================

.. automodule:: synkhronos
   :members: fork, distribute, close, function, broadcast, gather, reduce, all_reduce, all_gather, scatter

.. autoclass:: synkhronos.master.Function
   :special-members:
   :members: get_input_shmems, as_theano, __call__, theano_function, inputs_scatter, collect_modes, reduce_ops, output_to_cpu, name
