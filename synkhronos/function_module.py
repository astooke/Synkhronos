
import numpy as np

from .util import struct
from .data_builder import data
from .data_module import Data
from .scatterer import scatterer
from .reducers import reducers
from . import comm
from . import exct


sync = None


###############################################################################
#                                                                             #
#                       Base Synk Function                                    #
#                                                                             #
###############################################################################


class BaseFunction(object):

    def __init__(self, ID, functions, n_scatter, n_bcast, slc_shareds,
                 update_vars, collect_modes):
        self._ID = ID
        self._functions = functions
        for f in self._functions.values():
            f.trust_input = True  # NOTE: avoids bug slowdown unpickled
        self._n_scat = n_scatter
        self._n_bcast = n_bcast
        self._slc_shareds = slc_shareds
        self._n_slc_sh = 0 if slc_shareds is None else len(slc_shareds)
        self._update_vars = update_vars
        self._collect_modes = collect_modes
        self._n_input = n_scatter + n_bcast
        self._n_output = len(functions.theano_function.outputs)
        sliced_f = functions.sliced if hasattr(functions, "sliced") else functions.f
        self._n_slc_updates = len(sliced_f.outputs) - self._n_output
        self._slc_out_set = list(range(self._n_output, self._n_slc_updates))
        self._full_output_set = list(range(self._n_output))
        self._current_output_set = self._full_output_set
        self._define_collect(collect_modes)
        self._set_reduce_fs(collect_modes, sliced_f.outputs)

    def _get_distro_info(self):
        info = dict(
            ID=self._ID,
            functions=self._functions,
            n_scatter=self._n_scat,
            n_bcast=self._n_bcast,
            slc_shareds=self._slc_shareds,
            update_vars=self._update_vars,
            collect_modes=self._collect_modes,
        )
        return info

    def _define_collect(self, collect_modes):
        bare_ops = [m.lstrip("c_") if m is not None else m for m in collect_modes]
        self._collect = struct(
            modes=collect_modes,
            nccl=[b == m and b is not None for b, m in zip(bare_ops, collect_modes)],
            ops=bare_ops,
            avgs=[False if m is None else "avg" in m for m in collect_modes],
        )

    def _set_reduce_fs(self, collect_modes, outputs):
        reduce_fs = list()
        avg_fs = list()
        for mode, out in zip(collect_modes, outputs):
            reduce_fs.append(reducers.get_reduce_f(out.variable, mode))
            if mode is None or "avg" not in mode:
                avg_f = None
            else:
                avg_f = reducers.get_avg_f(out.variable)
            avg_fs.append(avg_f)
        self._reduce_fs = reduce_fs
        self._avg_fs = avg_fs

    def get_shared(self):
        return self._functions.f.get_shared()

    def _check_batch_s(self, batch_s, scat_inputs):
        if not self._slc_shareds:
            return
        lengths = [s.container.data.shape[0] for s in self._slc_shareds]
        if batch_s is None:
            if not lengths.count(lengths[0]) == len(lengths):
                raise ValueError("No batch_s input but had different length "
                    "sliceable shareds. Vars: {}, Lengths: {}".format(
                        self._slc_shareds, lengths))
        elif isinstance(batch_s, np.ndarray):
            length = batch_s.size
            if batch_s.max() > min(lengths):
                raise ValueError("Had batch_s explicit index requested, {}, "
                    "out of range of at least one sliceable shared variable. "
                    "Vars: {}, Lengths: {}".format(batch_s.max(),
                        self._slc_shareds, lengths))
        elif isinstance(batch_s, slice):
            length = batch_s.stop - batch_s.start
            if batch_s.start > min(lengths):
                raise ValueError("Had batch_s slice.start requested, {}, "
                    "higher than the length of at least one sliceable shared "
                    "variable. Vars: {}, Lengths: {}".format(batch_s.start,
                        self._slc_shareds, lengths))
            if batch_s.stop > min(lengths):
                raise ValueError("Had batch_s slice.stop requested, {}, higher "
                    "than the length of at least one sliceable shared variable. "
                    " Vars: {}, Lengths: {}".format(batch_s.end,
                        self._slc_shareds, lengths))
        if scat_inputs:
            if length != len(scat_inputs[0]):
                raise ValueError("Had both scattered inputs and sliceable "
                    "shareds, but their requested lengths (after applying batch "
                    "and batch_s) did not match. scattered length: {}, shared "
                    "length: {}".format(len(scat_inputs[0]), length))

    # def _run_function_old(self, num_slices, output_subset):
    #     my_inputs = scatterer.get_my_inputs(self._n_scat, self._n_bcast)
    #     my_results = \
    #         self._run_sliced_f(my_inputs, num_slices, output_subset) \
    #         if num_slices > 1 else \
    #         self._f(*my_inputs, output_subset=output_subset)
    #     return my_results

    def _run_function(self, num_slices, output_subset):
        my_inputs = scatterer.get_my_inputs(self._n_scat, self._n_bcast)
        batch_s = scatterer.get_my_batch_s(self._n_slc_sh)
        self._check_batch_s(batch_s, my_inputs[:self._n_scat])
        if num_slices > 1:
            my_results = self._run_sliced_f(my_inputs, batch_s, num_slices, output_subset)
        elif batch_s is None:
            my_results = \
                self._functions.f(*my_inputs, output_subset=output_subset)
        elif isinstance(batch_s, slice):
            start = np.array(batch_s.start, dtype='int64')
            stop = np.array(batch_s.stop, dtype='int64')
            my_results = \
                self._functions.slc_in(*my_inputs, start, stop, output_subset=output_subset)
        elif isinstance(batch_s, np.ndarray):
            my_results = \
                self._functions.lst_in(*my_inputs, batch_s, output_subset=output_subset)
        else:
            raise RuntimeError("Unrecognized batch_s type: {}".format(type(batch_s)))
        return my_results

    def _run_sliced_f(self, my_inputs, batch_s, num_slices, output_subset):
        if batch_s is None:
            if hasattr(self._functions, "sliced"):
                sliced_f = self._functions.sliced
            elif hasattr(self._functions, "sliced_slc_in"):
                sliced_f = self._functions.sliced_slc_in
            else:
                sliced_f = self._functions.f
        elif isinstance(batch_s, slice):
            sliced_f = self._functions.sliced_slc_in
        elif isinstance(batch_s, np.ndarray):
            sliced_f = self._functions.sliced_lst_in
        else:
            raise RuntimeError("Unrecognized batch_s type: {}".format(type(batch_s)))
        accum_rs = None
        for sliced_inputs in self._slice_inputs(my_inputs, batch_s, num_slices):
            sliced_rs = sliced_f(*sliced_inputs, output_subset=output_subset)
            accum_rs = self._accum_my_results(accum_rs, sliced_rs)
        if any(self._collect.avgs):
            self._avg_my_results(accum_rs, num_slices)
        my_results = accum_rs[:len(self._current_output_set)]
        my_updates = accum_rs[len(self._current_output_set):]
        for var, update in zip(self._update_vars, my_updates):
            var.container.data = update
        return my_results  # (always a list)

    # def _run_sliced_f_old(self, my_inputs, num_slices, output_subset):
    #     accum_rs = None
    #     for sliced_inputs in self._slice_inputs(my_inputs, num_slices):
    #         sliced_rs = self._sliced_f(*sliced_inputs, output_subset=output_subset)
    #         accum_rs = self._accum_my_results(accum_rs, sliced_rs)
    #     if any(self._collect.avgs):
    #         self._avg_my_results(accum_rs, num_slices)
    #     my_results = accum_rs[:len(self._current_output_set)]
    #     my_updates = accum_rs[len(self._current_output_set):]
    #     for var, update in zip(self._update_vars, my_updates):
    #         var.container.data = update
    #     return my_results  # (always a list)

    def _accum_my_results(self, accum_rs, sliced_rs):
        if accum_rs is None:
            return sliced_rs
        reduce_fs = [self._reduce_fs[i]
            for i in self._current_output_set + self._slc_out_set]
        for i, (f, a_r, s_r) in enumerate(zip(reduce_fs, accum_rs, sliced_rs)):
            accum_rs[i] = f(a_r, s_r)
        return accum_rs

    def _avg_my_results(self, accum_rs, num_slices):
        inv_num = 1 / num_slices
        avg_fs = [self._avg_fs[i]
            for i in self._current_output_set + self._slc_out_set]
        for i, (f, r) in enumerate(zip(avg_fs, accum_rs)):
            if f is not None:
                accum_rs[i] = f(r, inv_num)
        return accum_rs

    def _slice_inputs(self, inputs, batch_s, num_slices):
        length = None
        if self._n_scat > 0:
            length = len(inputs[0])  # length of scattered inputs checked previously
        if self._slc_shareds:
            if batch_s is None:
                s_length = self._slc_shareds[0].container.data.shape[0]  # length checked previously
            elif isinstance(batch_s, slice):
                s_length = batch_s.stop - batch_s.start
            elif isinstance(batch_s, np.ndarray):
                s_length = batch_s.size
            if length is None:
                length = s_length
            else:
                assert s_length == length  # (should already be enforced)
        edges = np.linspace(0, length, num_slices + 1, dtype='int64')
        for slc in [slice(*edges[i:i + 2]) for i in range(num_slices)]:
            sliced_inputs = list()
            for inpt in inputs[:self._n_scat]:
                sliced_inputs.append(inpt[slc])
            for inpt in inputs[self._n_scat:]:
                sliced_inputs.append(inpt)
            if self._slc_shareds:
                if batch_s is None:
                    start = np.array(slc.start)
                    stop = np.array(slc.stop)
                    sliced_inputs += [start, stop]
                elif isinstance(batch_s, np.ndarray):
                    sliced_inputs.append(batch_s[slc])
                elif isinstance(batch_s, slice):
                    start = np.array(batch_s.start + slc.start, dtype='int64')
                    stop = np.array(batch_s.start + slc.stop, dtype='int64')
                    sliced_inputs += [start, stop]
            yield tuple(sliced_inputs)

    # def _slice_inputs_old(self, inputs, num_slices):
    #     length = None
    #     if self._n_scat > 0:
    #         length = len(inputs[0])
    #     if self._slc_shareds:
    #         s_lengths = [s.container.data.shape[0] for s in self._slc_shareds]
    #         s_len = s_lengths[0]
    #         if s_lengths.count(s_len) != len(s_lengths):
    #             raise ValueError("Had different lengths for sliceable "
    #                 "shareds: {}, {}".format(self._slc_shareds, s_lengths))
    #         if length is not None and s_len != length:
    #             raise ValueError("Had different lengths for sliceable shareds "
    #                 "{} vs sliceable inputs {}".format(s_len, length))
    #         if length is None:
    #             length = s_len
    #     edges = np.linspace(0, length, num_slices + 1, dtype='int64')
    #     for slc in [slice(*edges[i:i + 2]) for i in range(num_slices)]:
    #         sliced_inputs = list()
    #         for inpt in inputs[:self._n_scat]:
    #             sliced_inputs.append(inpt[slc])
    #         for inpt in inputs[self._n_scat:]:
    #             sliced_inputs.append(inpt)
    #         if self._slc_shareds:
    #             sliced_inputs += [np.array(slc.start), np.array(slc.stop)]
    #         yield tuple(sliced_inputs)


###############################################################################
#                                                                             #
#                       Worker Synk Function                                  #
#                                                                             #
###############################################################################


class WorkerFunction(BaseFunction):

    def __call__(self):
        """
        1. Gather the right inputs from mp shared values.
        2. Execute local theano function on those inputs.
        3. Send results back to master.
        """
        num_slices, output_subset = self.receive_f_info()
        if self._n_scat > 0 or self._n_slc_sh > 0:
            scatterer.check_idxs_alloc()
        my_results = self._run_function(num_slices, output_subset)
        self.send_results(my_results)

    def receive_f_info(self):
        num_slices = sync.n_slices.value
        if self._n_output == 0:
            return num_slices, None
        if sync.is_new_subset.value:
            o_set = [i for i in range(self._n_output)
                     if sync.output_subset[i]]
            self._current_output_set = o_set
        output_subset = None \
            if len(self._current_output_set) == self._n_output else \
            self._current_output_set
        return num_slices, output_subset

    def send_results(self, my_results):
        for i, r in zip(self._current_output_set, my_results):
            nccl = self._collect.nccl[i]
            op = self._collect.ops[i]
            send(r, op, nccl)


def send(arr, op, nccl=True):
    if op is None:
        return
    if nccl and comm.gpu is not None:
        comm.gpu.send(arr, op)
    else:
        comm.cpu.send(arr)


###############################################################################
#                                                                             #
#                    Helper methods for Master Synk Function                  #
#                                                                             #
###############################################################################


class FunctionHelpers(BaseFunction):

    def __init__(self, inputs, bcast_inputs, to_cpu, return_list=True,
                **kwargs):
        super().__init__(n_scatter=len(inputs), n_bcast=len(bcast_inputs), **kwargs)
        self._input_orderer = build_input_orderer(inputs + bcast_inputs)
        self._input_vars = inputs + bcast_inputs
        self._to_cpu = to_cpu
        self._return_list = return_list
        self._prev_output_subset = None

    def _order_inputs(self, args, kwargs):
        """ Combine args and kwargs into one list of input args. """
        n_args = len(args) + len(kwargs)
        if n_args != self._n_input:
            raise TypeError("Incorrect number of data inputs to function.")
        if n_args == 0:
            return ()
        ordered_inputs = list(args) + [None] * len(kwargs)
        for var, arg in kwargs.items():
            idx = self._input_orderer.get(var, None)
            if idx is None:
                raise ValueError("Unrecognized keyword var or name: ", var)
            if ordered_inputs[idx] is not None:
                raise ValueError("Redundant input for variable: ", var)
            ordered_inputs[idx] = arg
        return tuple(ordered_inputs)

    def _share_input_data(self, synk_inputs, batch, batch_s):
        if self._n_input > 0:
            check_synk_inputs(synk_inputs, self._input_vars)
            scatterer.assign_inputs(synk_inputs, batch, self._n_scat)
        if self._n_slc_sh > 0:
            scatterer.assign_batch_s(batch_s)
        elif batch_s is not None:
            raise TypeError("Had param 'batch_s', but no sliceable shareds.")

    def _update_f_info(self, num_slices, output_subset):
        if num_slices < 1 or int(num_slices) != num_slices:
            raise ValueError("Invalid number of slices: ", num_slices)
        if self._n_scat == 0 and not self._slc_shareds and num_slices > 1:
            raise ValueError("Requested num_slices > 1 but nothing to slice!")
        sync.n_slices.value = int(num_slices)
        is_new_subset = output_subset != self._prev_output_subset
        if is_new_subset:
            if output_subset is None:
                self._current_output_set = self._full_current_output_set
            else:
                output_subset = check_output_subset(self._n_output, output_subset)
                self._current_output_set = output_subset
            for i in range(self._n_output):
                sync.output_subset[i] = i in self._current_output_set
            self._prev_output_subset = output_subset
        sync.is_new_subset.value = is_new_subset

    def _collect_results(self, my_results):
        results = list()
        for o, r in zip(self._current_output_set, my_results):
            nccl = self._collect.nccl[o]
            op = self._collect.ops[o]
            results.append(collect(r, op, nccl))
        for i, o in enumerate(self._current_output_set):
            if self._to_cpu[o]:
                results[i] = np.asarray(results[i])
        return results


###############################################################################
#                                                                             #
#        For running or initializing Synk master Functions                    #
#                                                                             #
###############################################################################


def check_output_subset(n_outputs, output_subset):
    if not isinstance(output_subset, list):
        raise TypeError("Optional param 'output_subset' must be a "
            "list of ints.")
    output_subset = list(set(output_subset))
    for idx in output_subset:
        if not isinstance(idx, int):
            raise TypeError("Optional param 'output_subset' must a "
                "list of ints.")
        if idx < 0 or idx > n_outputs - 1:
            raise ValueError("Output subset entry out of range: ", idx)
    return output_subset


def build_input_orderer(inputs):
    input_orderer = dict()
    for idx, var in enumerate(inputs):
        input_orderer[var] = idx
        if var.name is not None:
            input_orderer[var.name] = idx
    return input_orderer


def check_synk_inputs(synk_datas, vars):
    for idx, (s_data, var) in enumerate(zip(synk_datas, vars)):
        if not isinstance(s_data, Data):
            raise TypeError("All function inputs must be of Synkhronos type Data.")
        if s_data.dtype != var.dtype:
            raise TypeError("Incorrect input dtype for position {}; expected: "
                "{}, received: {}.".format(idx, var.dtype, s_data.dtype))
        if s_data.ndim != var.ndim:
            raise TypeError("Incorrect input dimensions for position {}; "
                "expected: {}, received: {}.".format(idx, var.ndim, s_data.ndim))


def collect(arr, op, nccl=True):
    if op is None:
        return arr
    if nccl and comm.gpu is not None:
        return comm.gpu.collect(arr, op)
    else:
        return comm.cpu.collect(arr, op)


###############################################################################
#                                                                             #
#                   API for (Master) Functions                                #
#                                                                             #
###############################################################################


class Function(FunctionHelpers):
    """ Class of instances returned by ``synkhronos.function()``.  """

    def __call__(self, *args, output_subset=None, batch=None, batch_s=None,
                 num_slices=1, **kwargs):
        """ Callable as in Theano function.

        When called, a Synkhronos function:

            1. Assigns input data evenly across all GPUs,
            2. Signals to workers to start and which function to call,
            3. Calls the underlying Theano function on assigned data subset,
            4. Collect results from workers and returns them.

        Args:
            *args (Data): Normal data inputs to Theano function
            output_subset: as in Theano
            batch: indexes to select from scattering input data (see notes)
            batch_s: indexes to select from scattered implicit inputs (see notes)
            num_slices (int): call the function over this many slices of the
                selected, scattered data and accumulate results (avoid
                out-of-memory)
            **kwargs (Data): Normal data inputs to Theano function

        Batching:
            The kwarg ``batch`` can be of types: (int, slice, list (of ints),
            numpy array (1-d, int)). It applies *before* scattering, to the
            whole input data set.  If type int, this acts as data[:int].

            The kwarg ``batch_s`` can be of type (slice, list (of ints), numpy
            array (1-d, int)) or a list of all the same type (one of those
            three), with one entry for each GPU. It applies *after* scattering,
            to data already residing on the GPU. If only one of the above types
            is provided, rather than a list of them, it is used in all GPUs.

            In both ``batch`` and ``batch_s``, full slice types are not
            supported; start and stop fields must be ints, step None.

        Slicing:
            Function slicing by the ``num_slices`` kwarg applies within each worker,
            after individual worker data assignment.  Results are accumulated
            within each worker and reduced only once at the end.  Likewise, any
            updates are computed and accumulated using the original variable
            values, and the updates are applied only once at the end.

        Raises:
            RuntimeError: If not distributed or if synkhronos closed.
        """
        exct.check_active()
        ordered_inputs = self._order_inputs(args, kwargs)
        self._share_input_data(ordered_inputs, batch, batch_s)
        self._update_f_info(num_slices, output_subset)
        exct.launch(exct.FUNCTION, self._ID)
        my_results = self._run_function(num_slices, output_subset)
        outputs = self._collect_results(my_results)
        exct.join()
        if not self._return_list and len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    @property
    def name(self):
        """As in Theano functions."""
        return self._functions.theano_function.name

    @property
    def output_modes(self):
        """Returns the reduce operations used to collect function outputs."""
        return self._output_modes

    @property
    def update_modes(self):
        """Returns the reduce operations used to accumulate updates (only when
        slicing)"""
        return self._update_modes

    def as_theano(self, *args, **kwargs):
        """Call the function in the master process only, as normal Theano.

        Args:
            *args (data): Normal inputs to the Theano function
            **kwargs (data): Normal inputs to the Theano function
        """
        # results = self._f(*args, **kwargs)
        # if not isinstance(results, list):
        #     results = [results]
        # o_subset = kwargs.pop("output_subset", None)
        # output_set = range(self._n_output) if o_subset is None else o_subset
        # for idx_r, idx_o in enumerate(output_set):
        #     if self._outputs.to_cpu[idx_o]:
        #         results[idx_r] = np.asarray(results[idx_r])
        # if len(results) == 1:
        #     results = results[0]
        # return results

        return self._functions.theano_function(*args, **kwargs)

    def build_inputs(self, *args, force_cast=False, oversize=1., minibatch=False,
                     **kwargs):
        """Convenience method which internally calls ``synkhronos.data()`` for
        each input variable associated with this function.  Provide data inputs
        as if calling the Theano function.

        Args:
            *args: data inputs
            force_cast (bool, optional): see ``synkhronos.data()``
            oversize (float [1,2], optional): see ``synkhronos.data()``
            minibatch (bool, optional): see ``synkhronos.data()``
            **kwargs: data inputs

        The kwargs ``force_cast``, ``oversize``, and ``minibatch`` are passed
        to all calls to ``synkhronos.data()``

        Returns:
            synkhronos.data_module.Data: data object for function input.
        """
        ordered_inputs = self._order_inputs(args, kwargs)
        if not isinstance(minibatch, list):
            minibatch = [minibatch] * len(ordered_inputs)
        assert len(minibatch) == len(ordered_inputs)
        synk_datas = list()
        for var, inpt, mb in zip(self._input_vars, ordered_inputs, minibatch):
            synk_data = data(var=var,
                             value=inpt,
                             minibatch=mb,
                             force_cast=force_cast,
                             oversize=oversize,
                             name=var.name,
                             )
            synk_datas.append(synk_data)
        if len(synk_datas) == 1:
            return synk_datas[0]
        else:
            return tuple(synk_datas)
