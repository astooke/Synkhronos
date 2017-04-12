
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

    def __init__(self, ID, theano_function, sliced_function,
                 n_scatter, n_bcast, slc_shareds, update_vars,
                 collect_modes):
        self._ID = ID
        self._f = theano_function
        self._f.trust_input = True
        self._sliced_f = sliced_function
        self._sliced_f.trust_input = True  # NOTE: avoids bug slowdown unpickled
        self._n_scat = n_scatter
        self._n_bcast = n_bcast
        self._slc_shareds = slc_shareds
        self._update_vars = update_vars
        self._collect_modes = collect_modes
        self._n_input = n_scatter + n_bcast
        self._n_output = len(theano_function.outputs)
        self._n_slc_updates = len(sliced_function.outputs) - self._n_output
        self._slc_out_set = list(range(self._n_output, self._n_slc_updates))
        self._full_output_set = list(range(self._n_output))
        self._current_output_set = self._full_output_set
        self._define_collect(collect_modes)
        self._set_reduce_fs(collect_modes, sliced_function.outputs)

    def _get_distro_info(self):
        info = dict(
            ID=self._ID,
            theano_function=self._f,
            sliced_function=self._sliced_f if self._sliced_f is not self._f else None,
            n_scatter=self._n_scat,
            n_bcast=self._n_bcast,
            slc_shareds=self._slc_shareds,
            update_vars=self._update_vars,
            collect_modes=self._collect_modes,
        )
        return info

    def _define_collect(self, collect_modes):
        bare_ops = [m.lstrip("c_") for m in collect_modes]
        self._collect = struct(
            modes=collect_modes,
            nccl=[b == m and b is not None for b, m in zip(bare_ops, collect_modes)],
            ops=bare_ops,
            avgs=["avg" in m for m in collect_modes],
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
        return self._f.get_shared()

    def _run_function(self, num_slices, output_subset):
        my_inputs = scatterer.get_my_inputs(self._n_scat, self._n_bcast)
        my_results = \
            self._run_sliced_f(my_inputs, num_slices, output_subset) \
            if num_slices > 1 else \
            self._f(*my_inputs, output_subset=output_subset)
        return my_results

    def _run_sliced_f(self, my_inputs, num_slices, output_subset):
        accum_rs = None
        for sliced_inputs in self._slice_inputs(my_inputs, num_slices):
            sliced_rs = self._sliced_f(*sliced_inputs, output_subset=output_subset)
            accum_rs = self._accum_my_results(accum_rs, sliced_rs)
        if any(self._collect.avgs):
            self._avg_my_results(accum_rs, num_slices)
        my_results = accum_rs[:len(self._current_output_set)]
        my_updates = accum_rs[len(self._current_output_set):]
        for var, update in zip(self._update_vars, my_updates):
            var.container.data = update
        return my_results  # (always a list)

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

    def _slice_inputs(self, inputs, num_slices):
        length = None
        if self._n_scat > 0:
            length = len(inputs[0])
        if self._slc_shareds:
            s_lengths = [s.container.data.shape[0] for s in self._slc_shareds]
            s_len = s_lengths[0]
            if s_lengths.count(s_len) != len(s_lengths):
                raise ValueError("Had different lengths for sliceable "
                    "shareds: {}, {}".format(self._slc_shareds, s_lengths))
            if length is not None and s_len != length:
                raise ValueError("Had different lengths for sliceable shareds "
                    "{} vs sliceable inputs {}".format(s_len, length))
            if length is None:
                length = s_len
        edges = np.linspace(0, length, num_slices + 1, dtype='int64')
        for slc in [slice(*edges[i:i + 2]) for i in range(num_slices)]:
            sliced_inputs = list()
            for inpt in inputs[:self._n_scat]:
                sliced_inputs.append(inpt[slc])
            for inpt in inputs[self._n_scat:]:
                sliced_inputs.append(inpt)
            if self._slc_shareds:
                sliced_inputs += [np.array(slc.start), np.array(slc.stop)]
            yield tuple(sliced_inputs)


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
        if self._n_scat > 0:
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

    @property
    def name(self):
        return self._f.name

    @property
    def output_modes(self):
        return self._output_modes

    @property
    def update_modes(self):
        return self._update_modes

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

    def _share_input_data(self, synk_inputs, batch):
        if self._n_input > 0:
            check_synk_inputs(synk_inputs, self._input_vars)
            scatterer.assign_inputs(synk_inputs, batch, self._n_scat)
        # TODO: handle batching for sliceable shared variables

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

    def __call__(self, *args, output_subset=None, batch=None, num_slices=1,
                **kwargs):
        """ Callable as in Theano function.

        When called, Synkhronos functions:

            1. Share input data,
            2. Signal to workers to start and what to do,
            3. Call the local theano function on assigned data subset,
            4. Collect results from workers and return it.

        Theano function keyword argument ``output_subset`` is supported.

        Args:
            *args (data): Normal data inputs to Theano function
            **kwargs (data): Normal data inputs to Theano function

        Raises:
            RuntimeError: If not distributed or if synkhronos closed.
        """
        exct.check_active()
        ordered_inputs = self._order_inputs(args, kwargs)
        self._share_input_data(ordered_inputs, batch)
        self._update_f_info(num_slices, output_subset)
        exct.launch(exct.FUNCTION, self._ID)
        my_results = self._run_function(num_slices, output_subset)
        outputs = self._collect_results(my_results)
        exct.join()
        if not self._return_list and len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def as_theano(self, *args, **kwargs):
        """Call the function in the master process only, as normal Theano.

        This method will return outputs to the CPU if they were originally
        requested there, unlike using ``function.theano_function()``, which is
        built to hold all outputs on the GPU.

        Args:
            *args (data): Normal data inputs to the Theano function
            **kwargs (data): Normal data inputs to the Theano function
        """
        # FIXME:  possibly out of date.
        results = self._f(*args, **kwargs)
        if not isinstance(results, list):
            results = [results]
        o_subset = kwargs.pop("output_subset", None)
        output_set = range(self._n_output) if o_subset is None else o_subset
        for idx_r, idx_o in enumerate(output_set):
            if self._outputs.to_cpu[idx_o]:
                results[idx_r] = np.asarray(results[idx_r])
        if len(results) == 1:
            results = results[0]
        return results

    def build_inputs(self, *args, force_cast=False, oversize=1, minibatch=False,
                     **kwargs):
        """ convenience method which internally calls synkhronos.data() for
        each input variable associated with this function; provide data inputs
        as if calling the Theano function.
        # TODO: move force_cast and oversize to function signature?
        """
        # FIXME: possibly out of date
        ordered_inputs = self._order_inputs(args, kwargs)
        synk_datas = list()
        for var, inpt in zip(self._input_vars, ordered_inputs):
            synk_data = data(var=var,
                             value=inpt,
                             minibatch=minibatch,
                             force_cast=force_cast,
                             oversize=oversize,
                             name=var.name,
                             )
            synk_datas.append(synk_data)
        if len(synk_datas) == 1:
            return synk_datas[0]
        else:
            return tuple(synk_datas)


