
import theano
import theano.tensor as T
import pickle
from collections import OrderedDict
import time

from .function_module import Function, WorkerFunction
from .collectives import shareds_registry
from . import exct
from .util import struct, PKL_FILE
# import ipdb
sync = None

synk_functions = list()


###############################################################################
#                                                                             #
#                       API for building Functions                            #
#                                                                             #
###############################################################################


def function(inputs, outputs=None, bcast_inputs=None, updates=None,
             givens=None, sliceable_shareds=None, **kwargs):
    """Replacement for ``theano.function()``, with a similar interface.  Builds
    underlying Theano functions, including support for function slicing.

    Args:
        inputs: as in Theano, to be scattered among workers
        outputs: as in Theano, with option to specify reduce operation (see
            notes below)
        bcast_inputs:  as inputs in Theano, to be broadcast to all workers
        updates: as in Theano, with option to specify reduct operation (see
            notes below)
        givens: as in Theano
        sliceable_shareds: any implicit inputs (Theano shared variables) acting
            as data-parallel data (i.e. to be subjected to the kwarg ``batch_s``
            and /or to function slicing) must be listed here
        **kwargs: passed on to all internal calls to ``theano.function()``

    Reduce Operations:
      Outputs: May be specified simply as Theano tensor variables, as in normal
      Theano, or as two-tuples, as in (var, reduce-op), where reduce-op can be:
      "avg", "sum", "max", "min", "prod", or None.  Default is "avg".

      Updates: May be specified as a list of two-tuples, as in normal Theano, or
      may include triples, as in (var, update, reduce-op).  Unlike for outputs,
      the reduce-op here applies only when using function slicing.  Every slice
      is computed using the original values, and the update is accumulated over
      the slices.  At the end of the function call, all updates are applied only
      locally, within each worker.  This provides clear control to user over
      when to communicate.

    Returns:
        sykhronos.Function: callable object, replacing a theano.Function

    Raises:
        RuntimeError: If Sykhronos not yet forked, or if already distributed
        TypeError: If incorrect format for arguments.
        ValueError: If entry in ``sliceable_shareds`` is not used in function,
            or for invalid reduce operation requested.
    """
    if not exct.state.forked:
        raise RuntimeError("Must fork before making functions for GPU.")
    if exct.state.distributed:
        raise RuntimeError("Cannot make new functions after distributing (for now).")

    if not isinstance(inputs, list):
        raise TypeError("Input 'inputs' must be list.")
    bcast_inputs = [] if bcast_inputs is None else bcast_inputs
    if not isinstance(bcast_inputs, list):
        raise TypeError("Input 'bcast_inputs' must be list if not None.")

    reg_outputs, gpu_outputs, to_cpu, output_modes = process_outputs(outputs)
    updates, update_vars, sliced_update_outs, update_modes = process_updates(updates)

    theano_function = theano.function(
        inputs=inputs + bcast_inputs,
        outputs=reg_outputs,
        updates=updates,
        givens=givens,
        **kwargs
    )

    functions = struct(theano_function=theano_function)
    functions["f"] = theano.function(
        inputs=inputs + bcast_inputs,
        outputs=gpu_outputs,
        updates=updates,
        givens=givens,
        **kwargs
    )

    if not updates and not sliceable_shareds:
        functions["sliced"] = theano.function(
            inputs=inputs + bcast_inputs,
            outputs=gpu_outputs + sliced_update_outs,
            givens=givens,
            **kwargs
        )

    if sliceable_shareds:
        slc_givens, lst_givens, slc_inputs, lst_input = \
            process_givens(givens, sliceable_shareds, theano_function.get_shared())

        functions["slc_in"] = theano.function(
            inputs=inputs + bcast_inputs + slc_inputs,
            outputs=gpu_outputs,
            updates=updates,
            givens=slc_givens,
            **kwargs
        )
        functions["lst_in"] = theano.function(
            inputs=inputs + bcast_inputs + [lst_input],
            outputs=gpu_outputs,
            updates=updates,
            givens=lst_givens,
            **kwargs
        )
        functions["sliced_slc_in"] = theano.function(
            inputs=inputs + bcast_inputs + slc_inputs,
            outputs=gpu_outputs + sliced_update_outs,
            givens=slc_givens,
            **kwargs
        )
        functions["sliced_lst_in"] = theano.function(
            inputs=inputs + bcast_inputs + [lst_input],
            outputs=gpu_outputs + sliced_update_outs,
            givens=lst_givens,
            **kwargs
        )

    synk_function = Function(ID=len(synk_functions),
                             functions=functions,
                             inputs=inputs,
                             bcast_inputs=bcast_inputs,
                             slc_shareds=sliceable_shareds,
                             update_vars=update_vars,
                             to_cpu=to_cpu,
                             collect_modes=output_modes + update_modes,
                             return_list=isinstance(outputs, list),
                             )
    synk_functions.append(synk_function)
    shareds_registry.register_func(theano_function)
    return synk_function


def distribute():
    """Sets up theano functions from master on workers.

    Pickles all theano functions built with this package (i.e. using
    ``synkhronos.function()``) into one file, which workers unpickle.  Theano's
    behavior is to include all shared variable values in the file.  Workers are
    aware of correspondences among input and shared variables used across
    multiple functions.

    In the future, distribution will happen automatically, lazily at the time of
    any function call when it is necessary.  It will remain optional for the
    user to call, as it may be time-consuming.

    The pickle file is automatically deleted by a worker.

    Raises:
        RuntimeError: If not yet forked or if already distributed.
    """
    if not exct.state.forked:
        raise RuntimeError("Need to fork before distributing functions.")

    print("Synkhronos distributing functions...")
    t_start = time.time()
    distribution = [sf._get_distro_info() for sf in synk_functions]
    with open(PKL_FILE, "wb") as f:
        pickle.dump(distribution, f, pickle.HIGHEST_PROTOCOL)
    exct.launch(exct.DISTRIBUTE)
    exct.join()
    print("...distribution complete ({:.0f} s).".format(time.time() - t_start))
    exct.state.distributed = True


###############################################################################
#                                                                             #
#                              Helpers                                        #
#                                                                             #
###############################################################################


COLLECT_MODES = ["avg", "sum", "prod", "min", "max", "gather",
    "c_avg", "c_sum", "c_prod", "c_min", "c_max", "c_gather", None]


def process_outputs(outputs):
    if outputs is None:
        return None, [], [], []
    output_vars = list()
    output_modes = list()
    from theano.gpuarray.type import GpuArrayVariable
    len_err = TypeError("Output tuples must be length 2: (var, collect_mode).")
    if isinstance(outputs, tuple):
        if len(outputs) != 2: raise len_err
        output_vars.append(outputs[0])
        output_modes.append(outputs[1])
    elif isinstance(outputs, list):
        for o in outputs:
            if isinstance(o, tuple):
                if len(o) != 2: raise len_err
                output_vars.append(o[0])
                output_modes.append(o[1])
            else:
                output_vars.append(o)
                output_modes.append("avg")  # (default)
    else:
        output_vars.append(outputs)
        output_modes.append("avg")
    check_collect_modes(output_modes)
    to_cpu = [not isinstance(var, GpuArrayVariable) for var in output_vars]
    gpu_vars = [var.transfer(None) for var in output_vars]
    return output_vars, gpu_vars, to_cpu, output_modes


def process_updates(updates):
    if updates is None:
        return None, [], [], []
    in_err = TypeError("Input 'updates' should be a list of tuples: "
            "(var, new_value [, slice_mode])")
    reg_updates = list()
    update_vars = list()
    update_gpu_outs = list()
    update_modes = list()
    if isinstance(updates, OrderedDict):  # (legacy only)
        for k, v in updates.items():
            reg_updates.append((k, v))
            update_vars.append(k)
            update_gpu_outs.append(v.transfer(None))
            update_modes.append("avg")
    else:
        if not isinstance(updates, list): raise in_err
        for u in updates:
            if not isinstance(u, tuple) or len(u) not in (2, 3): raise in_err
            reg_updates.append((u[0], u[1]))
            update_vars.append(u[0])
            update_gpu_outs.append(u[1].transfer(None))
            update_modes.append(u[2] if len(u) == 3 else "avg")
    check_collect_modes(update_modes)
    return reg_updates, update_vars, update_gpu_outs, update_modes


def process_givens(givens, sliceable_shareds, f_shareds):
    if givens is None:
        givens = list()
    if isinstance(givens, (list, tuple)):
        givens = {g[0]: g[1] for g in givens}
    if not isinstance(sliceable_shareds, list):
        raise TypeError("Optional param `sliceable_shareds` must be list.")
    for var in sliceable_shareds:
        if var not in f_shareds:
            raise ValueError("At least one of sliceable_shareds not in "
                "function's shareds: sliceable: {}, function's: {}".format(
                    sliceable_shareds, f_shareds))
    start_input = T.lscalar('start')
    stop_input = T.lscalar('stop')
    slc_inputs = [start_input, stop_input]
    lst_input = T.lvector('lst')
    slc_givens = dict()
    lst_givens = dict()
    remaining_ss = list(sliceable_shareds)
    for k, v in givens.items():
        if v in sliceable_shareds:
            slc_givens[k] = v[start_input:stop_input].transfer(None)
            lst_givens[k] = v[lst_input].transfer(None)  # NOTE: needed on gpu subtensor, probably just theano bug
            if v in remaining_ss:
                remaining_ss.pop(remaining_ss.index(v))
        else:
            slc_givens[k] = v
            lst_givens[k] = v
    for var in remaining_ss:
        slc_givens[var] = var[start_input:stop_input].transfer(None)
        lst_givens[var] = var[lst_input].transfer(None)
    # FIXME:  might not replace everywhere var is used, for instance if it is
    # used in a given but is already an ancestor to another part of the graph,
    # only the given will have the var replaced.
    return slc_givens, lst_givens, slc_inputs, lst_input


# def process_givens_old(givens, sliced_shareds):
#     if sliced_shareds is None:
#         return givens, None, [], []
#     import theano.tensor as T
#     giv_err = TypeError("If using 'sliced_shareds', givens must be list of 2-tuples.")
#     s_err = TypeError("Input 'sliced_shareds' must be list, elements are "
#         "individual shared variables or 2-tuples: (var, given_var)")
#     if givens is None: givens = list()
#     if not isinstance(givens, list):
#         raise giv_err
#     for g in givens:
#         if not isinstance(g, tuple) or len(g) != 2:
#             raise giv_err
#     if not isinstance(sliced_shareds, list): raise s_err
#     start = T.lscalar()
#     end = T.lscalar()
#     slc_givens = list()
#     slc_shareds = list()
#     for ss in sliced_shareds:
#         if isinstance(ss, tuple):
#             if len(ss) != 2: raise s_err
#             givens.append(ss)
#             slc_givens.append((ss[0], ss[1][start:end]))
#             slc_shareds.append(ss[1])
#         else:
#             slc_givens.append((ss, ss[start:end]))
#             slc_shareds.append(ss)
#     if len(givens) == 0: givens = None
#     if len(slc_givens) == 0: slc_givens = None
#     slc_idx_inputs = [] if slc_shareds is None else [start, end]
#     return givens, slc_givens, slc_idx_inputs, slc_shareds


def check_collect_modes(collect_modes):
    if any([mode not in COLLECT_MODES for mode in collect_modes]):
        raise ValueError("Had an invalid collect mode in: \n{}"
            "\n\tpossible modes are: \n{}".format(collect_modes, COLLECT_MODES))


###############################################################################
#                                                                             #
#                           Worker Tasks                                      #
#                                                                             #
###############################################################################


def receive_distribution():
    with open(PKL_FILE, "rb") as f:
        distribution = pickle.load(f)
    if sync.barrier.wait() == 0:  # (only one worker does it)
        import os
        os.remove(PKL_FILE)  # (leave no trace)
    synk_funcs = list()
    shareds_registry.reset()
    for i, f_info in enumerate(distribution):
        assert f_info["ID"] == i
        synk_funcs.append(WorkerFunction(**f_info))
        shareds_registry.register_func(f_info["functions"]["theano_function"])
    return synk_funcs
