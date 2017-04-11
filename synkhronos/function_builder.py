
import theano
import pickle

from .function import Function, WorkerFunction
from .collectives import shareds_registry
from . import exct
from .util import PKL_FILE

sync = None

synk_functions = list()


###############################################################################
#                                                                             #
#                       API for building Functions                            #
#                                                                             #
###############################################################################


def function(inputs, outputs=None, bcast_inputs=None, updates=None,
             givens=None, sliceable_shareds=None,
             **kwargs):
    if not exct.state.forked:
        raise RuntimeError("Must fork before making functions for GPU.")
    if exct.state.distributed:
        raise RuntimeError("Cannot make new functions after distributing (for now).")

    if not isinstance(inputs, list):
        raise TypeError("Input 'inputs' must be list.")
    bcast_inputs = [] if bcast_inputs is None else bcast_inputs
    if not isinstance(bcast_inputs, list):
        raise TypeError("Input 'bcast_inputs' must be list if not None.")

    gpu_outputs, to_cpu, output_modes = \
        process_outputs(outputs)
    reg_updates, update_vars, slc_update_gpu_outs, update_modes = \
        process_updates(updates)
    reg_givens, slc_givens, slc_idx_inputs, slc_shareds = \
        process_givens(givens, sliceable_shareds)

    theano_function = theano.function(
        inputs=inputs + bcast_inputs,
        outputs=gpu_outputs,  # a list, so function always returns list
        updates=reg_updates,
        givens=reg_givens,
        **kwargs,
    )
    if len(update_vars) == 0 and slc_shareds is None:
        sliced_function = theano_function
    else:
        sliced_function = theano.function(
            inputs=inputs + bcast_inputs + slc_idx_inputs,
            outputs=gpu_outputs + slc_update_gpu_outs,
            givens=slc_givens,
            **kwargs,
        )

    synk_function = Function(ID=len(synk_functions),
                             theano_function=theano_function,
                             sliced_function=sliced_function,
                             inputs=inputs,
                             bcast_inputs=bcast_inputs,
                             slc_shareds=slc_shareds,
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
    distribution = [sf._get_distro_info() for sf in synk_functions]
    with open(PKL_FILE, "wb") as f:
        pickle.dump(distribution, f, pickle.HIGHEST_PROTOCOL)
    exct.launch(exct.DISTRIBUTE)
    exct.join()
    print("...distribution complete.")
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
        return [], [], []
    output_vars = list()
    output_modes = list()
    from theano.gpuarray.type import GpuArrayVariable
    len_err = ValueError("Output tuples must be length 2: (var, collect_mode).")
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
        output_vars.append(o)
        output_modes.append("avg")
    check_collect_modes(output_modes)
    to_cpu = [not isinstance(var, GpuArrayVariable) for var in output_vars]
    gpu_vars = [var.transfer(None) for var in output_vars]
    return gpu_vars, to_cpu, output_modes


def process_updates(updates):
    if updates is None:
        return None, [], []
    in_err = TypeError("Input 'updates' must be a list of tuples: "
            "(var, new_value, [collect_mode])")
    if not isinstance(updates, list): raise in_err
    reg_updates = list()
    update_vars = list()
    update_gpu_outs = list()
    update_modes = list()
    for u in updates:
        if not isinstance(u, tuple) or len(u) not in (2, 3): raise in_err
        reg_updates.append((u[0], u[1]))
        update_vars.append(u[0])
        update_gpu_outs.append(u[1].transfer(None))
        update_modes.append(u[2] if len(u) == 3 else "avg")
    check_collect_modes(update_modes)
    return reg_updates, update_vars, update_gpu_outs, update_modes


def process_givens(givens, sliced_shareds):
    if sliced_shareds is None:
        return givens, None, [], []
    import theano.tensor as T
    giv_err = TypeError("If using 'sliced_shareds', givens must be list of 2-tuples.")
    s_err = TypeError("Input 'sliced_shareds' must be list, elements are "
        "individual shared variables or 2-tuples: (var, given_var)")
    givens = list() if givens is None else givens
    if not isinstance(givens, list):
        raise giv_err
    for g in givens:
        if not isinstance(g, tuple) or len(g) != 2:
            raise giv_err
    if not isinstance(sliced_shareds, list): raise s_err
    start = T.lscalar()
    end = T.lscalar()
    slc_givens = list()
    slc_shareds = list()
    for ss in sliced_shareds:
        if isinstance(ss, tuple):
            if len(ss) != 2: raise s_err
            givens.append(ss)
            slc_givens.append((ss[0], ss[1][start:end]))
            slc_shareds.append(ss[1])
        else:
            slc_givens.append((ss, ss[start:end]))
            slc_shareds.append(ss)
    givens = None if len(givens) == 0 else givens
    slc_givens = None if len(sliced_shareds) == 0 else givens
    slc_idx_inputs = [] if slc_givens is None else [start, end]
    return givens, slc_givens, slc_idx_inputs, slc_shareds


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
        if f_info["sliced_function"] is None:  # (avoided duplicate pickling)
            f_info["sliced_function"] = f_info["theano_function"]
        assert f_info["ID"] == i
        synk_funcs.append(WorkerFunction(**f_info))
        shareds_registry.register_func(f_info["theano_function"])
    return synk_functions
