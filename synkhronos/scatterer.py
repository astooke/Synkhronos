
import numpy as np

from .util import PREFIX
from .shmemarray import NpShmemArray


sync = None


###############################################################################
#                                                                             #
#          Scatterer: for moving input data to workers                        #
#                                                                             #
###############################################################################


class Scatterer(object):

    def __init__(self):
        self.n_parallel = None
        self.rank = None
        self.synk_datas = list()
        self.tag = 0
        self.tag_s = 0

    def assign_rank(self, n_parallel, rank):
        self.n_parallel = n_parallel
        self.rank = rank

    def __len__(self):
        return len(self.synk_datas)

    def __getitem__(self, k):
        return self.synk_datas[k]

    def append(self, synk_data):
        self.synk_datas.append(synk_data)

    def get_my_inputs(self, n_scat_inputs, n_bcast_inputs=0):
        n_inputs = n_scat_inputs + n_bcast_inputs
        if n_inputs == 0:
            return ()
        if n_scat_inputs > 0:
            my_idxs = slice(*sync.assign_idxs[self.rank:self.rank + 2])
            minibatch_0 = min(sync.assign_idxs)  # minib is exactly right size
            minibatch_idxs = \
                slice(my_idxs.start + minibatch_0, my_idxs.stop + minibatch_0)
            if sync.use_idxs_arr.value:
                my_idxs = sync.idxs_arr[my_idxs]
        my_inputs = list()
        for data_ID in sync.data_IDs[:n_scat_inputs]:
            synk_data = self.synk_datas[data_ID]
            if synk_data._minibatch:  # (assumes already shuffled if needbe)
                my_inputs.append(synk_data._data[minibatch_idxs])
            else:
                my_inputs.append(synk_data._data[my_idxs])
        for data_ID in sync.data_IDs[n_scat_inputs:n_inputs]:
            synk_data = self.synk_datas[data_ID]
            my_inputs.append(synk_data._data)
        return tuple(my_inputs)

    def get_my_batch_s(self, n_slc_shareds):
        if n_slc_shareds > 0 and sync.use_batch_s.value:
            batch_s = slice(*sync.assign_idxs_s[self.rank])
            if sync.use_idxs_arr_s.value:
                batch_s = sync.idxs_arr_s[batch_s]
        else:
            batch_s = None
        return batch_s

    ###########################################################################
    #                       Worker-only                                       #

    def check_idxs_alloc(self):
        """ (lazy update) """
        if sync.use_idxs_arr.value:
            if self.tag != sync.tag.value:
                alloc_idxs_arr()
                self.tag = sync.tag.value
        if sync.use_idxs_arr_s.value:
            if self.tag_s != sync.tag_s.value:
                alloc_idxs_arr_s()
                self.tag_s = sync.tag_s.value

    ###########################################################################
    #                           Master-only                                   #

    def assign_inputs(self, synk_datas, batch, num_scat=None):
        batch = check_batch_types(batch)
        if num_scat is None:
            num_scat = len(synk_datas)
        if num_scat > 0:
            assign_idxs(self.n_parallel, synk_datas[:num_scat], batch)
            update_idxs_arr(batch)
        update_IDs(synk_datas)

    def assign_batch_s(self, batch_s):
        batch_s = check_batch_s_types(batch_s, self.n_parallel)
        if batch_s is not None:
            assign_idxs_s(batch_s, self.n_parallel)
            update_idxs_arr_s(batch_s)


scatterer = Scatterer()


def alloc_idxs_arr(size=None):
    create = size is not None
    if create:
        sync.size.value = size
        sync.tag.value += 1
    else:
        size = sync.size.value
    tag = PREFIX + "_scat_idxs_" + str(sync.tag.value)
    sync.idxs_arr = NpShmemArray('int64', size, tag, create)


def alloc_idxs_arr_s(size=None):
    create = size is not None
    if create:
        sync.size_s.value = size
        sync.tag_s.value += 1
    else:
        size = sync.size_s.value
    tag = PREFIX + "_idxs_s_" + str(sync.tag_s.value)
    sync.idxs_arr_s = NpShmemArray('int64', size, tag, create)


###############################################################################
#                                                                             #
#                      Functions for Master                                   #
#                                                                             #
###############################################################################


def check_batch_types(batch):
    if batch is not None:
        if isinstance(batch, (list, tuple)):
            batch = np.array(batch)
        if isinstance(batch, np.ndarray):
            if batch.ndim > 1:
                raise ValueError("Array for param 'batch' must be "
                    "1-dimensional, got: {}".format(batch.ndim))
            if not np.issubdtype(batch, int):
                raise ValueError("Array for param 'batch' must be integer "
                    "dtype, got: {}".format(batch.dtype.name))
        elif isinstance(batch, slice):
            # TODO: come back and support full slice.
            assert batch.start is not None and batch.start >= 0
            assert batch.stop is not None and batch.stop > 0
            assert batch.step is None
        elif not isinstance(batch, int):
            raise TypeError("Param 'batch' must be either an integer, a slice, "
                "or a list, tuple, or 1-D numpy array of integers.")
    return batch


def assign_idxs(n_parallel, scat_datas, batch):
    scat_lens = [len(sd) for sd in scat_datas if not sd._minibatch]
    minibatch_lens = [len(sd) for sd in scat_datas if sd._minibatch]
    check_len = min(scat_lens) if len(scat_lens) > 0 else min(minibatch_lens)
    if batch is not None:
        if isinstance(batch, int):  # (size from 0 is used)
            max_idx = batch
            start = 0
            end = batch
        elif isinstance(batch, slice):  # (slice is used)
            max_idx = batch.stop
            start = batch.start
            end = batch.stop
        else:  # (explicit indices are used)
            max_idx = max(batch)
            start = 0
            end = len(batch)
        if max_idx > check_len:
            raise ValueError("Requested index out of range of input lengths.")
    else:  # (i.e. no batch directive provided, use full array scat_lens)
        start = 0
        end = check_len
        if scat_lens.count(end) + minibatch_lens.count(end) != \
                len(scat_lens) + len(minibatch_lens):  # (fast)
            raise ValueError("If not providing param 'batch', all "
                "inputs must be the same length.  Had scat_lengths: {}"
                " and minibatch_lengths: {}".format(scat_lens, minibatch_lens))
    if minibatch_lens:
        if min(minibatch_lens) < end - start:
            raise ValueError("Had minibatch input length less than size of "
                "request batch.  Minibatch lengths: {}".format(minibatch_lens))
    sync.assign_idxs[:] = np.linspace(start, end, n_parallel + 1, dtype='int64')


def update_idxs_arr(batch):
    if isinstance(batch, np.ndarray):
        sync.use_idxs_arr.value = True
        if sync.idxs_arr is None or batch.size > sync.idxs_arr.size:
            size = int(batch.size * 1.1)
            alloc_idxs_arr(size)
        sync.idxs_arr[:batch.size] = batch
    else:
        sync.use_idxs_arr.value = False


def update_IDs(synk_datas):
    for i, sd in enumerate(synk_datas):
        sync.data_IDs[i] = sd._ID


def check_batch_s_types(batch_s, n_parallel):
    if batch_s is None:
        sync.use_batch_s.value = False
    else:
        sync.use_batch_s.value = True
        if isinstance(batch_s, list):
            idx_lists = isinstance(batch_s[0], (list, np.ndarray))
            idx_slices = isinstance(batch_s[0], slice)
            idx_list = isinstance(batch_s[0], int)
            if idx_lists:
                for i, b in enumerate(batch_s):
                    if not isinstance(b, (list, np.ndarray)):
                        raise TypeError("All must be same kind")
                    batch_s[i] = np.asarray(b)
                for b in batch_s:
                    if not np.issubdtype(b.dtype, int):
                        raise TypeError("must be integer type")
                    if b.ndim > 1:
                        raise TypeError("must be vector (1-dim)")
            elif idx_slices:
                for b in batch_s:
                    if not isinstance(b, slice):
                        raise TypeError("All must be same kind")
                    # TODO: come back and support full slicing
                    assert b.start is not None and b.start >= 0
                    assert b.stop is not None and b.stop > 0
                    assert b.step is None
            elif idx_list:
                batch_s = np.array(batch_s)
            else:
                raise TypeError("If input is list, elements must be list, int, numpy array, or slice")
            if (idx_slices or idx_lists) and len(batch_s) != n_parallel:
                raise TypeError("must be one entry for all, or one for each")

        if isinstance(batch_s, np.ndarray):
            if not np.issubdtype(batch_s.dtype, int):
                raise TypeError("must be integer type")
            if batch_s.ndim > 1:
                raise TypeError("must be vector (1-dim)")
        elif isinstance(batch_s, slice):
            assert batch_s.start is not None and batch_s.start >= 0
            assert batch_s.stop is not None and batch_s.stop > 0
            assert batch_s.step is None

        if not isinstance(batch_s, (slice, list, np.ndarray)):
            raise TypeError("must be slice, list, or numpy array")

    return batch_s


def assign_idxs_s(batch_s, n_parallel):
    sync.use_batch_s.value = True
    if isinstance(batch_s, list):
        if isinstance(batch_s[0], np.ndarray):
            idxs = np.vstack([0, b.size] for b in batch_s)
            idxs = np.cumsum(idxs).reshape(idxs.shape)
        elif isinstance(batch_s[0], slice):
            idxs = np.vstack([[b.start, b.end] for b in batch_s])
    elif isinstance(batch_s, np.ndarray):
        idxs = np.tile([0, batch_s.size], [n_parallel, 1])
    elif isinstance(batch_s, slice):
        idxs = np.tile([batch_s.start, batch_s.end], [n_parallel, 1])
    sync.assign_idxs_s[:] = idxs
    # NOTE: different from scattering inputs; each worker has distinct pair idxs.


def update_idxs_arr_s(batch_s):
    batches_concat = None
    if isinstance(batch_s, np.ndarray):
        batches_concat = batch_s
    elif isinstance(batch_s, list):
        if isinstance(batch_s[0], np.ndarray):
            batches_concat = np.concatenate(batch_s)
    if batches_concat is None:
        sync.use_idxs_arr_s.value = False
    else:
        sync.use_idxs_arr_s.value = True
        if sync.idxs_arr_s is None or batches_concat.size > sync.idxs_arr_s.size:
            size = int(batches_concat.size * 1.1)
            alloc_idxs_arr_s(size)
        sync.idxs_arr_s[:batches_concat.size] = batches_concat
