
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
        self.create = None
        self.synk_datas = list()
        self.tag = -1

    def assign_rank(self, n_parallel, rank, create):
        self.n_parallel = n_parallel
        self.rank = rank
        self.create = create

    def __len__(self):
        return len(self.synk_datas)

    def __getitem__(self, k):
        return self.synk_datas[k]

    def append(self, synk_data):
        self.synk_datas.append(synk_data)

    def get_my_inputs(self, n_scat_inputs, n_bcast_inputs):
        n_tot_inputs = n_scat_inputs + n_bcast_inputs
        if n_tot_inputs == 0:
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
        for data_ID in sync.data_IDs[n_scat_inputs:n_tot_inputs]:
            synk_data = self.synk_datas[data_ID]
            my_inputs.append(synk_data._data)
        return tuple(my_inputs)

    def _alloc_idxs_arr(self, size, tag):
        tag = PREFIX + "_scat_idxs_" + str(tag)
        sync.idxs_arr = NpShmemArray('int64', size, tag, self.create)

    ###########################################################################
    #                       Worker-only                                       #

    def check_idxs_alloc(self):
        """ (lazy update) """
        if sync.use_idxs_arr.value:
            if self.tag != sync.tag.value:
                size = sync.size.value
                self.tag = sync.tag.value
                self._alloc_idxs_arr(size, self.tag)

    def get_data(self, data_ID):
        return self.synk_datas[data_ID]

    ###########################################################################
    #                           Master-only                                   #

    def assign_inputs(self, synk_datas, batch, num_scat):
        batch = check_batch_types(batch)
        if num_scat > 0:
            sync.assign_idxs[:] = \
                build_scat_idxs(self.n_parallel, synk_datas[:num_scat], batch)
            if batch is not None and not isinstance(batch, (int, slice)):
                sync.use_idxs_arr.value = True
                n_idxs = len(batch)
                if sync.idxs_arr is None or n_idxs > sync.idxs_arr.size:
                    self.alloc_idxs_arr(n_idxs)  # (will be oversized)
                sync.idxs_arr[:n_idxs] = batch
            else:
                sync.use_idxs_arr.value = False
        for i, synk_data in enumerate(synk_datas):
            sync.data_IDs[i] = synk_data._ID

    def alloc_idxs_arr(self, n_idxs):
        size = int(n_idxs * 1.1)  # (always some extra)
        sync.tag.value += 1
        sync.size.value = size
        self._alloc_idxs_arr(size, sync.tag.value)


scatterer = Scatterer()


###############################################################################
#                                                                             #
#                      Functions for Master                                   #
#                                                                             #
###############################################################################


def check_batch_types(batch):
    if batch is not None:
        if isinstance(batch, (list, tuple)):
            batch = np.array(batch, dtype='int64')
        if isinstance(batch, np.ndarray):
            if batch.ndim > 1:
                raise ValueError("Array for param 'batch' must be "
                    "1-dimensional, got: ", batch.ndim)
            if "int" not in batch.dtype.name:
                raise ValueError("Array for param 'batch' must be integer "
                    "dtype, got: ", batch.dtype.name)
        elif not isinstance(batch, (int, slice)):
            raise TypeError("Param 'batch' must be either an integer, a slice, "
                "or a list, tuple, or 1-D numpy array of integers.")
    return batch


def build_scat_idxs(n_parallel, synk_datas, batch):
    scat_lens = [len(sd) for sd in synk_datas if not sd._minibatch]
    minibatch_lens = [len(sd) for sd in synk_datas if sd._minibatch]
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
    return np.linspace(start, end, n_parallel + 1, dtype=np.int64)
