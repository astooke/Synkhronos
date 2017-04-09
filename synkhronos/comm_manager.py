
import zmq
import numpy as np

from .accumulators import accumulators

sync = None


###############################################################################
#                                                                             #
#                        Overall Comms Manager                                #
#                                                                             #
###############################################################################


class Comm(object):

    def __init__(self):
        self.cpu = None
        self.gpu = None
        try:
            from pygpu import collectives as gpu_coll
            self.gpu_coll_imported = True
        except ImportError as exc:
            self.gpu_coll_imported = False

    def connect_as_master(self, n_parallel, rank, master_rank, use_gpu,
                          min_port=1024, max_port=65535):
        self.cpu = CpuCommMaster()
        self.cpu.connect(n_parallel, master_rank, min_port, max_port)
        if use_gpu and self.gpu_coll_imported:
            self.gpu = GpuCommMaster()
            self.gpu.connect(n_parallel, rank, master_rank)

    def connect_as_worker(self, n_parallel, rank, master_rank, use_gpu):
        self.cpu = CpuCommWorker()
        self.cpu.connect(n_parallel, rank)
        if use_gpu and self.gpu_coll_imported:
            self.gpu = GpuCommWorker()
            self.gpu.connect(n_parallel, rank, master_rank)

    ###########################################################################
    #                       Support for Functions                             #

    def collect(self, arr, op, nccl=True):  # (Master only)
        if op is None:
            return arr
        comm = self.gpu if nccl and self.gpu is not None else self.cpu
        return comm.collect(arr, op)

    def send(self, arr, op, nccl=True):  # (Worker only)
        if op is None:
            return
        comm = self.gpu if nccl and self.gpu is not None else self.cpu
        comm.send(arr, op)

    ###########################################################################
    #                   Support for Shared Variable Collectives               #

    def broadcast(self, arr, nccl=True):
        comm = self.gpu if nccl and self.gpu is not None else self.cpu
        comm.broadcast(arr)

    def gather(self, arr, dest=None, nd_up=1, nccl=True):
        comm = self.gpu if nccl and self.gpu is not None else self.cpu
        return comm.gather(arr, dest, nd_up)

    def all_gather(self, arr, dest, nd_up=1, nccl=True):
        comm = self.gpu if nccl and self.gpu is not None else self.cpu
        return comm.all_gather(arr, dest, nd_up)

    def reduce(self, arr, op, dest, nccl=True):
        comm = self.gpu if nccl and self.gpu is not None else self.cpu
        return comm.reduce(arr, op, dest)

    def all_reduce(self, arr, op, nccl=True):
        comm = self.gpu if nccl and self.gpu is not None else self.cpu
        return comm.all_reduce(arr, op)


comm = Comm()


###############################################################################
#                                                                             #
#                          CPU Comm (using ZeroMQ)                            #
#                                                                             #
###############################################################################


class CpuCommMaster(object):

    def __init__(self):
        self.context = None
        self.sockets = None
        self.ports = None
        self.n = None

    def connect(self, n_parallel, master_rank, min_port=1024, max_port=65535):
        context = zmq.Context()
        sockets = list()
        ports = list()
        for i in range(n_parallel - 1):
            socket = context.socket(zmq.PAIR)
            port = socket.bind_to_random_port(
                "tcp://*", min_port=min_port, max_port=max_port)
            sockets.append(socket)
            ports.append(port)
        ports_send = list(ports)
        ports_send.append(ports[master_rank])  # (last worker gets this one)
        sync.dict["ports"] = ports_send
        for _ in range(n_parallel - 1):
            sync.semaphore.release()  # (let the workers connect)
        self.context = context
        self.sockets = sockets
        self.ports = ports
        self.n = n_parallel
        self.vec_ones = np.ones(self.n)

    ###########################################################################
    #                          Support for Function                           #

    def collect(self, arr, op):
        if op == "gather":
            return self.gather(arr)
        else:
            return self.reduce(arr, op)

    ###########################################################################
    #               Support for Shared Variable Collectives                   #

    def reduce(self, arr, op, dest=None):
        dtype = arr.dtype
        shape = arr.shape
        recv_buf = np.empty((self.n, *shape), dtype=dtype)
        dest = np.empty(*shape, dtype=dtype) if dest is None else dest
        assert dest.dtype == dtype
        assert dest.shape == shape
        recv_buf[-1] = np.asarray(arr)
        for i, socket in enumerate(self.sockets):
            recv_buf[i] = recv_nd_array(socket, arr)
        if op in ["sum", "avg"]:
            dest[:] = self.vec_ones.dot(recv_buf)  # parallel; np.mean is not
            if op == "avg":
                dest /= self.n
        elif op == "max":
            dest[:] = recv_buf.max(axis=0)
        elif op == "min":
            dest[:] = recv_buf.min(axis=0)
        elif op == "prod":
            dest[:] = recv_buf.prod(axis=0)
        return dest

    def all_reduce(self, arr, op, dest=None):
        recv_arr = self.reduce(arr, op, dest)
        self.broadcast(recv_arr)

    def broadcast(self, arr):
        for socket in self.sockets:
            send_nd_array(socket, arr)

    def gather(self, arr, nd_up=1, dest=None):
        if nd_up > 1:
            raise NotImplementedError
        arrs = [np.asarray(arr)]
        for socket in self.sockets:
            arrs.append(recv_nd_array(socket))
        combine = np.concatenate if nd_up == 0 else np.vstack
        if dest is not None:
            dest[:] = combine(arrs)  # TODO: put a try around this for bad dest
        else:
            dest = combine(arrs)
        return dest

    def all_gather(self, arr, nd_up=1, dest=None):
        recv_arr = self.gather(arr, nd_up, dest)
        self.broadcast(recv_arr)

    def scatter(self, arr, scat_dim=0):
        arr = np.asarray(arr)
        if scat_dim != 0:
            raise NotImplementedError
        n_data = arr.shape[scat_dim]
        n = -(- n_data // self.n)  # (ceiling div)
        last_n = n
        for socket in self.sockets[:-1]:
            send_nd_array(socket, arr[last_n:last_n + n])
            last_n += n
        send_nd_array(self.sockets[-1], arr[last_n:])
        return arr[:n]


class CpuCommWorker(object):

    def __init__(self):
        self.context = None
        self.socket = None
        self.port = None

    def connect(self, rank):
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        sync.semaphore.acquire()
        port = sync.dict["ports"][rank]
        socket.connect("tcp://localhost:%s" % port)
        self.context = context
        self.socket = socket
        self.port = port

    ###########################################################################
    #                       Support for Functions                             #

    def send(self, arr):  # (reduce, gather)
        send_nd_array(self.socket, np.asarray(arr))

    ###########################################################################
    #                Support for Shared Variable Collectives                  #

    def broadcast(self, arr=None):
        return recv_nd_array(self.socket)

    def gather(self, arr):
        send_nd_array(self.socket, np.asarray(arr))

    def all_gather(self, arr, dest=None):
        send_nd_array(self.socket, np.asarray(arr))
        return recv_nd_array(self.socket)

    def reduce(self, arr, op=None):
        send_nd_array(self.socket, np.asarray(arr))

    def all_reduce(self, arr, op=None, dest=None):
        send_nd_array(self.socket, np.asarray(arr))
        return recv_nd_array(self.socket)


def send_nd_array(socket, arr):
    socket.send_string(arr.dtype.name)
    socket.send_string(str(arr.shape).lstrip('(').rstrip(')').rstrip(','))
    socket.send(arr, copy=False)


def recv_nd_array(socket, expected_arr=None):
    dtype = socket.recv_string()
    shape = socket.recv_string()
    shape = () if not shape else tuple([int(s) for s in shape.split(',')])
    if expected_arr is not None:
        assert expected_arr.dtype.name == dtype
        assert expected_arr.shape == shape
    arr = socket.recv(copy=False)
    return np.frombuffer(arr, dtype=dtype).reshape(shape)


###############################################################################
#                        No more super class...that's confusing               #

class GpuComm(object):

    def __init__(self):
        self.master_rank = None
        self.n_gpu = None
        self.avg_fac = 1.

    def connect(self, n_gpu, rank, master_rank):
        import theano
        import theano.gpuarray
        from pygpu import collectives as gpu_coll
        gpu_ctx = theano.gpuarray.get_context(None)
        clique_id = gpu_coll.GpuCommCliqueId(gpu_ctx)
        if rank == master_rank:
            sync.dict["gpu_comm_id"] = clique_id.comm_id
            sync.barrier.wait()
        else:
            sync.barrier.wait()
            clique_id.comm_id = sync.dict["gpu_comm_id"]
        self.comm = gpu_coll.GpuComm(clique_id, n_gpu, rank)
        self.n_gpu = n_gpu
        self.avg_fac = 1. / n_gpu
        self.master_rank = master_rank


class GpuCommMaster(GpuComm):

    ###########################################################################
    #                           Support for Functions                         #

    def collect(self, arr, op):
        if op == "gather":
            return self.comm.all_gather(src=arr)
        else:
            avg = op == "avg"
            op = "sum" if avg else op
            self.comm.reduce(src=arr, op=op, dest=arr)
            if avg:
                avg_f = accumulators.get_avg_f(arr)
                arr = avg_f(arr, self.avg_fac)
            return arr

    ###########################################################################
    #                  Support for Shared Variable Collectives                #

    def broadcast(self, arr):
        self.comm.broadcast(src=arr)

    def gather(self, arr, dest=None, nd_up=1):
        return self.comm.all_gather(src=arr, dest=dest, nd_up=nd_up)

    def all_gather(self, arr, dest=None, nd_up=1):
        return self.comm.all_gather(src=arr, dest=dest, nd_up=nd_up)

    def reduce(self, arr, op, dest=None):
        return self._reduce(arr, op, dest)

    def all_reduce(self, arr, op, dest=None):
        return self._reduce(arr, op, dest)

    def _reduce(self, arr, op, dest=None):
        avg = op == "avg"
        op = "sum" if avg else op
        r = self.comm.reduce(src=arr, op=op, dest=dest)
        if dest is not None:
            r = dest
        if avg:
            avg_f = accumulators.get_avg_f(r)
            r = avg_f(r, self.avg_fac)
        return r  # TODO: make sure shared var gets result of this function.


class GpuCommWorker(GpuComm):

    ###########################################################################
    #                           Support for Functions                         #

    def send(self, arr, op):
        if op is None:
            return
        elif op == "gather":
            self.comm.all_gather(src=arr)
        else:
            op = "sum" if op == "avg" else op
            # NOTE: kwarg "dest" only needed for NCCL bug.
            self.comm.reduce(src=arr, op=op, root=self.master_rank, dest=arr)

    ###########################################################################
    #                   Support for Shared Variable Collectives               #

    def broadcast(self, arr):
        self.comm.broadcast(array=arr, root=self.master_rank)

    def gather(self, arr, nd_up=1):
        self.comm.all_gather(src=arr, nd_up=1)

    def all_gather(self, arr, dest):
        return self.comm.all_gather(src=arr, dest=dest, nd_up=0)  # FIXME

    def reduce(self, arr, op):
        op = "sum" if op == "avg" else op
        self.comm.reduce(src=arr, op=op, root=self.master_rank)

    def all_reduce(self, arr, op, dest):
        avg = op == "avg"
        op = "sum" if avg else op
        self.comm.all_reduce(src=arr, op=op, dest=dest)
        avg_f = accumulators.get_avg_f(dest)
        return avg_f(dest, self.avg_fac)  # TODO: make sure shared var gets it.
