
import zmq
import numpy as np
import theano
import theano.gpuarray
try:
    from pygpu import collectives as gpu_coll
except ImportError as exc:
    gpu_coll = None

from .reducers import reducers

sync = None
cpu = None
gpu = None


def connect_as_master(n_parallel, rank, master_rank, use_gpu,
                      min_port=1024, max_port=65535):
    global cpu, gpu
    cpu = CpuCommMaster(n_parallel, master_rank, min_port, max_port)
    if use_gpu:
        if gpu_coll is None:
            print("WARNING: Using GPUs but unable to import GPU "
                "collectives from pygpu (may need to install NCCL); "
                "reverting to CPU-based collectives.")
        else:
            gpu = GpuCommMaster(n_parallel, rank, master_rank)


def connect_as_worker(n_parallel, rank, master_rank, use_gpu):
    global cpu, gpu
    cpu = CpuCommWorker(rank)
    if use_gpu and gpu_coll is not None:
        gpu = GpuCommWorker(n_parallel, rank, master_rank)


###############################################################################
#                                                                             #
#                          CPU Comm (using ZeroMQ)                            #
#                                                                             #
###############################################################################


class CpuCommMaster(object):

    def __init__(self, n_parallel, master_rank, min_port=1024, max_port=65535):
        context = zmq.Context()
        pair_sockets = list()
        pair_ports = list()
        for i in range(n_parallel - 1):
            socket = context.socket(zmq.PAIR)
            port = socket.bind_to_random_port(
                "tcp://*", min_port=min_port, max_port=max_port)
            pair_sockets.append(socket)
            pair_ports.append(port)
        ports_send = list(pair_ports)
        ports_send.append(pair_ports[master_rank])  # (last worker gets this one)
        pub_socket = context.socket(zmq.PUB)
        pub_port = socket.bind_to_random_port(
            "tcp://*", min_port=min_port, max_port=max_port)
        ports_send.append(pub_port)
        sync.dict["ports"] = ports_send
        for _ in range(n_parallel - 1):
            sync.semaphore.release()  # (let the workers connect)
        self.context = context
        self.pair_sockets = pair_sockets
        self.pair_ports = pair_ports
        self.pub_socket = pub_socket
        self.pub_port = pub_port
        self.n = n_parallel
        self.vec_ones = np.ones(self.n)

    ###########################################################################
    #                       Support for Functions                             #

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
        for i, socket in enumerate(self.pair_sockets):
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
        else:
            raise ValueError("Unrecognized op: {}".format(op))
        return dest

    def all_reduce(self, arr, op, dest=None):
        recv_arr = self.reduce(arr, op, dest)
        self.broadcast(recv_arr)
        return recv_arr

    def broadcast(self, arr):
        send_nd_array(self.pub_socket, arr)

    def gather(self, arr, nd_up=1, dest=None):
        if nd_up > 1:
            raise NotImplementedError
        recv_arrs = [np.asarray(arr)]
        for socket in self.pair_sockets:
            recv_arrs.append(recv_nd_array(socket))
        combine = np.concatenate if nd_up == 0 else np.vstack
        if dest is not None:
            dest[:] = combine(recv_arrs)
        else:
            dest = combine(recv_arrs)
        return dest

    def all_gather(self, arr, nd_up=1, dest=None):
        recv_arr = self.gather(arr, nd_up, dest)
        self.broadcast(recv_arr)

    def scatter(self, arr):
        arr = np.asarray(arr)
        n_data = len(arr)
        n = -(- n_data // self.n)  # (ceiling div)
        last_n = n
        for socket in self.pair_sockets[:-1]:
            send_nd_array(socket, arr[last_n:last_n + n])
            last_n += n
        send_nd_array(self.pair_sockets[-1], arr[last_n:])
        return arr[:n]


class CpuCommWorker(object):

    def __init__(self, rank):
        context = zmq.Context()
        pair_socket = context.socket(zmq.PAIR)
        sync.semaphore.acquire()
        pair_port = sync.dict["ports"][rank]
        pair_socket.connect("tcp://localhost:%s" % pair_port)
        sub_socket = context.socket(zmq.SUB)
        sub_port = sync.dict["ports"][-1]
        sub_socket.connect("tcp://localhost:%s" % sub_port)
        self.context = context
        self.pair_socket = pair_socket
        self.pair_port = pair_port
        self.sub_socket = sub_socket
        self.sub_port = sub_port

    def send(self, arr):  # (Functions, reduce, gather)
        send_nd_array(self.pair_socket, np.asarray(arr))

    def recv_pub(self):  # (broadcast)
        return recv_nd_array(self.sub_socket)

    def recv_pair(self):  # (scatter)
        return recv_nd_array(self.pair_socket)

    def send_recv(self, arr):  # (all_gather, all_reduce)
        send_nd_array(self.pair_socket, np.asarray(arr))
        return recv_nd_array(self.sub_socket)


def send_nd_array(socket, arr):
    socket.send_string(arr.dtype.name)
    socket.send_string(str(arr.shape).lstrip('(').rstrip(')').rstrip(','))
    socket.send(arr, copy=False)


def recv_nd_array(socket):
    dtype = socket.recv_string()
    shape = socket.recv_string()
    shape = () if not shape else tuple([int(s) for s in shape.split(',')])
    arr = socket.recv(copy=False)
    return np.frombuffer(arr, dtype=dtype).reshape(shape)


###############################################################################
#                                                                             #
#                   GPU Comm (using NCCL via pygpu)                           #
#                                                                             #
###############################################################################


class GpuComm(object):

    def __init__(self, n_gpu, rank, master_rank):
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
            return self.comm.all_gather(src=arr, nd_up=0)
        else:
            avg = op == "avg"
            op = "sum" if avg else op
            # NOTE: all_reduce needed for NCCL bug in reduce
            # self.comm.reduce(src=arr, op=op, dest=arr)
            self.comm.all_reduce(src=arr, op=op, dest=arr)
            if avg:
                avg_f = reducers.get_avg_f(arr)
                arr = avg_f(arr, self.avg_fac)
            return arr

    ###########################################################################
    #                  Support for Shared Variable Collectives                #

    def broadcast(self, arr):
        self.comm.broadcast(src=arr)

    def gather(self, arr, nd_up=1):
        return self.comm.all_gather(src=arr, nd_up=nd_up)

    def all_gather(self, arr):
        return self.comm.all_gather(src=arr, nd_up=0)

    def reduce(self, arr, op, dest=None):
        self._reduce(arr, op, dest, all_reduce=False)

    def all_reduce(self, arr, op, dest=None):
        self._reduce(arr, op, dest, all_reduce=True)

    def _reduce(self, arr, op, dest=None, all_reduce=True):
        avg = op == "avg"
        if avg: op = "sum"
        reduce_method = self.comm.all_reduce if all_reduce else self.comm.reduce
        r = reduce_method(src=arr, op=op, dest=dest)
        if dest is not None: r = dest
        if avg:
            avg_f = reducers.get_avg_f(r)
            r = avg_f(r, self.avg_fac)
        return r  # (collectives.py uses shared_var.set_value(r))


class GpuCommWorker(GpuComm):

    ###########################################################################
    #                           Support for Functions                         #

    def send(self, arr, op):
        if op is None:
            return
        elif op == "gather":
            self.comm.all_gather(src=arr, nd_up=0)
        else:
            op = "sum" if op == "avg" else op
            # NOTE: all_reduce needed for for NCCL bug in reduce.
            # self.comm.reduce(src=arr, op=op, root=self.master_rank)
            self.comm.all_reduce(src=arr, op=op, dest=arr)

    ###########################################################################
    #                   Support for Shared Variable Collectives               #

    def broadcast(self, arr):
        self.comm.broadcast(array=arr, root=self.master_rank)

    def gather(self, arr, nd_up=1):
        self.comm.all_gather(src=arr, nd_up=nd_up)

    def all_gather(self, arr):
        return self.comm.all_gather(src=arr, nd_up=0)

    def reduce(self, arr, op):
        if op == "avg": op = "sum"
        self.comm.reduce(src=arr, op=op, root=self.master_rank)

    def all_reduce(self, arr, op):
        avg = op == "avg"
        if avg: op = "sum"
        self.comm.all_reduce(src=arr, op=op, dest=arr)
        if avg:
            avg_f = reducers.get_avg_f(arr)
            arr = avg_f(arr, self.avg_fac)
        return arr  # (collectives.py uses shared_var.set_value(arr))
