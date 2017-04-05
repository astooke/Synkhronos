
import zmq
import numpy as np


class CpuCommMaster(object):

    def __init__(self, n_parallel, min_port=1024, max_port=65535):
        context = zmq.Context()
        sockets = list()
        ports = list()
        for i in range(n_parallel - 1):
            socket = context.socket(zmq.PAIR)
            port = socket.bind_to_random_port(
                "tcp://*", min_port=min_port, max_port=max_port)
            sockets.append(socket)
            ports.append(port)
        self.context = context
        self.sockets = sockets
        self.ports = ports
        self.n = n_parallel
        self.vec_ones = np.ones(self.n)

    def reduce(self, arr, op):
        dtype = arr.dtype
        shape = arr.shape
        recv_buf = np.empty((self.n, *shape), dtype=dtype)
        recv_buf[-1] = np.asarray(arr)
        for i, socket in enumerate(self.sockets):
            recv_buf[i] = recv_nd_array(socket, arr)
        if op in ["sum", "avg"]:
            recv_arr = self.vec_ones.dot(recv_buf)  # parallel; np.mean is not
            if op == "avg":
                recv_arr /= self.n
        elif op == "max":
            recv_arr = recv_buf.max(axis=0)
        elif op == "min":
            recv_arr = recv_buf.min(axis=0)
        elif op == "prod":
            recv_arr = recv_buf.prod(axis=0)
        return recv_arr

    def all_reduce(self, arr, op):
        recv_arr = self.reduce(arr, op)
        self.broadcast(recv_arr)

    def broadcast(self, arr):
        for socket in self.sockets:
            send_nd_array(socket, arr)

    def gather(self, arr, nd_up=1):
        if nd_up > 1:
            raise NotImplementedError
        arrs = list(np.asarray(arr))
        for socket in self.sockets:
            arrs.append(recv_nd_array(socket))
        if nd_up == 1:
            return np.vstack(arrs)
        else:
            return np.concatenate(arrs)

    def all_gather(self, arr, nd_up=1):
        recv_arr = self.gather(arr, nd_up)
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

    def __init__(self, port):
        context = zmq.Context()
        socket = context.socket(zmq.PAIR)
        socket.connect("tcp://localhost:%s" % port)
        self.context = context
        self.socket = socket
        self.port = port

    def send(self, arr):  # (reduce, gather)
        send_nd_array(self.socket, np.asarray(arr))

    def recv(self):  # (broadcast, scatter)
        return recv_nd_array(self.socket)

    def send_recv(self, arr):  # (all_reduce, all_gather)
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
