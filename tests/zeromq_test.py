
import multiprocessing as mp 
import zmq 
import time
# import theano.gpuarray



def master(n_pairs, barrier, sync_dict):
    # theano.gpuarray.use("cuda" + str(n_pairs))
    context = zmq.Context()
    n_parallel = n_pairs + 1
    min_port = 1024
    max_port = 65535
    sockets = list()
    ports = list()
    
    rank_pair_sockets = list()
    pair_sockets = list()
    rank_pair_ports = list()
    for i in range(n_parallel):
        if i == 0:
            rank_pair_sockets.append(None)
            rank_pair_ports.append(None)
        else:
            socket = context.socket(zmq.PAIR)
            port = socket.bind_to_random_port(
                "tcp://*", min_port=min_port, max_port=max_port)
            print("master had pair port ", i, ": ", port)
            rank_pair_sockets.append(socket)
            rank_pair_ports.append(port)
            pair_sockets.append(socket)

    sync_dict["pair_ports"] = rank_pair_ports
    barrier.wait()

    for i, socket in enumerate(rank_pair_sockets):
        if i == 0:
            print("skipping port idx ", i)
        else:
            print("sending test string in loop, ", i)
            socket.send_string("test")
    print("dont with test string loop")

    # for i in range(n_pairs):
    #     socket = context.socket(zmq.PAIR)
    #     port = socket.bind_to_random_port(
    #         "tcp://*", min_port=min_port, max_port=max_port)
    #     sockets.append(socket)
    #     ports.append(port)

    # sync_dict["ports"] = ports 
    # barrier.wait()
    # for i, sock in enumerate(sockets):
    #     print("attempting to send to ", i)
    #     sock.send_string("test")
    # print("finished send loop")


def main(n_pairs=7):
    n_pairs = int(n_pairs)

    barrier = mp.Barrier(n_pairs + 1)
    mgr = mp.Manager()
    sync_dict = mgr.dict()

    workers = [mp.Process(target=worker, args=(rank + 1, barrier, sync_dict))
            for rank in range(n_pairs)]

    for w in workers:
        w.start()

    master(n_pairs, barrier, sync_dict)

    for w in workers:
        w.join() 


def worker(rank, barrier, sync_dict):
    # theano.gpuarray.use("cuda" + str(rank))
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    barrier.wait()
    time.sleep(rank + 1)
    port = sync_dict["pair_ports"][rank]
    # socket = context.socket(zmq.PAIR)
    print(rank, " connecting to port ", port)
    socket.connect("tcp://localhost:%s" % port)
    print(rank, " polling for test string")
    for i in range(3):
        poll_result = socket.poll(timeout=250)
        print(rank, " result of poll: ", poll_result)
        if poll_result:
            break
    
    print(rank, " attempting to receive string")
    test_string = socket.recv_string()
    assert test_string == "test"
    print(rank, " passed recv test")


if __name__ == "__main__":
    main()
