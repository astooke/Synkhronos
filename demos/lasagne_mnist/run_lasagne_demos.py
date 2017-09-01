
"""
Call this script with varying batch sizes (e.g. 500, 5000) and watch the time
per epoch to see multi-GPU speedup.
"""

import argparse
import sys
import subprocess

SCRIPTS = ["demos/lasagne_mnist/train_mnist_theano.py",
           "demos/lasagne_mnist/train_mnist_cpu_data.py",
           "demos/lasagne_mnist/train_mnist_gpu_data.py",
           ]


def launch_demo(run_script, demo_args):
    call_string = "python {} {}".format(run_script, demo_args)
    print("\nCalling:\n", call_string, "\n")
    call_list = call_string.split(" ")
    p = subprocess.Popen(call_list)
    p.wait()


def main(batch_size):
    demo_args = "cnn " + str(batch_size)
    for script in SCRIPTS:
        launch_demo(script, demo_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_size", type=int, default=500, nargs='?')
    args = parser.parse_args(sys.argv[1:])
    main(args.batch_size)
