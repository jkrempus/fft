#!/usr/bin/env python3
import subprocess
from os import pardir, path
import sys;
import operator
import random

executable = path.abspath(path.join(path.dirname(__file__), "../test"))
print("executable", executable)

precision = 1.2e-7

already_run_tests = set()
m = 22 #maximal log2n

def run_single(sizes, is_real, is_inverse):
    cmd = [executable, "fft"] + [str(i) for i in sizes]
    cmd.append("-p={}".format(precision))
    if is_real: cmd.append("-r")
    if is_inverse: cmd.append("-i")
    print(cmd, file=sys.stderr)
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=None)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

def run(sizes):
    if sizes in already_run_tests: return False
    already_run_tests.add(sizes)

    run_single(sizes, False, False)
    run_single(sizes, False, True)
    run_single(sizes, True, False)
    run_single(sizes, True, True)
    return True

for i in range(1, m + 1): run((i,))
for i in range(1, m // 2 + 1): run((i, i))
for i in range(1, m // 3 + 1): run((i, i, i))

def keyfn(s): return len(s), sum(s)

def get_all_sizes(s, sz_left, max_len):
    r = [s] if (len(s)) > 0 else []
    if len(s) < max_len:
        for e in range(1, sz_left + 1):
            r += get_all_sizes(s + (e,), sz_left - e, max_len)

    return r

def get_weights(all_sizes):
    hist = dict()
    for e in all_sizes:
        hist[keyfn(e)] = hist.get(keyfn(e), 0) + 1

    return [1.0 / hist[keyfn(e)] for e in all_sizes]

all_sizes = get_all_sizes((), m, 5)
weights = get_weights(all_sizes)

print("num sizes", len(all_sizes))

num_tested = 0
remaining_size = 1.0
while remaining_size > 0:
    for e in random.choices(range(len(weights)), weights, k=1000):
        s = all_sizes[e]
        if run(s):
            num_tested += 1
            remaining_size -= 2 ** sum(s) * 1e-9
            print(
                "remaining size", remaining_size,
                "G ratio", float(num_tested) / len(all_sizes))
            if remaining_size <= 0: break
        else:
            print("skipped", s)
