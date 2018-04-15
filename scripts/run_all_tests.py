#!/usr/bin/env python3
import subprocess
from os import pardir, path
import sys;
import operator

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
