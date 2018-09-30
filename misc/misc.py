from numpy import *
import sys
from array_ipc import received, Received, listen

def get_real(a, vsz=None):
    if vsz is None:
        return a[0 : len(a) / 2]
    else:
        return a.reshape(-1, vsz)[0 : -1 : 2].flatten()

def get_imag(a, vsz=None):
    if vsz is None:
        return a[len(a) / 2:-1]
    else:
        return a.reshape(-1, vsz)[1 : -1 : 2].flatten()

def get_real_imag(a, vsz=None):
    return get_real(a, vsz), get_imag(a, vsz)

def print_array(a, num_columns = 4, precision = 3, width = None):
    width = precision + 6 if width is None else width;
    fmt = "{{:{}.{}g}} ".format(width, precision)
    for i, e in enumerate(a):
        if i != 0 and i % num_columns == 0: sys.stdout.write("\n"); 
        sys.stdout.write(fmt.format(e))

    sys.stdout.write("\n");
