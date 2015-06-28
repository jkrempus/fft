from numpy import *
from ctypes import *
import socket
import os
import sys

elemetype_dict = { 0 : float32, 1 : float64 }

class PackageType(object):
    onedim_array = 0;

def recvall(sock, sz):
    r = ""
    while len(r) < sz:
        r += sock.recv(sz - len(r))

    return r

def read_int64(sock):
    data = recvall(sock, 8)
    return c_int64.from_buffer(create_string_buffer(data)).value

def read_array(sock):
    recvall(sock, 8)   #read package_type and ignore it
    
    name_len = read_int64(sock)
    name = recvall(sock, name_len)
    
    elemtype = elemetype_dict[read_int64(sock)]
    
    data_len = read_int64(sock);
    byte_len = data_len * elemtype().nbytes;
    data = recvall(sock, byte_len)
    r = zeros((data_len), dtype=elemtype)
    memmove(r.ctypes.data, create_string_buffer(data), byte_len);

    return name, r;

def listen(address, callback):
    try: 
        os.unlink(address)
    except:
        pass

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(address)
    server.listen(5)
    while True:
        sock, _ = server.accept()
        callback(*read_array(sock))
