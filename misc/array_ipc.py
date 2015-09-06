from numpy import *
from ctypes import *
import socket
import os
import sys
import threading

elemetype_dict = {
    0 : float32,
    1 : float64,
    2 : int8,
    3 : int16,
    4 : int32,
    5 : int64,
    6 : uint8,
    7 : uint16,
    8 : uint32,
    9 : uint64}

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

def create_server(address):
    try: 
        os.unlink(address)
    except:
        pass

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(address)
    server.listen(100)
    return server


def listen_loop(server, callback):
    while True:
        print "accepting..."
        sock, _ = server.accept()
        print "accepted."
        callback(*read_array(sock))
        print "done."
        sock.close()

class Received(dict):
    def __getattr__(self, key): return self[key][-1]

received = Received()
def store_callback(name, arr):
    received[name] = received.get(name, [])
    received[name].append(arr)

def listen(address, callback = store_callback):
    server = create_server(address) 
    threading.Thread(target=listen_loop, args=(server, callback)).start()
    return server

