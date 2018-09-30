#ifndef _ARRAY_IPC_H_
#define _ARRAY_IPC_H_

#ifdef ARRAY_IPC_ENABLED

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace array_ipc
{
namespace
{
enum class PackageType { onedim_array };

template<typename T> int64_t element_type_id();

template<> int64_t element_type_id<float>(){ return 0; }
template<> int64_t element_type_id<double>(){ return 1; }
template<> int64_t element_type_id<int8_t>(){ return 2; }
template<> int64_t element_type_id<int16_t>(){ return 3; }
template<> int64_t element_type_id<int32_t>(){ return 4; }
template<> int64_t element_type_id<int64_t>(){ return 5; }
template<> int64_t element_type_id<uint8_t>(){ return 6; }
template<> int64_t element_type_id<uint16_t>(){ return 7; }
template<> int64_t element_type_id<uint32_t>(){ return 8; }
template<> int64_t element_type_id<uint64_t>(){ return 9; }

void write_all(int sock, const void* buf, size_t n)
{
  while(n > 0)
  {
    int r = write(sock, buf, n);
    if(r <= 0) 
    {
      perror("write()\n");
      abort();
    }
    n -= r;
    buf = (void*)(size_t(buf) + r);
  }
}

template<typename T>
void write_val(int sock, const T& value)
{
  write_all(sock, &value, sizeof(value));
}

template<typename T>
void write_array(int sock, const char* name, const T* ptr, size_t len)
{
  write_val(sock, int64_t(PackageType::onedim_array));

  int64_t name_len = strlen(name);  
  write_val(sock, name_len);
  write_all(sock, name, name_len);
  
  write_val(sock, element_type_id<T>());

  write_val(sock, int64_t(len));
  write_all(sock, ptr, len * sizeof(T));
}

static int uds_connect(const char* name)
{
  int sock = socket(AF_UNIX, SOCK_STREAM, 0);
  if (sock < 0) {
    perror("socket()\n");
    abort();
  }

  struct sockaddr_un server;
  server.sun_family = AF_UNIX;
  strcpy(server.sun_path, name);

  int r = connect(sock, (struct sockaddr *) &server, sizeof(struct sockaddr_un));
  if (r < 0) {
    /*
    close(sock);
    perror("connect()");
    abort();*/
    return -1;
  }

  return sock;
}
}

template<typename T>
void send(const char* name, const T* ptr, size_t len)
{
  int sock = uds_connect("/tmp/array_ipc");
  if(sock >= 0)
  {
    write_array(sock, name, ptr, len);
    close(sock);  
  }
}
}
#else
namespace array_ipc
{
template<typename T> void send(const char* name, T* ptr, int len) { }
}
#endif
#endif
