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

enum class PackageType { onedim_array };

template<typename T> int64_t element_type_id();

template<> int64_t element_type_id<float>(){ return 0; }
template<> int64_t element_type_id<double>(){ return 1; }

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
void write_array(int sock, const char* name, T* ptr, size_t len)
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
    close(sock);
    perror("connect()");
    abort();
  }

  return sock;
}

template<typename T>
void send(const char* name, T* ptr, size_t len)
{
  int sock = uds_connect("/tmp/array_ipc");
  write_array(sock, name, ptr, len);
  close(sock);  
}
}

#ifdef TEST_ARRAY_IPC
int main()
{
  float a[] = {1, 2, 3, 4};
  array_ipc::send("a", a, 4);
  return 0;
}
#endif
