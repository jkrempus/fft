#include <cmath>
#include <cstdio>
#include <cstring>

template<class T>
void print_code(const char* type, const char* name, int n, T (*fun)(T))
{
  printf("%s %s[%d] = {\n  ", type, name, n);
  for(int i = 0; i < n; i++)
  {
    printf("%a%s",
      fun(T(M_PI) * std::ldexp(T(1), -i)),
      i == n - 1 ? "};" : ", ");
    
    if((i + 1) % 4 == 0) printf("\n  ");
  }
}

int main()
{
  print_code<float>("float", "fsin", 64, std::sin);
  printf("\n");
  print_code<float>("float", "fcos", 64, std::cos);
  return 0;
}
