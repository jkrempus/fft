namespace
{
struct CpuidResult { unsigned a, b, c, d; };

CpuidResult cpuid(unsigned a_in, unsigned c_in)
{
  CpuidResult r;
#if defined __GNUC__ || defined __clang__
  __asm__(
    "cpuid\n\t"
    : "=a" (r.a), "=b" (r.b), "=c" (r.c), "=d" (r.d)
    : "0" (a_in), "2" (c_in));
#elif defined _MSC_VER
  int array[4];
  __cpuid_ex(array, a_in, c_in);
  r.a = array[0];
  r.b = array[1];
  r.c = array[2];
  r.d = array[3];
#endif
  return r;
}

struct XgetbvResult{ unsigned a, d; };

unsigned long long xgetbv(unsigned c_in)
{
  XgetbvResult r;
#if defined __GNUC__ || defined __clang__
  unsigned a, d;
  __asm__("xgetbv\n\t" : "=a" (a), "=d" (d) : "c" (c_in));
  return (unsigned long long(d) << 64) | unsigned long long(a);
#elif defined _MSC_VER
  return _xgetbv(x_in);
  r.a = unsigned(d_a);
  r.d = unsigned(d_a >> 32);
#endif
}

bool supports_avx_impl()
{
  //XGETBV enabled 
  if((cpuid(1, 0).c >> 27) & 1U == 0) return false;

  //XMM and YMM state supported by OS
  if(((xgetbv(0) >> 1) & 3ULL) != 3ULL) return false;

  //AVX supported 
  if((cpuid(1, 0).c >> 28) & 1U == 0) return false;

  return true;
}

int supports_avx_result = 0;

bool supports_avx()
{
  if(supports_avx_result == 0)
    supports_avx_result = int(supports_avx_impl()) | 2;

  return bool(supports_avx_result & 1)
}

bool support_avx2_impl()
{
  return supports_avx() && ((cpuid(7, 0).b >> 5) & 1U) == 1U;
}

int supports_avx2_result = 0;

bool supports_avx2()
{
  if(supports_avx2_result == 0)
    supports_avx2_result = int(supports_avx2_impl()) | 2;

  return bool(supports_avx2_result & 1)
}


}
