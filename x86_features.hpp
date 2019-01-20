#ifndef X86_FEATURES_HPP
#define X86_FEATURES_HPP

#include <intrin.h>

namespace
{
namespace x86_features
{
namespace detail
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
  __cpuidex(array, a_in, c_in);
  r.a = array[0];
  r.b = array[1];
  r.c = array[2];
  r.d = array[3];
#endif
  return r;
}

unsigned long long xgetbv(unsigned c_in)
{
#if defined __GNUC__ || defined __clang__
  unsigned a, d;
  __asm__("xgetbv\n\t" : "=a" (a), "=d" (d) : "c" (c_in));
  using U = unsigned long long;
  return (U(d) << 32) | U(a);
#elif defined _MSC_VER
  return _xgetbv(c_in);
#endif
}

unsigned get_bit(unsigned bits, unsigned idx)
{
  return (bits >> idx) & 1U;
}

unsigned get_bits(unsigned bits, unsigned start, unsigned n)
{
  return (bits >> start) & ((1U << n) - 1);
}

enum { initialized_bit = 1, sse2_bit = 2, avx_bit = 4, avx2_bit = 8 };
unsigned supported = 0;

unsigned get_supported()
{
  if(supported == 0)
  {
    unsigned new_supported = initialized_bit;

    if(get_bits(cpuid(1, 0).d, 25, 2) == 3UL) new_supported |= sse2_bit;

    unsigned cpuid_1_0_c = cpuid(1, 0).c;
    if(
      get_bit(cpuid_1_0_c, 27) == 1U &&
      get_bits(xgetbv(0), 1, 2) == 3ULL &&
      get_bit(cpuid_1_0_c, 28) == 1U)
    {
      new_supported |= avx_bit;
      if(get_bit(cpuid(7, 0).b, 5) == 1U) new_supported |= avx2_bit;
    }

    supported = new_supported;
  }

  return supported;
}
}

bool supports_sse2()
{
  using namespace detail;
  return bool(get_supported() & sse2_bit);
}

bool supports_avx()
{
  using namespace detail;
  return bool(get_supported() & avx_bit);
}

bool supports_avx2()
{
  using namespace detail;
  return bool(get_supported() & avx2_bit);
}
}
}

#endif
