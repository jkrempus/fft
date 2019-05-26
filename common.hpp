#ifndef FFT_COMMON_H
#define FFT_COMMON_H

#ifndef AFFT_NO_STDLIB
#include <cstddef>
typedef size_t Uint;
typedef ptrdiff_t Int;
#else
typedef long Int;
typedef unsigned long Uint;
#endif

#ifdef _MSC_VER
#define FORCEINLINE __forceinline
#define HOT
#define NOINLINE __declspec(noinline)
#elif defined __GNUC__ || defined __clang__
#define FORCEINLINE __attribute__((always_inline)) inline
#define HOT __attribute__((hot))
#define NOINLINE __attribute__((noinline))
#endif

//#define ASSERT(condition) ((condition) || *((volatile int*) 0))
#define ASSERT(condition)

namespace
{
constexpr Int max_int = Int(Uint(-1) >> 1);

constexpr Int large_fft_size = 1 << 13;
constexpr Int optimal_size = 1 << 10;
constexpr Int max_vec_size = 8;
constexpr const Int align_bytes = 64;

template<typename V> using ET = typename V::T;

template<typename T, typename U>
struct SameType { static const bool value = false; };

template<typename T>
struct SameType<T, T> { static const bool value = true; };

template<bool cond, typename T, typename F> struct SelectType
{
  using type = F;
};

template<typename T, typename F> struct SelectType<true, T, F>
{
  using type = T;
};

template<typename T>
constexpr T max(const T& a, const T& b){ return a > b ? a : b; }

Int log2(Int a)
{
  Int r = 0;
  while(a > 1)
  {
    r++;
    a >>= 1; 
  }

  return r;
}

void remove_redundant_dimensions(
  const Int* src, Int src_n, Int* dst, Int& dst_n)
{
  dst_n = 0;
  for(Int i = 0; i < src_n; i++)
    if(src[i] != 1)
    {
      dst[dst_n] = src[i];
      dst_n++;
    }

  if(dst_n == 0)
  {
    dst[dst_n] = 1;
    dst_n++;
  }
}

Int reverse_bits(Int a_in, Int nbits)
{
  if constexpr(sizeof(Int) == 8)
  {
    Uint a = a_in;
    Uint c1 =  0x5555555555555555ULL;
    Uint c2 =  0x3333333333333333ULL;
    Uint c4 =  0x0f0f0f0f0f0f0f0fULL;
    Uint c8 =  0x00ff00ff00ff00ffULL;
    Uint c16 = 0x0000ffff0000ffffULL;
    a = ((a >> 1) & c1) | ((a & c1) << 1);
    a = ((a >> 2) & c2) | ((a & c2) << 2);
    a = ((a >> 4) & c4) | ((a & c4) << 4);
    a = ((a >> 8) & c8) | ((a & c8) << 8);
    a = ((a >> 16) & c16) | ((a & c16) << 16);
    a = (a >> 32) | (a << 32);
    return Int(a >> (64 - nbits));
  }
  else
  {
    Uint a = a_in;
    Uint c1 =  0x55555555ULL;
    Uint c2 =  0x33333333ULL;
    Uint c4 =  0x0f0f0f0fULL;
    Uint c8 =  0x00ff00ffULL;
    a = ((a >> 1) & c1) | ((a & c1) << 1);
    a = ((a >> 2) & c2) | ((a & c2) << 2);
    a = ((a >> 4) & c4) | ((a & c4) << 4);
    a = ((a >> 8) & c8) | ((a & c8) << 8);
    a = (a >> 16) | (a << 16);
    return Int(a >> (32 - nbits));
  }
}

struct BitReversed
{
  Uint i;
  Uint br;
  Uint mask;

  FORCEINLINE BitReversed(Uint limit) : i(0), br(0), mask(limit - 1) { }

  FORCEINLINE void advance()
  {
    Uint br_mask = mask >> 1;
    for(Uint j = i; j & 1; j >>= 1) br_mask >>= 1;
    br_mask ^= mask;
    br ^= br_mask;
    i++;
  }
};

template<typename V>
FORCEINLINE typename V::Vec load(const typename V::T* ptr)
{
  return V::template load<true>(ptr);
}

template<typename V>
FORCEINLINE typename V::Vec unaligned_load(const typename V::T* ptr)
{
  return V::template load<false>(ptr);
}

template<typename V>
FORCEINLINE void store(typename V::Vec val, ET<V>* dst)
{
  V::template store<true>(val, dst);
}

template<typename V>
FORCEINLINE void unaligned_store(typename V::Vec val, ET<V>* dst)
{
  V::template store<false>(val, dst);
}

template<typename V>
FORCEINLINE void unaligned_copy(const ET<V>* src, Int n, ET<V>* dst)
{
  if(n > 0 && n < V::vec_size) return;

  ASSERT(i & (V::vec_size - 1) == 0);

  for(Int i = 0; i < n; i += V::vec_size) 
    unaligned_store<V>(unaligned_load<V>(src + i), dst + i);
}

template<typename T = char>
Int align_size(Int size)
{
  static_assert(align_bytes % sizeof(T) == 0, "");
  return (size + align_bytes / sizeof(T) - 1) & ~(align_bytes / sizeof(T) - 1);
};

static Int aligned_increment(Int sz, Int bytes) { return align_size(sz + bytes); }

template<typename T>
T* aligned_increment(T* ptr, Int bytes) { return (T*) align_size(Uint(ptr) + bytes); }

template<typename V>
struct Complex
{
  typedef typename V::Vec Vec;
  Vec re;
  Vec im;

  // mul_neg_i and adj need to be static methods because
  // MSVC refuses to inline them if they are non-static methods,
  // even though __forceinline is used.
  static FORCEINLINE constexpr Complex mul_neg_i(Complex a)
  {
    return {a.im, -a.re};
  }

  static FORCEINLINE constexpr Complex adj(Complex a)
  {
    return {a.re, -a.im};
  }

  static FORCEINLINE Complex load(const ET<V>* ptr)
  {
    return {
      V::template load<true>(ptr),
      V::template load<true>(ptr + V::vec_size)};
  }

  static FORCEINLINE Complex unaligned_load(const ET<V>* ptr)
  {
    return {
      V::template load<false>(ptr),
      V::template load<false>(ptr + V::vec_size)};
  }

  FORCEINLINE void store(ET<V>* ptr) const
  {
    V::template store<true>(re, ptr);
    V::template store<true>(im, ptr + V::vec_size);
  }

  FORCEINLINE void unaligned_store(ET<V>* ptr) const
  {
    V::template store<false>(re, ptr);
    V::template store<false>(im, ptr + V::vec_size);
  }
};

template<typename V>
FORCEINLINE constexpr Complex<V> operator+(Complex<V> a, Complex<V> b)
{
  return {a.re + b.re, a.im + b.im};
}

template<typename V>
FORCEINLINE constexpr Complex<V> operator-(Complex<V> a, Complex<V> b)
{
  return {a.re - b.re, a.im - b.im};
}

template<typename V>
FORCEINLINE constexpr Complex<V> operator*(Complex<V> a, Complex<V> b)
{
  return { V::msub(a.re, b.re, a.im * b.im), V::madd(a.re, b.im, a.im * b.re) };
}

template<typename V>
FORCEINLINE constexpr Complex<V> operator*(Complex<V> a, typename V::Vec b)
{
  return {a.re * b, a.im * b};
}

namespace complex_format
{
  struct Vec
  {
    static const Int idx_ratio = 2;

    template<bool aligned, typename V>
    static FORCEINLINE Complex<V> load(const ET<V>* re, const ET<V>*)
    {
      return {
        V::template load<aligned>(re),
        V::template load<aligned>(re + V::vec_size)};
    }

    template<bool aligned, typename V>
    static FORCEINLINE void store(Complex<V> a, ET<V>* re, ET<V>*)
    {
      V::template store<aligned>(a.re, re);
      V::template store<aligned>(a.im, re + V::vec_size);
    }

    template<typename V>
    static FORCEINLINE void stream_store(Complex<V> a, ET<V>* re, ET<V>*)
    {
      V::stream_store(a.re, re);
      V::stream_store(a.im, re + V::vec_size);
    }
  };

  template<bool strict_alignment>
  struct SplitImpl
  {
    static const Int idx_ratio = 1;

    template<bool aligned, typename V>
    static FORCEINLINE Complex<V> load(const ET<V>* re, const ET<V>* im)
    {
      return {
        V::template load<strict_alignment && aligned>(re),
        V::template load<strict_alignment && aligned>(im) };
    }

    template<bool aligned, typename V>
    static FORCEINLINE void store(Complex<V> a, ET<V>* re, ET<V>* im)
    {
      V::template store<strict_alignment && aligned>(a.re, re);
      V::template store<strict_alignment && aligned>(a.im, im);
    }

    template<typename V>
    static FORCEINLINE void stream_store(Complex<V> a, ET<V>* re, ET<V>* im)
    {
      V::stream_store(a.re, re);
      V::stream_store(a.im, im);
    }
  };

  template<bool strict_alignment>
  struct ScalImpl
  {
    static const Int idx_ratio = 2;

    template<bool aligned, typename V>
    static FORCEINLINE Complex<V> load(const ET<V>* re, const ET<V>*)
    {
      Complex<V> r;
      V::template load_deinterleaved<strict_alignment && aligned>(
        re, r.re, r.im);
      return r;
    }

    template<bool aligned, typename V>
    static FORCEINLINE void store(Complex<V> a, ET<V>* re, ET<V>*)
    {
      V::interleave(a.re, a.im, a.re, a.im);
      V::template store<strict_alignment && aligned>(a.re, re);
      V::template store<strict_alignment && aligned>(a.im, re + V::vec_size);
    }

    template<typename V>
    static FORCEINLINE void stream_store(Complex<V> a, ET<V>* re, ET<V>*)
    {
      V::interleave(a.re, a.im, a.re, a.im);
      V::stream_store(a.re, re);
      V::stream_store(a.im, re + V::vec_size);
    }
  };

  using AlignedScal = ScalImpl<true>;
  using AlignedSplit = SplitImpl<true>;

#ifdef STRICT_ALIGNMENT
  using Scal = ScalImpl<true>;
  using Split = SplitImpl<true>;
#else
  using Scal = ScalImpl<false>;
  using Split = SplitImpl<false>;
#endif

  template<class InputCf>
  struct Swapped
  {
    static const Int idx_ratio = InputCf::idx_ratio;

    template<bool aligned, typename V>
    static FORCEINLINE Complex<V> load(const ET<V>* re, const ET<V>* im)
    {
      auto a = InputCf::template load<aligned, V>(re, im);
      return {a.im, a.re};
    }

    template<bool aligned, typename V>
    static FORCEINLINE void store(Complex<V> a, ET<V>* re, ET<V>* im)
    {
      InputCf::template store<aligned, V>({a.im, a.re}, re, im);
    }

    template<typename V>
    static FORCEINLINE void stream_store(Complex<V> a, ET<V>* re, ET<V>* im)
    {
      InputCf::template stream_store<V>({a.im, a.re}, re, im);
    }
  };
}

template<typename V, typename Cf>
FORCEINLINE Complex<V> load(
  const typename V::T* re, const typename V::T* im, Int offset = 0)
{
  return Cf::template load<true, V>(re + offset, im + offset);
}

template<typename V, typename Cf>
FORCEINLINE Complex<V> unaligned_load(
  const typename V::T* re, const typename V::T* im, Int offset = 0)
{
  return Cf::template load<false, V>(re + offset, im + offset);
}

template<typename V, typename Cf>
FORCEINLINE constexpr Int stride() { return Cf::idx_ratio * V::vec_size; }

template<typename CF, typename V>
FORCEINLINE void store(
  Complex<V> val, ET<V>* dst_re, ET<V>* dst_im, Int offset = 0)
{
  CF::template store<true>(val, dst_re + offset, dst_im + offset);
}

template<typename CF, typename V>
FORCEINLINE void stream_store(
  Complex<V> val, ET<V>* dst_re, ET<V>* dst_im, Int offset = 0)
{
  CF::stream_store(val, dst_re + offset, dst_im + offset);
}

template<typename CF, typename V>
FORCEINLINE void unaligned_store(
  Complex<V> val, ET<V>* dst_re, ET<V>* dst_im, Int offset = 0)
{
  CF::template store<false>(val, dst_re + offset, dst_im + offset);
}

namespace cf = complex_format;

#define VEC_TYPEDEFS(V) \
  typedef typename V::T T; \
  typedef typename V::Vec Vec; \
  typedef Complex<V> C;

template<typename V, typename SrcCf, typename DstCf>
FORCEINLINE void complex_copy(
  const ET<V>* src_re, const ET<V>* src_im, Int n,
  typename V::T* dst_re, typename V::T* dst_im)
{
  for(Int s = 0, d = 0; s < n * SrcCf::idx_ratio;
      s += stride<V, SrcCf>(), d += stride<V, DstCf>())
  {
    store<DstCf>(
      load<V, SrcCf>(src_re, src_im, s),
      dst_re, dst_im, d);
  }
}

template<typename V>
FORCEINLINE Complex<V> reverse_complex(Complex<V> a)
{
  return { V::reverse(a.re), V::reverse(a.im) };
}


template<typename T>
struct SinCosTable { };

template<>
struct SinCosTable<float>
{
  constexpr static float sin[64] = {
    -0x1.777a5cp-24, 0x1p+0, 0x1.6a09e6p-1, 0x1.87de2cp-2, 
    0x1.8f8b84p-3, 0x1.917a6cp-4, 0x1.91f66p-5, 0x1.92156p-6, 
    0x1.921d2p-7, 0x1.921f1p-8, 0x1.921f8cp-9, 0x1.921facp-10, 
    0x1.921fb4p-11, 0x1.921fb6p-12, 0x1.921fb6p-13, 0x1.921fb6p-14, 
    0x1.921fb6p-15, 0x1.921fb6p-16, 0x1.921fb6p-17, 0x1.921fb6p-18, 
    0x1.921fb6p-19, 0x1.921fb6p-20, 0x1.921fb6p-21, 0x1.921fb6p-22, 
    0x1.921fb6p-23, 0x1.921fb6p-24, 0x1.921fb6p-25, 0x1.921fb6p-26, 
    0x1.921fb6p-27, 0x1.921fb6p-28, 0x1.921fb6p-29, 0x1.921fb6p-30, 
    0x1.921fb6p-31, 0x1.921fb6p-32, 0x1.921fb6p-33, 0x1.921fb6p-34, 
    0x1.921fb6p-35, 0x1.921fb6p-36, 0x1.921fb6p-37, 0x1.921fb6p-38, 
    0x1.921fb6p-39, 0x1.921fb6p-40, 0x1.921fb6p-41, 0x1.921fb6p-42, 
    0x1.921fb6p-43, 0x1.921fb6p-44, 0x1.921fb6p-45, 0x1.921fb6p-46, 
    0x1.921fb6p-47, 0x1.921fb6p-48, 0x1.921fb6p-49, 0x1.921fb6p-50, 
    0x1.921fb6p-51, 0x1.921fb6p-52, 0x1.921fb6p-53, 0x1.921fb6p-54, 
    0x1.921fb6p-55, 0x1.921fb6p-56, 0x1.921fb6p-57, 0x1.921fb6p-58, 
    0x1.921fb6p-59, 0x1.921fb6p-60, 0x1.921fb6p-61, 0x1.921fb6p-62};

  constexpr static float cos[64] = {
    -0x1p+0, -0x1.777a5cp-25, 0x1.6a09e6p-1, 0x1.d906bcp-1, 
    0x1.f6297cp-1, 0x1.fd88dap-1, 0x1.ff621ep-1, 0x1.ffd886p-1, 
    0x1.fff622p-1, 0x1.fffd88p-1, 0x1.ffff62p-1, 0x1.ffffd8p-1, 
    0x1.fffff6p-1, 0x1.fffffep-1, 0x1p+0, 0x1p+0, 
    0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, 
    0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, 
    0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, 
    0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, 
    0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, 
    0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, 
    0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, 
    0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, 
    0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, 
    0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, 
    0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, 
    0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0};
};

constexpr float SinCosTable<float>::sin[];
constexpr float SinCosTable<float>::cos[];

template<>
struct SinCosTable<double>
{
  constexpr static double sin[64] = {
    0x1.1a62633145c07p-53, 0x1p+0, 0x1.6a09e667f3bccp-1,
    0x1.87de2a6aea963p-2, 0x1.8f8b83c69a60ap-3, 0x1.917a6bc29b42cp-4,
    0x1.91f65f10dd814p-5, 0x1.92155f7a3667ep-6, 0x1.921d1fcdec784p-7,
    0x1.921f0fe670071p-8, 0x1.921f8becca4bap-9, 0x1.921faaee6472dp-10,
    0x1.921fb2aecb36p-11, 0x1.921fb49ee4ea6p-12, 0x1.921fb51aeb57bp-13,
    0x1.921fb539ecf31p-14, 0x1.921fb541ad59ep-15, 0x1.921fb5439d73ap-16,
    0x1.921fb544197ap-17, 0x1.921fb544387bap-18, 0x1.921fb544403c1p-19,
    0x1.921fb544422c2p-20, 0x1.921fb54442a83p-21, 0x1.921fb54442c73p-22,
    0x1.921fb54442cefp-23, 0x1.921fb54442d0ep-24, 0x1.921fb54442d15p-25,
    0x1.921fb54442d17p-26, 0x1.921fb54442d18p-27, 0x1.921fb54442d18p-28,
    0x1.921fb54442d18p-29, 0x1.921fb54442d18p-30, 0x1.921fb54442d18p-31,
    0x1.921fb54442d18p-32, 0x1.921fb54442d18p-33, 0x1.921fb54442d18p-34,
    0x1.921fb54442d18p-35, 0x1.921fb54442d18p-36, 0x1.921fb54442d18p-37,
    0x1.921fb54442d18p-38, 0x1.921fb54442d18p-39, 0x1.921fb54442d18p-40,
    0x1.921fb54442d18p-41, 0x1.921fb54442d18p-42, 0x1.921fb54442d18p-43,
    0x1.921fb54442d18p-44, 0x1.921fb54442d18p-45, 0x1.921fb54442d18p-46,
    0x1.921fb54442d18p-47, 0x1.921fb54442d18p-48, 0x1.921fb54442d18p-49,
    0x1.921fb54442d18p-50, 0x1.921fb54442d18p-51, 0x1.921fb54442d18p-52,
    0x1.921fb54442d18p-53, 0x1.921fb54442d18p-54, 0x1.921fb54442d18p-55,
    0x1.921fb54442d18p-56, 0x1.921fb54442d18p-57, 0x1.921fb54442d18p-58,
    0x1.921fb54442d18p-59, 0x1.921fb54442d18p-60, 0x1.921fb54442d18p-61,
    0x1.921fb54442d18p-62};

  constexpr static double cos[64] = {
    -0x1p+0, 0x1.1a62633145c07p-54, 0x1.6a09e667f3bcdp-1,
    0x1.d906bcf328d46p-1, 0x1.f6297cff75cbp-1, 0x1.fd88da3d12526p-1,
    0x1.ff621e3796d7ep-1, 0x1.ffd886084cd0dp-1, 0x1.fff62169b92dbp-1,
    0x1.fffd8858e8a92p-1, 0x1.ffff621621d02p-1, 0x1.ffffd88586ee6p-1,
    0x1.fffff62161a34p-1, 0x1.fffffd8858675p-1, 0x1.ffffff621619cp-1,
    0x1.ffffffd885867p-1, 0x1.fffffff62161ap-1, 0x1.fffffffd88586p-1,
    0x1.ffffffff62162p-1, 0x1.ffffffffd8858p-1, 0x1.fffffffff6216p-1,
    0x1.fffffffffd886p-1, 0x1.ffffffffff621p-1, 0x1.ffffffffffd88p-1,
    0x1.fffffffffff62p-1, 0x1.fffffffffffd9p-1, 0x1.ffffffffffff6p-1,
    0x1.ffffffffffffep-1, 0x1.fffffffffffffp-1, 0x1p+0,
    0x1p+0, 0x1p+0, 0x1p+0,
    0x1p+0, 0x1p+0, 0x1p+0,
    0x1p+0, 0x1p+0, 0x1p+0,
    0x1p+0, 0x1p+0, 0x1p+0,
    0x1p+0, 0x1p+0, 0x1p+0,
    0x1p+0, 0x1p+0, 0x1p+0,
    0x1p+0, 0x1p+0, 0x1p+0,
    0x1p+0, 0x1p+0, 0x1p+0,
    0x1p+0, 0x1p+0, 0x1p+0,
    0x1p+0, 0x1p+0, 0x1p+0,
    0x1p+0, 0x1p+0, 0x1p+0,
    0x1p+0};
};

constexpr double SinCosTable<double>::sin[];
constexpr double SinCosTable<double>::cos[];

template<typename T_>
struct Scalar
{
  typedef T_ T;
  typedef T_ Vec;
  static constexpr Int vec_size = 1;
  static constexpr bool prefer_three_passes = false;

  template<Int elements_per_vec>
  static FORCEINLINE void interleave_multi(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    r0 = a0;
    r1 = a1;
  }

  static FORCEINLINE void interleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    r0 = a0;
    r1 = a1;
  }

  static FORCEINLINE void deinterleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    r0 = a0;
    r1 = a1;
  }

  static FORCEINLINE Vec vec(T a){ return a; }
  static FORCEINLINE Vec reverse(Vec v) { return v; }

  static FORCEINLINE constexpr Vec madd(Vec a, Vec b, Vec c)
  {
    return a * b + c;
  }

  static FORCEINLINE constexpr Vec msub(Vec a, Vec b, Vec c)
  {
    return a * b - c;
  }

  template<bool aligned>
  static FORCEINLINE Vec load(const T* p) { return *p; }
  
  template<bool aligned>
  static FORCEINLINE void load_deinterleaved(const T* src, Vec& dst0, Vec& dst1)
  {
    dst0 = src[0];
    dst1 = src[1];
  }

  template<bool aligned>
  static FORCEINLINE void store(Vec val, T* p) { *p = val; }

  static FORCEINLINE void stream_store(Vec val, T* p) { *p = val; }
  static void sfence(){ }
};

template<typename V>
void store_two_pass_twiddle(Complex<V> first, typename V::T* dst)
{
  first.store(dst);
  auto second = first * first;
  auto third = second * first;
  second.store(dst + stride<V, cf::Vec>());
  third.store(dst + 2 * stride<V, cf::Vec>());
}

template<typename T>
void swap(T& a, T& b)
{
  T tmpa = a;
  T tmpb = b;
  a = tmpb;
  b = tmpa;
}

template<typename V>
void compute_twiddle_step(
  typename V::T* src_re, typename V::T* src_im,
  Int full_dst_size,
  Int dst_size,
  typename V::T* dst_re, typename V::T* dst_im)
{
  VEC_TYPEDEFS(V);
  if(V::vec_size != 1 && dst_size / 2 < V::vec_size)
    return compute_twiddle_step<Scalar<T>>(
      src_re, src_im, full_dst_size, dst_size, dst_re, dst_im);

  Int table_index = log2(full_dst_size);
  auto c = V::vec(SinCosTable<T>::cos[table_index]);
  auto s = V::vec(SinCosTable<T>::sin[table_index]);

  for(Int j = 0; j < dst_size / 2; j += V::vec_size)
  {
    Vec a_re = unaligned_load<V>(src_re + j);
    Vec a_im = unaligned_load<V>(src_im + j);
    Vec b_re = a_re * c + a_im * s;
    Vec b_im = a_im * c - a_re * s;

    Vec c_re, d_re;
    V::interleave(a_re, b_re, c_re, d_re);
    unaligned_store<V>(c_re, dst_re + 2 * j);
    unaligned_store<V>(d_re, dst_re + 2 * j + V::vec_size);

    Vec c_im, d_im;
    V::interleave(a_im, b_im, c_im, d_im);
    unaligned_store<V>(c_im, dst_im + 2 * j);
    unaligned_store<V>(d_im, dst_im + 2 * j + V::vec_size);
  }
}

template<typename V>
void compute_twiddle(Int n, typename V::T* dst_re, typename V::T* dst_im)
{
  VEC_TYPEDEFS(V);

  auto end_re = dst_re + n;
  auto end_im = dst_im + n;
  end_re[-1] = T(1); 
  end_im[-1] = T(0); 

  for(Int size = 2; size <= n; size *= 2)
    compute_twiddle_step<V>(
      end_re - size / 2, end_im - size / 2,
      size, size,
      end_re - size, end_im - size);
}

template<typename V>
void compute_twiddle_range(
  Int n, typename V::T* dst_re, typename V::T* dst_im)
{
  VEC_TYPEDEFS(V);

  auto end_re = dst_re + n;
  auto end_im = dst_im + n;

  end_re[-1] = T(0);
  end_im[-1] = T(0);
  end_re[-2] = T(1);
  end_im[-2] = T(0);

  for(Int size = 2; size < n; size *= 2)
    compute_twiddle_step<V>(
      end_re - size, end_im - size,
      size, size,
      end_re - 2 * size, end_im - 2 * size);
}

template<typename V>
Int twiddle_for_step_memsize(Int dft_size, Int npasses)
{
  VEC_TYPEDEFS(V);

  if(dft_size < V::vec_size || npasses < 1 || npasses > 3) return 0;

  Int m = 
    npasses == 1 ? 1 : 
    npasses == 2 ? 3 : 5;

  return sizeof(T) * dft_size * m * cf::Vec::idx_ratio;
}

template<typename V>
void twiddle_for_step_create(
  const typename V::T* twiddle_range,
  Int twiddle_range_n,
  Int dft_size,
  Int npasses,
  typename V::T* dst)
{
  VEC_TYPEDEFS(V);

  if(dft_size < V::vec_size || npasses < 1 || npasses > 3) return;

  Int n = twiddle_range_n;

  ASSERT(n >= (dft_size << npasses));

  Int dst_idx_ratio = cf::Vec::idx_ratio;
  Int dst_stride = stride<V, cf::Vec>();

  if(npasses == 3)
  {
    auto src_row0 = twiddle_range + (n - 4 * dft_size);
    auto src_row1 = twiddle_range + (n - 8 * dft_size);
    Int vdft_size = dft_size / V::vec_size;
    BitReversed br(vdft_size);
    for(; Int(br.i) < vdft_size; br.advance())
    {
      store_two_pass_twiddle<V>(
        load<V, cf::Split>(src_row0, src_row0 + n, br.i * V::vec_size),
        dst + 5 * br.br * dst_stride);

      load<V, cf::Split>(src_row1, src_row1 + n, br.i * V::vec_size)
        .store(dst + 5 * br.br * dst_stride + 3 * dst_stride);

      load<V, cf::Split>(src_row1, src_row1 + n,  br.i * V::vec_size + dft_size)
        .store(dst + 5 * br.br * dst_stride + 4 * dst_stride);
    }
  }
  else if(npasses == 2)
  {
    auto src_row0 = twiddle_range + (n - 4 * dft_size);
    Int vdft_size = dft_size / V::vec_size;
    BitReversed br(vdft_size);
    for(; Int(br.i) < vdft_size; br.advance())
    {
      store_two_pass_twiddle<V>(
        load<V, cf::Split>(src_row0, src_row0 + n, br.i * V::vec_size),
        dst + 3 * br.br * dst_stride);
    }
  }
  else if(npasses == 1)
  {
    auto src_row0 = twiddle_range + (n - 2 * dft_size);
    Int vdft_size = dft_size / V::vec_size;
    BitReversed br(vdft_size);
    for(; Int(br.i) < vdft_size; br.advance())
      load<V, cf::Split>(src_row0, src_row0 + n, br.i * V::vec_size)
        .store(dst + br.br * dst_stride);
  }
}

Int product(const Int* b, const Int* e)
{
  Int r = 1;
  for(; b < e; b++) r *= *b;
  return r;
}

Int product(const Int* p, Int n) { return product(p, p + n); }

}

#endif
