#ifndef FFT_COMMON_H
#define FFT_COMMON_H

#if defined __arm__
typedef int Int;
typedef unsigned Uint;
#define WORD_SIZE 32
#else
typedef long Int;
typedef unsigned long Uint;
#define WORD_SIZE 64
#endif

const Int max_int = Int(Uint(-1) >> 1);

//#define FORCEINLINE __attribute__((always_inline)) inline
#define FORCEINLINE
#define HOT __attribute__((hot))
#define NOINLINE __attribute__((noinline))

#define ASSERT(condition) ((condition) || *((volatile int*) 0))

#if 0

#define DEBUG_OUTPUT

#include <cstdio>
template<typename T_> void dump(T_* ptr, Int n, const char* name, ...);

template<typename T>
void print_vec(T a)
{
  for(Int i = 0; i < sizeof(T) / sizeof(float); i++)
    printf("%f ", ((float*)&a)[i]);

  printf("\n"); 
}
#endif

Int large_fft_size = 1 << 13;
Int optimal_size = 1 << 11;
Int max_vec_size = 8;
const Int align_bytes = 64;

template<typename T, typename U>
struct SameType { static const bool value = false; };

template<typename T>
struct SameType<T, T> { static const bool value = true; };

Int tiny_log2(Int a)
{
  return
    a == 1 ? 0 :
    a == 2 ? 1 :
    a == 4 ? 2 :
    a == 8 ? 3 :
    a == 16 ? 4 : -1;
}

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

void remove_ones(
  const Int* src, Int src_n, Int* dst, Int& dst_n)
{
  dst_n = 0;
  for(Int i = 0; i < src_n; i++)
    if(src[i] != 1)
    {
      dst[dst_n] = src[i];
      dst_n++;
    }
}

#if WORD_SIZE == 64
Int reverse_bits(Int a_in, Int nbits)
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
#else
Int reverse_bits(Int a_in, Int nbits)
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
#endif

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

template<typename T>
FORCEINLINE void copy(const T* src, Int n, T* dst)
{
#if defined __GNUC__ || defined __clang__
  __builtin_memmove(dst, src, n * sizeof(T));
#else
  for(Int i = 0; i < n; i++) dst[i] = src[i];
#endif
}

template<typename T = char>
Int align_size(Int size)
{
  static_assert(align_bytes % sizeof(T) == 0, "");
  return (size + align_bytes / sizeof(T) - 1) & ~(align_bytes / sizeof(T) - 1);
};

template<typename V>
Int tiny_twiddle_bytes()
{
  return sizeof(typename V::Vec) * 2 * tiny_log2(V::vec_size);
}

static Int aligned_increment(Int sz, Int bytes) { return align_size(sz + bytes); }

template<typename T>
T* aligned_increment(T* ptr, Int bytes) { return (T*) align_size(Uint(ptr) + bytes); }

template<typename V>
struct Complex
{
  typedef typename V::Vec Vec;
  Vec re;
  Vec im;
  FORCEINLINE Complex mul_neg_i() { return {im, -re}; }
  FORCEINLINE Complex adj() { return {re, -im}; }
  FORCEINLINE Complex operator+(Complex other)
  {
    return {re + other.re, im + other.im};
  }

  FORCEINLINE Complex operator-(Complex other)
  {
    return {re - other.re, im - other.im};
  }

  FORCEINLINE Complex operator*(Complex other)
  {
    return {
      re * other.re - im * other.im,
      re * other.im + im * other.re};
  }
  
  FORCEINLINE Complex operator*(Vec other)
  {
    return {re * other, im * other};
  }
};

namespace complex_format
{
  struct Split
  {
    static const Int idx_ratio = 1;

    template<typename V, Uint flags = 0>
    static FORCEINLINE Complex<V> load(typename V::T* ptr, Int off)
    {
      return {
        V::template load<flags>(ptr),
        V::template load<flags>(ptr + off)};
    }

    template<typename V>
    static FORCEINLINE Complex<V> unaligned_load(typename V::T* ptr, Int off)
    {
      return {
        V::unaligned_load(ptr),
        V::unaligned_load(ptr + off)};
    }

    template<Uint flags = 0, typename V>
    static FORCEINLINE void store(Complex<V> a, typename V::T* ptr, Int off)
    {
      V::template store<flags>(a.re, ptr);
      V::template store<flags>(a.im, ptr + off);
    }
    
    template<typename V>
    static FORCEINLINE void unaligned_store(
      Complex<V> a, typename V::T* ptr, Int off)
    {
      V::unaligned_store(a.re, ptr);
      V::unaligned_store(a.im, ptr + off);
    }
  };

  struct Vec
  {
    static const Int idx_ratio = 2;

    template<typename V, Uint flags = 0>
    static FORCEINLINE Complex<V> load(typename V::T* ptr, Int off)
    {
      return {
        V::template load<flags>(ptr),
        V::template load<flags>(ptr + V::vec_size)};
    }

    template<typename V>
    static FORCEINLINE Complex<V> unaligned_load(typename V::T* ptr, Int off)
    {
      return {V::unaligned_load(ptr), V::unaligned_load(ptr + V::vec_size)};
    }

    template<Uint flags = 0, typename V>
    static FORCEINLINE void store(Complex<V> a, typename V::T* ptr, Int off)
    {
      V::template store<flags>(a.re, ptr);
      V::template store<flags>(a.im, ptr + V::vec_size);
    }

    template<typename V>
    static FORCEINLINE void unaligned_store(
      Complex<V> a, typename V::T* ptr, Int off)
    {
      V::unaligned_store(a.re, ptr);
      V::unaligned_store(a.im, ptr + V::vec_size);
    }
  };

  struct Scal
  {
    static const Int idx_ratio = 2;

    template<typename V, Uint flags = 0>
    static FORCEINLINE Complex<V> load(typename V::T* ptr, Int off)
    {
      Complex<V> r;
      V::deinterleave(
        V::template load<flags>(ptr),
        V::template load<flags>(ptr + V::vec_size),
        r.re, r.im);

      return r;
    }

    template<typename V>
    static FORCEINLINE Complex<V> unaligned_load(typename V::T* ptr, Int off)
    {
      Complex<V> r;
      V::deinterleave(
        V::unaligned_load(ptr), V::unaligned_load(ptr + V::vec_size),
        r.re, r.im);

      return r;
    }

    template<Uint flags = 0, typename V>
    static FORCEINLINE void store(Complex<V> a, typename V::T* ptr, Int off)
    {
      V::interleave(a.re, a.im, a.re, a.im);
      V::template store<flags>(a.re, ptr);
      V::template store<flags>(a.im, ptr + V::vec_size);
    }

    template<typename V>
    static FORCEINLINE void unaligned_store(
      Complex<V> a, typename V::T* ptr, Int off)
    {
      V::interleave(a.re, a.im, a.re, a.im);
      V::unaligned_store(a.re, ptr);
      V::unaligned_store(a.im, ptr + V::vec_size);
    }
  };

  template<class InputCf>
  struct Swapped
  {
    static const Int idx_ratio = InputCf::idx_ratio;

    template<typename V, Uint flags = 0>
    static FORCEINLINE Complex<V> load(typename V::T* ptr, Int off)
    {
      auto a = InputCf::template load<V, flags>(ptr, off);
      return {a.im, a.re};
    }

    template<typename V>
    static FORCEINLINE Complex<V> unaligned_load(typename V::T* ptr, Int off)
    {
      auto a = InputCf::template unaligned_load<V>(ptr, off);
      return {a.im, a.re};
    }

    template<Uint flags = 0, typename V>
    static FORCEINLINE void store(Complex<V> a, typename V::T* ptr, Int off)
    {
      InputCf::template store<flags, V>({a.im, a.re}, ptr, off);
    }

    template<typename V>
    static FORCEINLINE void unaligned_store(
      Complex<V> a, typename V::T* ptr, Int off)
    {
      InputCf::template unaligned_store<V>({a.im, a.re}, ptr, off);
    }
  };
}

template<typename V, typename Cf, Uint flags = 0>
FORCEINLINE Complex<V> load(typename V::T* ptr, Int off)
{
  return Cf::template load<V, flags>(ptr, off);
}

template<typename V, typename Cf>
FORCEINLINE Int stride() { return Cf::idx_ratio * V::vec_size; }

namespace cf = complex_format;

#define VEC_TYPEDEFS(V) \
  typedef typename V::T T; \
  typedef typename V::Vec Vec; \
  typedef Complex<V> C;

template<typename V, typename SrcCf, typename DstCf>
FORCEINLINE void complex_copy(
  typename V::T* src, Int src_off, Int n,
  typename V::T* dst, Int dst_off)
{
  for(auto* end = src + n * SrcCf::idx_ratio; src < end;)
  {
    DstCf::store(load<V, SrcCf>(src, src_off), dst, dst_off);
    src += stride<V, SrcCf>();
    dst += stride<V, DstCf>();
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
  static float sin[64];
  static float cos[64];
};

float SinCosTable<float>::sin[64] = {
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
  
float SinCosTable<float>::cos[64] = {
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


constexpr Uint stream_flag = 1;

template<typename T_>
struct Scalar
{
  typedef T_ T;
  typedef T_ Vec;
  const static Int vec_size = 1;
  
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
  
  static void FORCEINLINE transpose(
    Vec a0, Vec a1, Vec a2, Vec a3,
    Vec& r0, Vec& r1, Vec& r2, Vec& r3)
  {
    r0 = a0;
    r1 = a1;
    r2 = a2;
    r3 = a3;
  }

  static void FORCEINLINE transpose(
    Vec a0, Vec a1, Vec a2, Vec a3,
    Vec a4, Vec a5, Vec a6, Vec a7,
    Vec& r0, Vec& r1, Vec& r2, Vec& r3,
    Vec& r4, Vec& r5, Vec& r6, Vec& r7)
  {
    r0 = a0;
    r1 = a1;
    r2 = a2;
    r3 = a3;
    r4 = a4;
    r5 = a5;
    r6 = a6;
    r7 = a7;
  }

  static Vec FORCEINLINE vec(T a){ return a; }
  
  static Vec reverse(Vec v)
  {
    return v;
  }

  template<Uint flags = 0> static Vec load(T* p) { return *p; }
  static Vec unaligned_load(T* p) { return *p; }
  template<Uint flags = 0> static void store(Vec val, T* p) { *p = val; }
  static void unaligned_store(Vec val, T* p) { *p = val; }
};

#ifdef __ARM_NEON__
#include <arm_neon.h>

struct Neon
{
  typedef float T;
  typedef float32x4_t Vec;
  const static Int vec_size = 4;
  
  template<Int elements_per_vec>
  static FORCEINLINE void interleave_multi(
    Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    if(elements_per_vec == 4)
    {
      auto r = vzipq_f32(a0, a1);
      r0 = r.val[0];
      r1 = r.val[1];    
    }
    else if(elements_per_vec == 2)
    {
      __asm__("vswp %f0, %e1" : "+w" (a0), "+w" (a1));
      r0 = a0;
      r1 = a1;
    }
  }

  static FORCEINLINE void interleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    auto r = vzipq_f32(a0, a1);
    r0 = r.val[0];
    r1 = r.val[1];
  }

  static FORCEINLINE void deinterleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    auto r = vuzpq_f32(a0, a1);
    r0 = r.val[0];
    r1 = r.val[1];
  }
  
  // The input matrix has 4 rows and vec_size columns
  // TODO: this needs to be updated to support different element order
  static void FORCEINLINE transpose(
    Vec a0, Vec a1, Vec a2, Vec a3,
    Vec& r0, Vec& r1, Vec& r2, Vec& r3)
  {
#if 1
    //this seems to be slightly faster
    __asm__(
      "vtrn.32 %q0, %q1\n"
      "vtrn.32 %q2, %q3\n"
      "vswp %f0, %e2\n"
      "vswp %f1, %e3\n"
      : "+w" (a0), "+w" (a1), "+w" (a2), "+w" (a3));
    r0 = a0;
    r1 = a1;
    r2 = a2;
    r3 = a3;
#else
    auto b01 = vtrnq_f32(a0, a1);
    auto b23 = vtrnq_f32(a2, a3);
    r0 = vcombine_f32(vget_low_f32(b01.val[0]), vget_low_f32(b23.val[0]));
    r2 = vcombine_f32(vget_high_f32(b01.val[0]), vget_high_f32(b23.val[0]));
    r1 = vcombine_f32(vget_low_f32(b01.val[1]), vget_low_f32(b23.val[1]));
    r3 = vcombine_f32(vget_high_f32(b01.val[1]), vget_high_f32(b23.val[1]));
#endif
  }

  static void FORCEINLINE transpose(
    Vec a0, Vec a1, Vec a2, Vec a3,
    Vec a4, Vec a5, Vec a6, Vec a7,
    Vec& r0, Vec& r1, Vec& r2, Vec& r3,
    Vec& r4, Vec& r5, Vec& r6, Vec& r7)
  {
    transpose(a0, a1, a2, a3, r0, r2, r4, r6);
    transpose(a4, a5, a6, a7, r1, r3, r5, r7);
  }

  static Vec FORCEINLINE vec(T a){ return vdupq_n_f32(a); }
  
  static Vec reverse(Vec v) { return v; } //TODO

  static Vec load(T* p) { return vld1q_f32(p); }
  static Vec unaligned_load(T* p) { return vld1q_f32(p); }
  template<Uint flags = 0>
  static void store(Vec val, T* p) { vst1q_f32(p, val); }
  static void unaligned_store(Vec val, T* p) { vst1q_f32(p, val); }
};
#endif

#ifdef __SSE2__
#include <immintrin.h>

struct SseFloat
{
  typedef float T;
  typedef __m128 Vec;
  const static Int vec_size = 4;
  
  template<Int elements_per_vec>
  static FORCEINLINE void interleave_multi(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    if(elements_per_vec == 4)
    {
      r0 = _mm_unpacklo_ps(a0, a1);
      r1 = _mm_unpackhi_ps(a0, a1);
    }
    if(elements_per_vec == 2)
    {
      r0 = _mm_shuffle_ps(a0, a1, _MM_SHUFFLE(1, 0, 1, 0));
      r1 = _mm_shuffle_ps(a0, a1, _MM_SHUFFLE(3, 2, 3, 2));
    }
  }

  static FORCEINLINE void interleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    r0 = _mm_unpacklo_ps(a0, a1);
    r1 = _mm_unpackhi_ps(a0, a1);
  }
  
  static FORCEINLINE void deinterleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    r0 = _mm_shuffle_ps(a0, a1, _MM_SHUFFLE(2, 0, 2, 0));
    r1 = _mm_shuffle_ps(a0, a1, _MM_SHUFFLE(3, 1, 3, 1));
  }
  
  static void FORCEINLINE transpose(
    Vec a0, Vec a1, Vec a2, Vec a3,
    Vec& r0, Vec& r1, Vec& r2, Vec& r3)
  {
    Vec b0 = _mm_shuffle_ps(a0, a1, _MM_SHUFFLE(1, 0, 1, 0)); 
    Vec b1 = _mm_shuffle_ps(a2, a3, _MM_SHUFFLE(1, 0, 1, 0));
    Vec b2 = _mm_shuffle_ps(a0, a1, _MM_SHUFFLE(3, 2, 3, 2)); 
    Vec b3 = _mm_shuffle_ps(a2, a3, _MM_SHUFFLE(3, 2, 3, 2));
     
    r0 = _mm_shuffle_ps(b0, b1, _MM_SHUFFLE(2, 0, 2, 0));
    r1 = _mm_shuffle_ps(b0, b1, _MM_SHUFFLE(3, 1, 3, 1));
    r2 = _mm_shuffle_ps(b2, b3, _MM_SHUFFLE(2, 0, 2, 0));
    r3 = _mm_shuffle_ps(b2, b3, _MM_SHUFFLE(3, 1, 3, 1));
  }
  
  static void FORCEINLINE transpose(
    Vec a0, Vec a1, Vec a2, Vec a3,
    Vec a4, Vec a5, Vec a6, Vec a7,
    Vec& r0, Vec& r1, Vec& r2, Vec& r3,
    Vec& r4, Vec& r5, Vec& r6, Vec& r7)
  {
    transpose(a0, a1, a2, a3, r0, r2, r4, r6);
    transpose(a4, a5, a6, a7, r1, r3, r5, r7);
  }
  
  static Vec FORCEINLINE vec(T a){ return _mm_set1_ps(a); }
  
  static Vec reverse(Vec v)
  {
    return _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 1, 2, 3));
  }

  template<Uint flags = 0>
  static Vec load(T* p)
  {
    return (flags & stream_flag) ? 
      _mm_castsi128_ps(_mm_stream_load_si128((__m128i*) p)) :
      _mm_load_ps(p);
  }

  static Vec unaligned_load(T* p) { return _mm_loadu_ps(p); }
  template<Uint flags = 0>
  static void store(Vec val, T* p)
  {
    if((flags & stream_flag))
      _mm_stream_ps(p, val);
    else
      _mm_store_ps(p, val);
  }

  static void unaligned_store(Vec val, T* p) { _mm_storeu_ps(p, val); }
};
#endif

#ifdef __AVX__
#include <immintrin.h>

struct AvxFloat
{
  typedef float T;
  typedef __v8sf Vec;
  const static Int vec_size = 8;

  template<Int elements_per_vec>
  static FORCEINLINE void interleave_multi(
    Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    if(elements_per_vec == 8)
    {
      transpose_128(
        _mm256_unpacklo_ps(a0, a1),
        _mm256_unpackhi_ps(a0, a1),
        r0, r1);
    }
    else if(elements_per_vec == 4)
      transpose_128(
        _mm256_shuffle_ps(a0, a1, _MM_SHUFFLE(1, 0, 1, 0)), 
        _mm256_shuffle_ps(a0, a1, _MM_SHUFFLE(3, 2, 3, 2)),
        r0, r1);
    else if (elements_per_vec == 2)
      transpose_128(a0, a1, r0, r1);
  }

  static FORCEINLINE void interleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    r0 = _mm256_unpacklo_ps(a0, a1);
    r1 = _mm256_unpackhi_ps(a0, a1);
    transpose_128(r0, r1, r0, r1);
  }

  static FORCEINLINE void deinterleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    transpose_128(a0, a1, a0, a1);
    r0 = _mm256_shuffle_ps(a0, a1, _MM_SHUFFLE(2, 0, 2, 0));
    r1 = _mm256_shuffle_ps(a0, a1, _MM_SHUFFLE(3, 1, 3, 1));
  }
  
  static void FORCEINLINE transpose(
    Vec a0, Vec a1, Vec a2, Vec a3,
    Vec& r0, Vec& r1, Vec& r2, Vec& r3)
  {
    Vec b0 = _mm256_shuffle_ps(a0, a1, _MM_SHUFFLE(1, 0, 1, 0)); 
    Vec b1 = _mm256_shuffle_ps(a2, a3, _MM_SHUFFLE(1, 0, 1, 0));
    Vec b2 = _mm256_shuffle_ps(a0, a1, _MM_SHUFFLE(3, 2, 3, 2)); 
    Vec b3 = _mm256_shuffle_ps(a2, a3, _MM_SHUFFLE(3, 2, 3, 2));
     
    Vec c0 = _mm256_shuffle_ps(b0, b1, _MM_SHUFFLE(2, 0, 2, 0));
    Vec c1 = _mm256_shuffle_ps(b0, b1, _MM_SHUFFLE(3, 1, 3, 1));
    Vec c2 = _mm256_shuffle_ps(b2, b3, _MM_SHUFFLE(2, 0, 2, 0));
    Vec c3 = _mm256_shuffle_ps(b2, b3, _MM_SHUFFLE(3, 1, 3, 1));

    r0 = _mm256_permute2f128_ps(c0, c1, _MM_SHUFFLE(0, 2, 0, 0));
    r1 = _mm256_permute2f128_ps(c2, c3, _MM_SHUFFLE(0, 2, 0, 0));
    r2 = _mm256_permute2f128_ps(c0, c1, _MM_SHUFFLE(0, 3, 0, 1));
    r3 = _mm256_permute2f128_ps(c2, c3, _MM_SHUFFLE(0, 3, 0, 1));
  }

  static void FORCEINLINE transpose(
    Vec a0, Vec a1, Vec a2, Vec a3,
    Vec a4, Vec a5, Vec a6, Vec a7,
    Vec& r0, Vec& r1, Vec& r2, Vec& r3,
    Vec& r4, Vec& r5, Vec& r6, Vec& r7)
  {
    transpose4x4_two(a0, a1, a2, a3);
    transpose4x4_two(a4, a5, a6, a7);

    transpose_128(a0, a4, r0, r4);
    transpose_128(a1, a5, r1, r5);
    transpose_128(a2, a6, r2, r6);
    transpose_128(a3, a7, r3, r7);
  }

  static Vec FORCEINLINE vec(T a){ return _mm256_set1_ps(a); }

//private: 
  static void FORCEINLINE transpose_128(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    r0 = _mm256_permute2f128_ps(a0, a1, _MM_SHUFFLE(0, 2, 0, 0)),
    r1 = _mm256_permute2f128_ps(a0, a1, _MM_SHUFFLE(0, 3, 0, 1));
  }
  
  static void transpose4x4_two(Vec& a0, Vec& a1, Vec& a2, Vec& a3)
  {
    Vec b0 = _mm256_unpacklo_ps(a0, a2);
    Vec b1 = _mm256_unpacklo_ps(a1, a3);
    Vec b2 = _mm256_unpackhi_ps(a0, a2);
    Vec b3 = _mm256_unpackhi_ps(a1, a3);
    a0 = _mm256_unpacklo_ps(b0, b1);
    a1 = _mm256_unpackhi_ps(b0, b1);
    a2 = _mm256_unpacklo_ps(b2, b3);
    a3 = _mm256_unpackhi_ps(b2, b3);
  }

  static Vec reverse(Vec v)
  {
    v = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(0, 1, 2, 3));
    return _mm256_permute2f128_ps(v, v, _MM_SHUFFLE(0, 0, 0, 1));
  }

  template<Uint flags = 0>
  static Vec load(T* p)
  {
    return (flags & stream_flag) ?
      _mm256_castsi256_ps(_mm256_stream_load_si256((__m256i*) p)) :
      _mm256_load_ps(p);
  }

  static Vec unaligned_load(T* p) { return _mm256_loadu_ps(p); }
  template<Uint flags = 0>
  static void store(Vec val, T* p)
  {
    if((flags & stream_flag))
      _mm256_stream_ps(p, val);
    else
      _mm256_store_ps(p, val);
  }

  static void unaligned_store(Vec val, T* p) { _mm256_storeu_ps(p, val); }
};
#endif

template<typename V>
void store_two_pass_twiddle(Complex<V> first, typename V::T* dst)
{
  cf::Vec::store(first, dst, 0);
  auto second = first * first;
  auto third = second * first;
  cf::Vec::store(second, dst + stride<V, cf::Vec>(), 0);
  cf::Vec::store(third, dst + 2 * stride<V, cf::Vec>(), 0);
}

template<typename T>
void swap(T& a, T& b)
{
  T tmpa = a;
  T tmpb = b;
  a = tmpb;
  b = tmpa;
}

template<typename T>
void compute_twiddle_step(
  T* src_re, T* src_im,
  Int full_dst_size,
  Int dst_size,
  T* dst_re, T* dst_im)
{
  Int table_index = log2(full_dst_size);
  auto c = SinCosTable<T>::cos[table_index];
  auto s = SinCosTable<T>::sin[table_index];

  for(Int j = 0; j < dst_size / 2; j++)
  {
    T re = src_re[j];
    T im = src_im[j];
    dst_re[2 * j] = re;
    dst_im[2 * j] = im;
    dst_re[2 * j + 1] = re * c + im * s;
    dst_im[2 * j + 1] = im * c - re * s;
  }
}

template<typename T>
void compute_twiddle(
  Int full_dst_size, Int dst_size,
  T* dst_re, T* dst_im)
{
  auto end_re = dst_re + dst_size;
  auto end_im = dst_im + dst_size;
  end_re[-1] = T(1); 
  end_im[-1] = T(0); 

  Int ratio = full_dst_size / dst_size;
  for(Int size = 2; size <= dst_size; size *= 2)
    compute_twiddle_step(
      end_re - size / 2, end_im - size / 2,
      size * ratio,
      size,
      end_re - size, end_im - size);
}

template<typename V, typename NumPassesCallback>
void init_twiddle(
  const NumPassesCallback& num_passes_callback,
  Int n,
  typename V::T* working,
  typename V::T* dst,
  typename V::T* tiny_dst)
{
  VEC_TYPEDEFS(V);
  if(n <= 2) return;

  auto end_re = dst + n;
  auto end_im = dst + 2 * n;
  end_re[-1] = T(0);
  end_im[-1] = T(0);
  end_re[-2] = T(1);
  end_im[-2] = T(0);

  for(Int size = 2; size < n; size *= 2)
    compute_twiddle_step(
      end_re - size, end_im - size,
      size, size,
      end_re - 2 * size, end_im - 2 * size);

  for(Int size = 2; size < V::vec_size; size *= 2)
  {
    auto re = tiny_dst + 2 * V::vec_size * tiny_log2(size);
    auto im = re + V::vec_size;

    for(Int j = 0; j < V::vec_size; j++)
    {
      re[j] = (end_re - 2 * size)[j & (size - 1)];
      im[j] = (end_im - 2 * size)[j & (size - 1)];
    }
  }

  for(Int si = 0, di = 0; si < n;
    si += stride<V, cf::Split>(), di += stride<V, cf::Vec>())
  {
    cf::Vec::store(load<V, cf::Split>(dst + si, n), working + di, 0);
  }

  copy(working, 2 * n, dst);

  // It's all in Vec format after this point
  typedef cf::Vec CF;
  
  for(Int dft_size = 1, s = 0; dft_size < n; s++)
  {
    Int npasses = num_passes_callback(s, dft_size);

		if(npasses == 5 && dft_size == 1)
		{
			Int ds = dft_size << 3;
			auto src_row0 = working + (n - 4 * ds) * CF::idx_ratio;
			auto dst_row0 = dst + (n - 4 * ds) * CF::idx_ratio;
			for(Int i = 0; i < ds * CF::idx_ratio; i += stride<V, CF>())
				store_two_pass_twiddle<V>(load<V, CF>(src_row0 + i, 0), dst_row0 + 3 * i);
		}
    else if(npasses == 3 && dft_size >= V::vec_size)
    {
      auto src_row0 = working + (n - 4 * dft_size) * CF::idx_ratio;
      auto src_row1 = working + (n - 8 * dft_size) * CF::idx_ratio;
      auto dst_row1 = dst + (n - 8 * dft_size) * CF::idx_ratio;
      Int vdft_size = dft_size / V::vec_size;
      BitReversed br(vdft_size);
      for(; br.i < vdft_size; br.advance())
      {
        store_two_pass_twiddle<V>(
          load<V, CF>(src_row0 + br.i * stride<V, CF>(), 0),
          dst_row1 + 5 * br.br * stride<V, CF>());

        CF::store(
          load<V, CF>(src_row1 + br.i * stride<V, CF>(), 0),
          dst_row1 + 5 * br.br * stride<V, CF>() + 3 * stride<V, CF>(), 0);
        
        CF::store(
          load<V, CF>(src_row1 + br.i * stride<V, CF>() + dft_size * CF::idx_ratio, 0),
          dst_row1 + 5 * br.br * stride<V, CF>() + 4 * stride<V, CF>(), 0);
      }
    }
    else if(npasses == 2 && dft_size >= V::vec_size)
    {
      auto src_row0 = working + (n - 4 * dft_size) * CF::idx_ratio;
      auto dst_row0 = dst + (n - 4 * dft_size) * CF::idx_ratio;
      Int vdft_size = dft_size / V::vec_size;
      BitReversed br(vdft_size);
      for(; br.i < vdft_size; br.advance())
      {
        store_two_pass_twiddle<V>(
          load<V, CF>(src_row0 + br.i * stride<V, CF>(), 0),
          dst_row0 + 3 * br.br * stride<V, CF>());
      }
    }
    else if(npasses == 1 && dft_size >= V::vec_size)
    {
      auto src_row0 = working + (n - 2 * dft_size) * CF::idx_ratio;
      auto dst_row0 = dst + (n - 2 * dft_size) * CF::idx_ratio;
      Int vdft_size = dft_size / V::vec_size;
      BitReversed br(vdft_size);
      for(; br.i < vdft_size; br.advance())
        CF::store(
          load<V, CF>(src_row0 + br.i * stride<V, CF>(), 0),
          dst_row0 + br.br * stride<V, CF>(),
          0);
    }

    dft_size <<= npasses;
  }
}

Int product(const Int* b, const Int* e)
{
  Int r = 1;
  for(; b < e; b++) r *= *b;
  return r;
}

Int product(const Int* p, Int n) { return product(p, p + n); }

#endif
