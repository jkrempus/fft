typedef long Int;
typedef unsigned long Uint;

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

//Int large_fft_size = 1 << 14;
Int large_fft_size = 1 << 14;
Int optimal_size = 1 << 10;
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

FORCEINLINE Int power2_div(Int a, Int b)
{
  Int r = 1;
  for(; b < a; b += b) r += r;
  return r;
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

struct IterateMultidim
{
  Int ndim;
  const Int* size;  
  Int idx[sizeof(Int) * 8];

  IterateMultidim(Int ndim, const Int* size) : ndim(ndim), size(size)
  {
    ASSERT(ndim < sizeof(idx) / sizeof(idx[0]));
    for(Int i = 0; i < ndim; i++) idx[i] = 0;
  }

  bool empty() { return idx[0] == size[0]; }

  void advance()
  {
    for(Int i = ndim - 1; i >= 0; i--)
    {
      idx[i]++;
      if(idx[i] < size[i]) return;
      idx[i] = 0;
    }

    idx[0] = size[0];
  }
};

template<typename T>
struct PtrRange
{
  T* begin_;
  T* end_;
  T* begin(){ return begin_; }
  T* end(){ return end_; }
};

template<typename T> PtrRange<T> ptr_range(T* p, Int size) { return {p, p + size}; }
template<typename T> PtrRange<T> ptr_range(T* b, T* e) { return {b, e}; }

template<typename T>
FORCEINLINE void copy(const T* src, Int n, T* dst)
{
#if defined __GNUC__ || defined __clang__
  __builtin_memmove(dst, src, n * sizeof(T));
#else
  for(Int i = 0; i < n; i++) dst[i] = src[i];
#endif
}

template<typename V, typename SrcCf, typename DstCf>
FORCEINLINE void complex_copy(
  typename V::T* src, Int src_off, Int n,
  typename V::T* dst, Int dst_off)
{
  for(Int i = 0; i < n; i++)
  {
    auto c = SrcCf::load(src + i * SrcCf::stride, src_off);
    DstCf::store(c, dst + i * DstCf::stride, dst_off);
  }
}

template<typename T>
struct Complex
{
  T re;
  T im;
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
  
  FORCEINLINE Complex operator*(T other)
  {
    return {re * other, im * other};
  }
};

namespace complex_format
{
  template<typename V>
  struct Split
  {
    typedef typename V::T T;
    typedef typename V::Vec Vec;
    typedef Complex<typename V::Vec> C;
    static const Int stride = V::vec_size;
    static const Int idx_ratio = 1;

    static FORCEINLINE C load(T* ptr, Int off)
    {
      return { V::load(ptr), V::load(ptr + off)};
    }
    
    static FORCEINLINE C unaligned_load(T* ptr, Int off)
    {
      return {V::unaligned_load(ptr), V::unaligned_load(ptr + off)};
    }

    static FORCEINLINE void store(C a, T* ptr, Int off)
    {
      V::store(a.re, ptr);
      V::store(a.im, ptr + off);
    }

    static FORCEINLINE void unaligned_store(C a, T* ptr, Int off)
    {
      V::unaligned_store(a.re, ptr);
      V::unaligned_store(a.im, ptr + off);
    }
  };

  template<typename V>
  struct Vec
  {
    typedef typename V::T T;
    typedef typename V::Vec Vec_;
    typedef Complex<typename V::Vec> C;
    static const Int stride = 2 * V::vec_size;
    static const Int idx_ratio = 2;

    static FORCEINLINE C load(T* ptr, Int off)
    {
      return {V::load(ptr), V::load(ptr + V::vec_size)};
    }

    static FORCEINLINE C unaligned_load(T* ptr, Int off)
    {
      return {V::unaligned_load(ptr), V::unaligned_load(ptr + V::vec_size)};
    }

    static FORCEINLINE void store(C a, T* ptr, Int off)
    {
      V::store(a.re, ptr);
      V::store(a.im, ptr + V::vec_size);
    }

    static FORCEINLINE void unaligned_store(C a, T* ptr, Int off)
    {
      V::unaligned_store(a.re, ptr);
      V::unaligned_store(a.im, ptr + V::vec_size);
    }
  };

  template<typename V>
  struct Scal
  {
    typedef typename V::T T;
    typedef typename V::Vec Vec;
    typedef Complex<typename V::Vec> C;
    static const Int stride = 2 * V::vec_size;
    static const Int idx_ratio = 2;

    static FORCEINLINE C load(T* ptr, Int off)
    {
      C r;
      V::deinterleave(
        V::load(ptr), V::load(ptr + V::vec_size),
        r.re, r.im);

      return r;
    }

    static FORCEINLINE C unaligned_load(T* ptr, Int off)
    {
      C r;
      V::deinterleave(
        V::unaligned_load(ptr), V::unaligned_load(ptr + V::vec_size),
        r.re, r.im);

      return r;
    }

    static FORCEINLINE void store(C a, T* ptr, Int off)
    {
      V::interleave(a.re, a.im, a.re, a.im);
      V::store(a.re, ptr);
      V::store(a.im, ptr + V::vec_size);
    }

    static FORCEINLINE void unaligned_store(C a, T* ptr, Int off)
    {
      V::interleave(a.re, a.im, a.re, a.im);
      V::unaligned_store(a.re, ptr);
      V::unaligned_store(a.im, ptr + V::vec_size);
    }
  };
  
  template<template<typename> class InputCf>
  struct Swapped
  {
    template<typename V>
    struct Cf
    {
      typedef typename InputCf<V>::T T;
      typedef typename InputCf<V>::Vec Vec;
      typedef Complex<Vec> C;
      static const Int stride = InputCf<V>::stride;
      static const Int idx_ratio = InputCf<V>::idx_ratio;

      static FORCEINLINE C load(T* ptr, Int off)
      {
        C a = InputCf<V>::load(ptr, off);
        return {a.im, a.re};
      }

      static FORCEINLINE C unaligned_load(T* ptr, Int off)
      {
        C a = InputCf<V>::unaligned_load(ptr, off);
        return {a.im, a.re};
      }

      static FORCEINLINE void store(C a, T* ptr, Int off)
      {
        InputCf<V>::store({a.im, a.re}, ptr, off);
      }

      static FORCEINLINE void unaligned_store(C a, T* ptr, Int off)
      {
        InputCf<V>::unaligned_store({a.im, a.re}, ptr, off);
      }
    };
  };
}

namespace cf = complex_format;

#define VEC_TYPEDEFS(V) \
  typedef typename V::T T; \
  typedef typename V::Vec Vec; \
  typedef Complex<typename V::Vec> C; \
  typedef cf::Vec<V> VecCf;

template<typename V>
Complex<typename V::Vec> reverse_complex(Complex<typename V::Vec> a)
{
  return { V::reverse(a.re), V::reverse(a.im) };
}

template<typename T>
struct Arg
{
  Int n;
  Int im_off;
  Int dft_size;
  Int start_offset;
  Int end_offset;
  T* src;
  T* twiddle;
  T* tiny_twiddle;
  T* dst;
  Int* br_table;
};

template<typename T>
struct Step
{
  typedef void (*pass_fun)(const Arg<T>&);
  short npasses;
  bool is_out_of_place;
  bool is_recursive;
  pass_fun fun_ptr;
};

template<typename T>
struct State
{
  Int n;
  Int im_off;
  T* working0;
  T* working1;
  T* twiddle;
  Int* br_table;
  T* tiny_twiddle;
  Step<T> steps[8 * sizeof(Int)];
  Int nsteps;
  Int ncopies;
  typedef void (*tiny_transform_fun_type)(T* src, T* dst, Int im_off);
  tiny_transform_fun_type tiny_transform_fun;
};

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

  static Vec load(T* p) { return *p; }
  static Vec unaligned_load(T* p) { return *p; }
  static void store(Vec val, T* p) { *p = val; }
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

  static Vec load(T* p) { return _mm_load_ps(p); }
  static Vec unaligned_load(T* p) { return _mm_loadu_ps(p); }
  static void store(Vec val, T* p) { _mm_store_ps(p, val); }
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
    Vec b0 = _mm256_shuffle_ps(a0, a1, _MM_SHUFFLE(1, 0, 1, 0));
    Vec b1 = _mm256_shuffle_ps(a2, a3, _MM_SHUFFLE(1, 0, 1, 0));
    Vec b2 = _mm256_shuffle_ps(a0, a1, _MM_SHUFFLE(3, 2, 3, 2));
    Vec b3 = _mm256_shuffle_ps(a2, a3, _MM_SHUFFLE(3, 2, 3, 2));

    a0 = _mm256_shuffle_ps(b0, b1, _MM_SHUFFLE(2, 0, 2, 0));
    a1 = _mm256_shuffle_ps(b0, b1, _MM_SHUFFLE(3, 1, 3, 1));
    a2 = _mm256_shuffle_ps(b2, b3, _MM_SHUFFLE(2, 0, 2, 0));
    a3 = _mm256_shuffle_ps(b2, b3, _MM_SHUFFLE(3, 1, 3, 1));
  }
  
  static Vec reverse(Vec v)
  {
    v = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(0, 1, 2, 3));
    return _mm256_permute2f128_ps(v, v, _MM_SHUFFLE(0, 0, 0, 1));
  }

  static Vec load(T* p) { return _mm256_load_ps(p); }
  static Vec unaligned_load(T* p) { return _mm256_loadu_ps(p); }
  static void store(Vec val, T* p) { _mm256_store_ps(p, val); }
  static void unaligned_store(Vec val, T* p) { _mm256_storeu_ps(p, val); }
};
#endif

Int twiddle_elements(Int npasses)
{
  return
    npasses == 1 ? 1 :
    npasses == 2 ? 3 :
    npasses == 3 ? 5 : max_int;
}

template<typename V>
void store_two_pass_twiddle(
  Complex<typename V::Vec> first,
  typename V::T* dst)
{
  cf::Vec<V>::store(first, dst, 0);
  auto second = first * first;
  auto third = second * first;
  cf::Vec<V>::store(second, dst + cf::Vec<V>::stride, 0);
  cf::Vec<V>::store(third, dst + 2 * cf::Vec<V>::stride, 0);
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
    si += cf::Split<V>::stride, di += cf::Vec<V>::stride)
  {
    cf::Vec<V>::store(cf::Split<V>::load(dst + si, n), working + di, 0);
  }

  copy(working, 2 * n, dst);

  // It's all in Vec format after this point
  typedef cf::Vec<V> CF;
  
  for(Int dft_size = 1, s = 0; dft_size < n; s++)
  {
    //printf("nsteps %d npasses %d\n", step.nsteps, step.npasses);
    Int npasses = num_passes_callback(s);

		if(npasses == 5 && dft_size == 1)
		{
			Int ds = dft_size << 3;
			auto src_row0 = working + (n - 4 * ds) * CF::idx_ratio;
			auto dst_row0 = dst + (n - 4 * ds) * CF::idx_ratio;
			for(Int i = 0; i < ds * CF::idx_ratio; i += CF::stride)
				store_two_pass_twiddle<V>(CF::load(src_row0 + i, 0), dst_row0 + 3 * i);
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
          CF::load(src_row0 + br.i * CF::stride, 0),
          dst_row1 + 5 * br.br * CF::stride);

        CF::store(
          CF::load(src_row1 + br.i * CF::stride, 0),
          dst_row1 + 5 * br.br * CF::stride + 3 * CF::stride, 0);
        
        CF::store(
          CF::load(src_row1 + br.i * CF::stride + dft_size * CF::idx_ratio, 0),
          dst_row1 + 5 * br.br * CF::stride + 4 * CF::stride, 0);
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
          CF::load(src_row0 + br.i * CF::stride, 0),
          dst_row0 + 3 * br.br * CF::stride);
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
          CF::load(src_row0 + br.i * CF::stride, 0),
          dst_row0 + br.br * CF::stride,
          0);
    }

    dft_size <<= npasses;
  }
}

template<typename V, typename DstCf>
void init_br_table(State<typename V::T>& state)
{
  if(state.steps[state.nsteps - 1].npasses == 0) return;
  auto len = (state.n / V::vec_size) >> state.steps[state.nsteps - 1].npasses;
  for(BitReversed br(len); br.i < len; br.advance())
    state.br_table[br.i] = br.br * DstCf::stride;
}

template<typename T> T min(T a, T b){ return a < b ? a : b; }
template<typename T> T max(T a, T b){ return a > b ? a : b; }

template<typename T> T sq(T a){ return a * a; }

template<typename V, Int dft_size, typename SrcCf>
void ct_dft_size_pass(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  Int n = arg.n;
  auto src0 = arg.src;
  auto src1 = arg.src + n * SrcCf::idx_ratio / 2;
  auto dst = arg.dst;
  auto tw = VecCf::load(arg.tiny_twiddle + tiny_log2(dft_size) * VecCf::stride, 0);
  for(auto end = src1; src0 < end;)
  {
    C a0 = SrcCf::load(src0, arg.im_off);
    C a1 = SrcCf::load(src1, arg.im_off);
    src0 += SrcCf::stride;
    src1 += SrcCf::stride;

    C b0, b1; 
    if(dft_size == 1)
    {
      b0 = a0 + a1;
      b1 = a0 - a1;
    }
    else
    {
      C mul = tw * a1;
      b0 = a0 + mul;
      b1 = a0 - mul;
    }

    const Int nelem = V::vec_size / dft_size;
    C d0, d1;
    V::template interleave_multi<nelem>(b0.re, b1.re, d0.re, d1.re);
    V::template interleave_multi<nelem>(b0.im, b1.im, d0.im, d1.im);
    VecCf::store(d0, dst, 0);
    VecCf::store(d1, dst + VecCf::stride, 0);

    dst += 2 * VecCf::stride;
  }
}

template<typename V, typename SrcCf>
void first_two_passes_impl(Int n, const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  Int l = n * SrcCf::idx_ratio / 4;
  T* src0 = arg.src;
  T* src1 = arg.src + l;
  T* src2 = arg.src + 2 * l;
  T* src3 = arg.src + 3 * l;
  T* dst = arg.dst;

  for(T* end = src1; src0 < end;)
  {
    C a0 = SrcCf::load(src0, arg.im_off);
    C a1 = SrcCf::load(src1, arg.im_off);
    C a2 = SrcCf::load(src2, arg.im_off);
    C a3 = SrcCf::load(src3, arg.im_off);
    src0 += SrcCf::stride;
    src1 += SrcCf::stride;
    src2 += SrcCf::stride;
    src3 += SrcCf::stride;

    C b0 = a0 + a2;
    C b1 = a0 - a2;
    C b2 = a1 + a3;
    C b3 = a1 - a3;

    C c0 = b0 + b2; 
    C c2 = b0 - b2;
    C c1 = b1 + b3.mul_neg_i();
    C c3 = b1 - b3.mul_neg_i();

    C d0, d1, d2, d3;
    V::transpose(c0.re, c1.re, c2.re, c3.re, d0.re, d1.re, d2.re, d3.re);
    V::transpose(c0.im, c1.im, c2.im, c3.im, d0.im, d1.im, d2.im, d3.im);

    VecCf::store(d0, dst, 0); 
    VecCf::store(d1, dst + VecCf::stride, 0); 
    VecCf::store(d2, dst + 2 * VecCf::stride, 0); 
    VecCf::store(d3, dst + 3 * VecCf::stride, 0); 
    dst += 4 * VecCf::stride;
  }
}

template<typename V, typename SrcCf>
void first_pass_scalar(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  auto src = arg.src;
  auto dst = arg.dst;
  auto n = arg.n;
  for(Int i0 = 0; i0 < n / 2; i0++)
  {
    Int i1 = i0 + n / 2;
    auto a0 = SrcCf::load(src + i0 * SrcCf::stride, arg.im_off);
    auto a1 = SrcCf::load(src + i1 * SrcCf::stride ,arg.im_off);
    VecCf::store(a0 + a1, dst + i0 * VecCf::stride, 0);
    VecCf::store(a0 - a1, dst + i1 * VecCf::stride, 0);
  }
}

template<typename V, typename SrcCf>
void first_two_passes(const Arg<typename V::T>& arg)
{
  first_two_passes_impl<V, SrcCf>(arg.n, arg);
}

template<typename V, typename SrcCf, Int n>
void first_two_passes_ct_size(const Arg<typename V::T>& arg)
{
  first_two_passes_impl<V, SrcCf>(n, arg);
}

template<typename V, typename SrcCf>
FORCEINLINE void first_three_passes_impl(
  Int n,
	Int dst_chunk_size,
  Int im_off,
  typename V::T* src,
  typename V::T* dst)
{
  VEC_TYPEDEFS(V);
  Int l = n / 8 * SrcCf::idx_ratio;
  Vec invsqrt2 = V::vec(SinCosTable<T>::cos[2]);

  for(T* end = dst + dst_chunk_size * VecCf::idx_ratio; dst < end;)
  {
    C c0, c1, c2, c3;
    {
      C a0 = SrcCf::load(src + 0 * l, im_off);
      C a1 = SrcCf::load(src + 2 * l, im_off);
      C a2 = SrcCf::load(src + 4 * l, im_off);
      C a3 = SrcCf::load(src + 6 * l, im_off);
      C b0 = a0 + a2;
      C b1 = a0 - a2;
      C b2 = a1 + a3;
      C b3 = a1 - a3;
      c0 = b0 + b2; 
      c2 = b0 - b2;
      c1 = b1 + b3.mul_neg_i();
      c3 = b1 - b3.mul_neg_i();
    }

    C mul0, mul1, mul2, mul3;
    {
      C a0 = SrcCf::load(src + 1 * l, im_off);
      C a1 = SrcCf::load(src + 3 * l, im_off);
      C a2 = SrcCf::load(src + 5 * l, im_off);
      C a3 = SrcCf::load(src + 7 * l, im_off);
      C b0 = a0 + a2;
      C b1 = a0 - a2;
      C b2 = a1 + a3;
      C b3 = a1 - a3;
      C c4 = b0 + b2;
      C c6 = b0 - b2;
      C c5 = b1 + b3.mul_neg_i();
      C c7 = b1 - b3.mul_neg_i();

      mul0 = c4;
      mul1 = {invsqrt2 * (c5.re + c5.im), invsqrt2 * (c5.im - c5.re)};
      mul2 = c6.mul_neg_i();
      mul3 = {invsqrt2 * (c7.im - c7.re), invsqrt2 * (-c7.im - c7.re)};
    }

    src += SrcCf::stride;

    {
      Vec d[8];
      V::transpose(
        c0.re + mul0.re, c1.re + mul1.re, c2.re + mul2.re, c3.re + mul3.re,
        c0.re - mul0.re, c1.re - mul1.re, c2.re - mul2.re, c3.re - mul3.re,
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);

      for(Int i = 0; i < 8; i++) V::store(d[i], dst + i * VecCf::stride);
    }

    {
      Vec d[8];
      V::transpose(
        c0.im + mul0.im, c1.im + mul1.im, c2.im + mul2.im, c3.im + mul3.im,
        c0.im - mul0.im, c1.im - mul1.im, c2.im - mul2.im, c3.im - mul3.im,
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);

      for(Int i = 0; i < 8; i++)
        V::store(d[i], dst + i * VecCf::stride + V::vec_size);
    }

    dst += 8 * VecCf::stride;
    if(src == end) break;
  }
}

template<typename V, typename SrcCf>
void first_three_passes(const Arg<typename V::T>& arg)
{
  first_three_passes_impl<V, SrcCf>(arg.n, arg.n, arg.im_off, arg.src, arg.dst);
}

template<typename V, typename SrcCf, Int n>
void first_three_passes_ct_size(const Arg<typename V::T>& arg)
{
  first_three_passes_impl<V, SrcCf>(n, n, arg.im_off, arg.src, arg.dst);
}

template<typename T>
FORCEINLINE void two_passes_inner(
  Complex<T> src0, Complex<T> src1, Complex<T> src2, Complex<T> src3,
  Complex<T>& dst0, Complex<T>& dst1, Complex<T>& dst2, Complex<T>& dst3,
  Complex<T> tw0, Complex<T> tw1, Complex<T> tw2)
{
  typedef Complex<T> C;
  C mul0 =       src0;
  C mul1 = tw0 * src1;
  C mul2 = tw1 * src2;
  C mul3 = tw2 * src3;

  C sum02 = mul0 + mul2;
  C dif02 = mul0 - mul2;
  C sum13 = mul1 + mul3;
  C dif13 = mul1 - mul3;

  dst0 = sum02 + sum13;
  dst2 = sum02 - sum13;
  dst1 = dif02 + dif13.mul_neg_i();
  dst3 = dif02 - dif13.mul_neg_i();
}

template<typename V>
void two_passes(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  Int n = arg.n;
  Int dft_size = arg.dft_size;
  auto src = arg.src;

  auto off1 = (arg.n >> log2(dft_size)) / 4 * VecCf::stride;
  auto off2 = off1 + off1;
  auto off3 = off2 + off1;
  
  auto start = arg.start_offset * VecCf::idx_ratio;
  auto end = arg.end_offset * VecCf::idx_ratio;
  
  auto tw = arg.twiddle + VecCf::idx_ratio * (n - 4 * dft_size);
  if(start != 0)
    tw += 3 * VecCf::stride * (start >> log2(off1 + off3));

  for(auto p = src + start; p < src + end;)
  {
    auto tw0 = VecCf::load(tw, 0);
    auto tw1 = VecCf::load(tw + VecCf::stride, 0);
    auto tw2 = VecCf::load(tw + 2 * VecCf::stride, 0);
    tw += 3 * VecCf::stride;

    for(auto end1 = p + off1;;)
    {
      ASSERT(p >= arg.src);
      ASSERT(p + off3 < arg.src + arg.n * VecCf::idx_ratio);

      C d0, d1, d2, d3;
      two_passes_inner(
        VecCf::load(p, 0), VecCf::load(p + off1, 0),
        VecCf::load(p + off2, 0), VecCf::load(p + off3, 0),
        d0, d1, d2, d3, tw0, tw1, tw2);

      VecCf::store(d0, p, 0);
      VecCf::store(d2, p + off1, 0);
      VecCf::store(d1, p + off2, 0);
      VecCf::store(d3, p + off3, 0);

      p += VecCf::stride;
      if(!(p < end1)) break;
    }

    p += off3;
  }
}

template<typename V, typename DstCf>
void last_two_passes_impl(Int n, Int* br, const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  Int vn = n / V::vec_size;
  auto tw = arg.twiddle;

  auto src = arg.src;
  
  auto dst0 = arg.dst; 
  auto dst1 = dst0 + vn / 4 * DstCf::stride; 
  auto dst2 = dst1 + vn / 4 * DstCf::stride; 
  auto dst3 = dst2 + vn / 4 * DstCf::stride; 

  for(auto end = src + vn * VecCf::stride; src < end; )
  {
    auto tw0 = VecCf::load(tw, 0);
    auto tw1 = VecCf::load(tw + VecCf::stride, 0);
    auto tw2 = VecCf::load(tw + 2 * VecCf::stride, 0);
    tw += 3 * VecCf::stride;

    C d0, d1, d2, d3;
    two_passes_inner(
      VecCf::load(src, 0),
      VecCf::load(src + VecCf::stride, 0),
      VecCf::load(src + 2 * VecCf::stride, 0),
      VecCf::load(src + 3 * VecCf::stride, 0),
      d0, d1, d2, d3, tw0, tw1, tw2);

    src += 4 * VecCf::stride;

    Int d = br[0];
    br++;
    DstCf::store(d0, dst0 + d, arg.im_off);
    DstCf::store(d1, dst1 + d, arg.im_off);
    DstCf::store(d2, dst2 + d, arg.im_off);
    DstCf::store(d3, dst3 + d, arg.im_off);
  }
}

template<typename V, typename DstCf>
void last_two_passes(const Arg<typename V::T>& arg)
{
  last_two_passes_impl<V, DstCf>(arg.n, arg.br_table, arg);
}

template<typename V, typename DstCf, Int n>
void last_two_passes(const Arg<typename V::T>& arg)
{
  const Int len = n / V::vec_size / 4;
  Int br_table[len];
  for(BitReversed br(len); br.i < len; br.advance())
    br_table[br.i] = br.br * DstCf::stride;

  last_two_passes_impl<V, DstCf>(n, br_table, arg);
}

template<typename V, typename DstCf>
FORCEINLINE void last_pass_impl(Int n, Int* br, const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  Int vn = n / V::vec_size;
  auto tw = arg.twiddle;

  auto src = arg.src;
  auto dst0 = arg.dst; 
  auto dst1 = dst0 + vn / 2 * DstCf::stride; 

  for(auto end = src + vn * VecCf::stride; src < end; )
  {
    C a0 = VecCf::load(src, 0);
    C mul = VecCf::load(src + VecCf::stride, 0) * VecCf::load(tw, 0);
    tw += VecCf::stride;
    src += 2 * VecCf::stride;

    Int d = br[0];
    br++;
    DstCf::store(a0 + mul, dst0 + d, arg.im_off);
    DstCf::store(a0 - mul, dst1 + d, arg.im_off);
  }
}

template<typename V, typename DstCf>
void last_pass(const Arg<typename V::T>& arg)
{
  last_pass_impl<V, DstCf>(arg.n, arg.br_table, arg);
}

template<typename V, typename DstCf, Int n>
void last_pass(const Arg<typename V::T>& arg)
{
  const Int len = n / V::vec_size / 2;
  Int br_table[len];
  for(BitReversed br(len); br.i < len; br.advance())
    br_table[br.i] = br.br * DstCf::stride;

  last_pass_impl<V, DstCf>(n, br_table, arg);
}

template<Int sz, Int alignment>
struct AlignedMemory
{
  char mem[sz + (alignment - 1)];
  void* get() { return (void*)((Uint(mem) + alignment - 1) & ~(alignment - 1)); }
};

template<typename V, typename DstCf>
void bit_reverse_pass(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);

  Int vn = arg.n / V::vec_size;
  Int im_off = arg.im_off;
  const Int m = 16;
  if(vn < m * m)
  {
    for(BitReversed br(vn); br.i < vn; br.advance())
      DstCf::store(
        VecCf::load(arg.src + br.i * VecCf::stride, 0),
        arg.dst + br.br * DstCf::stride,
        im_off);
  }
  else
  {
    const Int br_table[m] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
    AlignedMemory<m * m * VecCf::stride * sizeof(T), align_bytes> mem;
    auto working = (T*) mem.get();
    Int stride = vn / m;

    for(BitReversed br(vn / (m * m)); br.i < vn / (m * m); br.advance())
    {
      T* src = arg.src + br.i * m * VecCf::stride;
      for(Int i0 = 0; i0 < m; i0++)
        for(Int i1 = 0; i1 < m; i1++)
          VecCf::store(VecCf::load(src + (i0 * stride + i1) * VecCf::stride, 0),
            working + (i0 * m + i1) * VecCf::stride, 0);

      T* dst = arg.dst + br.br * m * DstCf::stride;
      for(Int i0 = 0; i0 < m; i0++)
      {
        T* s = working + br_table[i0] * VecCf::stride;
        T* d = dst + i0 * stride * DstCf::stride;
        for(Int i1 = 0; i1 < m; i1++)
          DstCf::store(VecCf::load(s + (br_table[i1] << 4) * VecCf::stride, 0),
            d + i1 * DstCf::stride, im_off);
      }
    }
  }
}

template<typename V, typename DstCf>
FORCEINLINE void last_three_passes_impl(
  Int n,
  Int im_off,
  typename V::T* src,
  typename V::T* twiddle,
  Int* br_table,
  typename V::T* dst)
{
  VEC_TYPEDEFS(V);
  Int l1 = n / 8 / V::vec_size;
  Int l2 = 2 * l1;
  Int l3 = 3 * l1;
  Int l4 = 4 * l1;
  Int l5 = 5 * l1;
  Int l6 = 6 * l1;
  Int l7 = 7 * l1;

  auto s = src;
  auto br = br_table;
  for(auto end = br + l1; br < end; br++)
  {
    auto d = dst + br[0];

    C a0, a1, a2, a3, a4, a5, a6, a7;
    {
      C tw0 = VecCf::load(twiddle, 0);
      C tw1 = VecCf::load(twiddle + VecCf::stride, 0);
      C tw2 = VecCf::load(twiddle + 2 * VecCf::stride, 0);

      {
        C mul0 =       VecCf::load(s, 0);
        C mul1 = tw0 * VecCf::load(s + 2 * VecCf::stride, 0);
        C mul2 = tw1 * VecCf::load(s + 4 * VecCf::stride, 0);
        C mul3 = tw2 * VecCf::load(s + 6 * VecCf::stride, 0);

        C sum02 = mul0 + mul2;
        C dif02 = mul0 - mul2;
        C sum13 = mul1 + mul3;
        C dif13 = mul1 - mul3;

        a0 = sum02 + sum13; 
        a1 = dif02 + dif13.mul_neg_i();
        a2 = sum02 - sum13;
        a3 = dif02 - dif13.mul_neg_i();
      }

      {
        C mul0 =       VecCf::load(s + 1 * VecCf::stride, 0);
        C mul1 = tw0 * VecCf::load(s + 3 * VecCf::stride, 0);
        C mul2 = tw1 * VecCf::load(s + 5 * VecCf::stride, 0);
        C mul3 = tw2 * VecCf::load(s + 7 * VecCf::stride, 0);

        C sum02 = mul0 + mul2;
        C dif02 = mul0 - mul2;
        C sum13 = mul1 + mul3;
        C dif13 = mul1 - mul3;

        a4 = sum02 + sum13;
        a5 = dif02 + dif13.mul_neg_i();
        a6 = sum02 - sum13;
        a7 = dif02 - dif13.mul_neg_i();
      }
    }

    {
      C tw3 = VecCf::load(twiddle + 3 * VecCf::stride, 0);
      {
        auto mul = tw3 * a4;
        DstCf::store(a0 + mul, d + 0, im_off);
        DstCf::store(a0 - mul, d + l4 * DstCf::stride, im_off);
      }

      {
        auto mul = tw3.mul_neg_i() * a6;
        DstCf::store(a2 + mul, d + l2 * DstCf::stride, im_off);
        DstCf::store(a2 - mul, d + l6 * DstCf::stride, im_off);
      }
    }

    {
      C tw4 = VecCf::load(twiddle + 4 * VecCf::stride, 0);
      {
        auto mul = tw4 * a5;
        DstCf::store(a1 + mul, d + l1 * DstCf::stride, im_off);
        DstCf::store(a1 - mul, d + l5 * DstCf::stride, im_off);
      }

      {
        auto mul = tw4.mul_neg_i() * a7;
        DstCf::store(a3 + mul, d + l3 * DstCf::stride, im_off);
        DstCf::store(a3 - mul, d + l7 * DstCf::stride, im_off);
      }
    }

    s += 8 * VecCf::stride;
    twiddle += 5 * VecCf::stride;
  }
}

template<typename V, typename DstCf>
void last_three_passes_vec(const Arg<typename V::T>& arg)
{
  last_three_passes_impl<V, DstCf>(
    arg.n, arg.im_off, arg.src, arg.twiddle, arg.br_table, arg.dst);
}

template<typename V, typename DstCf, Int n>
void last_three_passes_vec_ct_size(const Arg<typename V::T>& arg)
{
  last_three_passes_impl<V, DstCf>(
    n, arg.im_off, arg.src, arg.twiddle, arg.br_table, arg.dst);
}

template<typename V>
void last_three_passes_in_place(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  Int n = arg.n;
  Int im_off = arg.im_off;
  
  auto start = arg.start_offset * VecCf::idx_ratio;
  auto end = arg.end_offset * VecCf::idx_ratio;
  
  auto src = arg.src;
  T* twiddle = arg.twiddle + start * 5 / 8;

  for(auto p = src + start; p < src + end;)
  {
    C a0, a1, a2, a3, a4, a5, a6, a7;
    {
      C tw0 = VecCf::load(twiddle, 0);
      C tw1 = VecCf::load(twiddle + VecCf::stride, 0);
      C tw2 = VecCf::load(twiddle + 2 * VecCf::stride, 0);

      {
        C mul0 =       VecCf::load(p, 0);
        C mul1 = tw0 * VecCf::load(p + 2 * VecCf::stride, 0);
        C mul2 = tw1 * VecCf::load(p + 4 * VecCf::stride, 0);
        C mul3 = tw2 * VecCf::load(p + 6 * VecCf::stride, 0);

        C sum02 = mul0 + mul2;
        C dif02 = mul0 - mul2;
        C sum13 = mul1 + mul3;
        C dif13 = mul1 - mul3;

        a0 = sum02 + sum13; 
        a1 = dif02 + dif13.mul_neg_i();
        a2 = sum02 - sum13;
        a3 = dif02 - dif13.mul_neg_i();
      }

      {
        C mul0 =       VecCf::load(p + 1 * VecCf::stride, 0);
        C mul1 = tw0 * VecCf::load(p + 3 * VecCf::stride, 0);
        C mul2 = tw1 * VecCf::load(p + 5 * VecCf::stride, 0);
        C mul3 = tw2 * VecCf::load(p + 7 * VecCf::stride, 0);

        C sum02 = mul0 + mul2;
        C dif02 = mul0 - mul2;
        C sum13 = mul1 + mul3;
        C dif13 = mul1 - mul3;

        a4 = sum02 + sum13;
        a5 = dif02 + dif13.mul_neg_i();
        a6 = sum02 - sum13;
        a7 = dif02 - dif13.mul_neg_i();
      }
    }

    {
      C tw3 = VecCf::load(twiddle + 3 * VecCf::stride, 0);
      {
        auto mul = tw3 * a4;
        VecCf::store(a0 + mul, p + 0, 0);
        VecCf::store(a0 - mul, p + 1 * VecCf::stride, 0);
      }

      {
        auto mul = tw3.mul_neg_i() * a6;
        VecCf::store(a2 + mul, p + 2 * VecCf::stride, 0);
        VecCf::store(a2 - mul, p + 3 * VecCf::stride, 0);
      }
    }

    {
      C tw4 = VecCf::load(twiddle + 4 * VecCf::stride, 0);
      {
        auto mul = tw4 * a5;
        VecCf::store(a1 + mul, p + 4 * VecCf::stride, 0);
        VecCf::store(a1 - mul, p + 5 * VecCf::stride, 0);
      }

      {
        auto mul = tw4.mul_neg_i() * a7;
        VecCf::store(a3 + mul, p + 6 * VecCf::stride, 0);
        VecCf::store(a3 - mul, p + 7 * VecCf::stride, 0);
      }
    }

    p += 8 * VecCf::stride;
    twiddle += 5 * VecCf::stride;
  }
}

template<typename V> void null_pass(const Arg<typename V::T>& arg) { }

template<typename V, typename SrcCf, typename DstCf, Int n>
void tiny_transform(typename V::T* src, typename V::T* dst, Int im_off)
{
  VEC_TYPEDEFS(V);
  if(n == 1)
  {
    DstCf::store(SrcCf::load(src, im_off), dst, im_off);
  }
  else if(n == 2)
  {
    auto a0 = SrcCf::load(src, im_off);
    auto a1 = SrcCf::load(src + SrcCf::stride, im_off);
    DstCf::store(a0 + a1, dst, im_off);
    DstCf::store(a0 - a1, dst + DstCf::stride, im_off);
  }
}

template<typename V, typename SrcCf, typename DstCf>
void init_steps(State<typename V::T>& state)
{
  VEC_TYPEDEFS(V);
  Int step_index = 0;
  state.ncopies = 0;

  if(state.n <= 2)
  {
    state.nsteps = 0;
    state.tiny_transform_fun = 
      state.n == 0 ?  &tiny_transform<V, SrcCf, DstCf, 0> :
      state.n == 1 ?  &tiny_transform<V, SrcCf, DstCf, 1> :
      state.n == 2 ?  &tiny_transform<V, SrcCf, DstCf, 2> : nullptr;

    return;
  }
  else
    state.tiny_transform_fun = nullptr;

  for(Int dft_size = 1; dft_size < state.n; step_index++)
  {
    Step<T> step;
    step.is_out_of_place = true;
    step.is_recursive = false;

		if(dft_size == 1 && state.n >= 8 * V::vec_size && V::vec_size == 8)
    {
      if(state.n == 8 * V::vec_size)
        step.fun_ptr = &first_three_passes_ct_size<V, SrcCf, 8 * V::vec_size>;
      else
        step.fun_ptr = &first_three_passes<V, SrcCf>;

      step.npasses = 3;
    }
    else if(dft_size == 1 && state.n >= 4 * V::vec_size && V::vec_size >= 4)
    {
      if(state.n == 4 * V::vec_size)
        step.fun_ptr = &first_two_passes_ct_size<V, SrcCf, 4 * V::vec_size>;
      else if(state.n == 8 * V::vec_size)
        step.fun_ptr = &first_two_passes_ct_size<V, SrcCf, 8 * V::vec_size>;
      else
        step.fun_ptr = &first_two_passes<V, SrcCf>;

      step.npasses = 2;
    }
    else if(dft_size == 1 && V::vec_size == 1)
    {
      step.fun_ptr = &first_pass_scalar<V, SrcCf>;
      step.npasses = 1;
    }
    else if(dft_size >= V::vec_size)
    {
      if(state.n < large_fft_size && dft_size * 8 == state.n)
      {
        if(state.n == V::vec_size * 8)
          step.fun_ptr = &last_three_passes_vec_ct_size<V, DstCf, V::vec_size * 8>;
        else
          step.fun_ptr = &last_three_passes_vec<V, DstCf>;

        step.npasses = 3;
      }
      else if(dft_size * 8 == state.n)
      {
        step.fun_ptr = &last_three_passes_in_place<V>;
        step.npasses = 3;
        step.is_out_of_place = false;
        step.is_recursive = true;
      }
      else if(state.n < large_fft_size && dft_size * 4 == state.n)
      {
        if(state.n == V::vec_size * 4)
          step.fun_ptr = &last_two_passes<V, DstCf, V::vec_size * 4>;
        else if(state.n == V::vec_size * 8)
          step.fun_ptr = &last_two_passes<V, DstCf, V::vec_size * 8>;
        else
          step.fun_ptr = &last_two_passes<V, DstCf>;

        step.npasses = 2;
      }
      else if(dft_size * 4 <= state.n)
      {
        step.fun_ptr = &two_passes<V>;
        step.npasses = 2;
        step.is_out_of_place = false;
        step.is_recursive = state.n >= large_fft_size;
      }
      else
      {
        if(state.n == 2 * V::vec_size)
          step.fun_ptr = &last_pass<V, DstCf, 2 * V::vec_size>;
        else if(state.n == 4 * V::vec_size)
          step.fun_ptr = &last_pass<V, DstCf, 4 * V::vec_size>;
        else if(state.n == 8 * V::vec_size)
          step.fun_ptr = &last_pass<V, DstCf, 8 * V::vec_size>;
        else
          step.fun_ptr = &last_pass<V, DstCf>;

        step.npasses = 1;
      }
    }
    else
    {
      if(V::vec_size > 1 && dft_size == 1)
        step.fun_ptr = &ct_dft_size_pass<V, 1, SrcCf>;
      else if(V::vec_size > 2 && dft_size == 2)
        step.fun_ptr = &ct_dft_size_pass<V, 2, cf::Vec<V>>;
      else if(V::vec_size > 4 && dft_size == 4)
        step.fun_ptr = &ct_dft_size_pass<V, 4, cf::Vec<V>>;
      else if(V::vec_size > 8 && dft_size == 8)
        step.fun_ptr = &ct_dft_size_pass<V, 8, cf::Vec<V>>;

      step.npasses = 1;
    }

    state.steps[step_index] = step;
    if(step.is_out_of_place) state.ncopies++;
    dft_size <<= step.npasses;
  }

  if(state.n >= large_fft_size)
  {
    Step<T> step;
    step.npasses = 0;
    step.is_out_of_place = true;
    step.is_recursive = false;
    step.fun_ptr = &bit_reverse_pass<V, DstCf>;
    state.steps[step_index] = step;
    state.ncopies++;
    step_index++;
  }

  state.nsteps = step_index;

#ifdef DEBUG_OUTPUT
  for(Int i = 0; i < state.nsteps; i++)
    printf("npasses %d\n", state.steps[i].npasses);
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

template<typename V>
Int state_struct_offset(Int n)
{
  VEC_TYPEDEFS(V);
  return align_size(
    sizeof(T) * 2 * n +                            //working0
    sizeof(T) * 2 * n +                            //working1
    sizeof(T) * 2 * n +                            //twiddle
    tiny_twiddle_bytes<V>() +                      //tiny_twiddle
    sizeof(Int) * n / V::vec_size / 2);            //br table
}

template<typename V>
Int fft_state_memory_size(Int n)
{
  VEC_TYPEDEFS(V);
  if(n <= V::vec_size && V::vec_size != 1)
    return fft_state_memory_size<Scalar<T>>(n);

  return state_struct_offset<V>(n) + sizeof(State<T>);
}

template<
  typename V,
  template<typename> class SrcCfT,
  template<typename> class DstCfT>
State<typename V::T>* fft_state(Int n, void* ptr)
{
  VEC_TYPEDEFS(V);
  if(n <= V::vec_size && V::vec_size != 1)
    return fft_state<Scalar<T>, SrcCfT, DstCfT>(n, ptr);

  auto state = (State<T>*)(Uint(ptr) + Uint(state_struct_offset<V>(n)));
  state->n = n;
  state->im_off = n;
  state->working0 = (T*) ptr;
  state->working1 = state->working0 + 2 * n;
  state->twiddle = state->working1 + 2 * n;
  state->tiny_twiddle = state->twiddle + 2 * n;
  state->br_table = (Int*)(Uint(state->tiny_twiddle) + tiny_twiddle_bytes<V>());
  init_steps<V, SrcCfT<V>, DstCfT<V>>(*state);

  init_twiddle<V>([state](Int s){ return state->steps[s].npasses; },
    n, state->working0, state->twiddle, state->tiny_twiddle);

  init_br_table<V, DstCfT<V>>(*state);
  return state;
}

template<typename T>
void* fft_state_memory_ptr(State<T>* state) { return state->working0; }

template<typename V>
Int inverse_fft_state_memory_size(Int n){ return fft_state_memory_size<V>(n); }

template<
  typename V,
  template<typename> class SrcCfT,
  template<typename> class DstCfT>
State<typename V::T>* inverse_fft_state(Int n, void* ptr)
{
  return fft_state<
    V,
    cf::Swapped<SrcCfT>::template Cf,
    cf::Swapped<DstCfT>::template Cf>(n, ptr);
}

template<typename T>
void* inverse_fft_state_memory_ptr(State<T>* state)
{
  return fft_state_memory_ptr((State<T>*) state);
}

template<typename T>
NOINLINE void recursive_passes(
  const State<T>* state, Int step, T* p, Int start, Int end)
{
  Int dft_size = 1;
  for(Int i = 0; i < step; i++) dft_size <<= state->steps[i].npasses;

  Arg<T> arg;
  arg.n = state->n;
  arg.im_off = 0;
  arg.dft_size = dft_size;
  arg.start_offset = start;
  arg.end_offset = end;
  arg.src = p;
  arg.dst = nullptr;
  arg.twiddle = state->twiddle;
  arg.br_table = nullptr;
  arg.tiny_twiddle = nullptr;
  
  state->steps[step].fun_ptr(arg);

  if(step + 1 < state->nsteps && state->steps[step + 1].is_recursive)
  {
    if(end - start > optimal_size)
    {
      Int next_sz = (end - start) >> state->steps[step].npasses;
      for(Int s = start; s < end; s += next_sz)
        recursive_passes(state, step + 1, p, s, s + next_sz);
    }
    else
      recursive_passes(state, step + 1, p, start, end);
  }
}

template<typename T>
FORCEINLINE void fft_impl(const State<T>* state, Int im_off, T* src, T* dst)
{
  if(state->tiny_transform_fun)
  {
    state->tiny_transform_fun(src, dst, im_off);
    return;
  }

  Arg<T> arg;
  arg.n = state->n;
  arg.im_off = im_off;
  arg.dft_size = 1;
  arg.start_offset = 0;
  arg.end_offset = state->n;
  arg.src = src;
  arg.twiddle = state->twiddle;
  arg.br_table = state->br_table;
  arg.tiny_twiddle = state->tiny_twiddle;

  auto w0 = state->working0;
  //auto w1 = state->working1;
  auto w1 = im_off == arg.n ? dst :
    im_off == -arg.n ? dst + im_off : state->working1;
 
  if((state->ncopies & 1)) swap(w0, w1);

  arg.src = src;
  arg.dst = w0;
  state->steps[0].fun_ptr(arg);
  arg.dft_size <<= state->steps[0].npasses;

  arg.src = w0;
  arg.dst = w1;
  for(Int step = 1; step < state->nsteps - 1; )
  {
    if(state->steps[step].is_recursive)
    {
      recursive_passes(state, step, arg.src, 0, state->n);
      while(step < state->nsteps && state->steps[step].is_recursive) step++;
    }
    else
    {
      state->steps[step].fun_ptr(arg);
      arg.dft_size <<= state->steps[step].npasses;
      if(state->steps[step].is_out_of_place) swap(arg.src, arg.dst);
      step++;
    }
  }

  arg.dst = dst;  
  state->steps[state->nsteps - 1].fun_ptr(arg);
}

template<typename T>
void fft(const State<T>* state, T* src, T* dst)
{
  fft_impl(state, state->n, src, dst);
}

template<typename T>
void inverse_fft(const State<T>* state, T* src, T* dst)
{
  fft_impl(state, state->n, src, dst);
}

template<
  typename V,
  template<typename> class SrcCfT,
  template<typename> class DstCfT,
  bool inverse>
void real_pass(
  Int n,
  typename V::T* src,
  Int src_off,
  typename V::T* twiddle,
  typename V::T* dst,
  Int dst_off)
{
  if(SameType<DstCfT<V>, cf::Vec<V>>::value) ASSERT(0);
  VEC_TYPEDEFS(V);

  const Int src_ratio = SrcCfT<V>::idx_ratio;
  const Int dst_ratio = DstCfT<V>::idx_ratio;

  Vec half = V::vec(0.5);

  Complex<T> middle = SrcCfT<Scalar<T>>::load(src + n / 4 * src_ratio, src_off);

  for(
    Int i0 = 1, i1 = n / 2 - V::vec_size, iw = 0; 
    i0 <= i1; 
    i0 += V::vec_size, i1 -= V::vec_size, iw += V::vec_size)
  {
    C w = cf::Split<V>::load(twiddle + iw, n / 2);
    C s0 = SrcCfT<V>::unaligned_load(src + i0 * src_ratio, src_off);
    C s1 = reverse_complex<V>(SrcCfT<V>::load(src + i1 * src_ratio, src_off));

    //printf("%f %f %f %f %f %f\n", w.re, w.im, s0.re, s0.im, s1.re, s1.im);

    C a, b;

    if(inverse)
    {
      a = s0 + s1.adj();
      b = (s1.adj() - s0) * w.adj();
    }
    else
    {
      a = (s0 + s1.adj()) * half;
      b = ((s0 - s1.adj()) * w) * half;
    }

    C d0 = a + b.mul_neg_i();
    C d1 = a.adj() + b.adj().mul_neg_i();

    DstCfT<V>::unaligned_store(d0, dst + i0 * dst_ratio, dst_off);
    DstCfT<V>::store(reverse_complex<V>(d1), dst + i1 * dst_ratio, dst_off);
  }

  // fixes the aliasing bug
  DstCfT<Scalar<T>>::store(
    middle.adj() * (inverse ? 2.0f : 1.0f), dst + n / 4 * dst_ratio, dst_off);

  if(inverse)
  {
    T r0 = SrcCfT<Scalar<T>>::load(src, src_off).re;
    T r1 = SrcCfT<Scalar<T>>::load(src + n / 2 * src_ratio, src_off).re;
    DstCfT<Scalar<T>>::store({r0 + r1, r0 - r1}, dst, dst_off);
  }
  else
  {
    Complex<T> r0 = SrcCfT<Scalar<T>>::load(src, src_off);
    DstCfT<Scalar<T>>::store({r0.re + r0.im, 0}, dst, dst_off);
    DstCfT<Scalar<T>>::store({r0.re - r0.im, 0}, dst + n / 2 * dst_ratio, dst_off);
  }
}

template<typename T>
struct RealState
{
  State<T>* state;
  T* twiddle;
  void (*real_pass)(Int, T*, Int, T*, T*, Int);
};

template<typename V>
Int real_state_twiddle_offset(Int n)
{
  VEC_TYPEDEFS(V);
  return align_size(fft_state_memory_size<V>(n / 2));
}

template<typename V>
Int real_state_struct_offset(Int n)
{
  VEC_TYPEDEFS(V);
  return align_size(real_state_twiddle_offset<V>(n) + sizeof(T) * n);
}

template<typename V>
Int rfft_state_memory_size(Int n)
{
  VEC_TYPEDEFS(V);
  return real_state_struct_offset<V>(n) + sizeof(RealState<T>);
}

template<typename V, template<typename> class DstCfT>
RealState<typename V::T>* rfft_state(Int n, void* ptr)
{
  VEC_TYPEDEFS(V);
  RealState<T>* r = (RealState<T>*)(Uint(ptr) + real_state_struct_offset<V>(n));
  r->state = fft_state<V, cf::Scal, cf::Split>(n / 2, ptr);

  r->twiddle = (T*)(Uint(ptr) + real_state_twiddle_offset<V>(n));
  r->real_pass = &real_pass<V, cf::Split, DstCfT, false>;
  
  Int m =  n / 2;
  compute_twiddle(m, m, r->twiddle, r->twiddle + m);
  copy(r->twiddle + 1, m - 1, r->twiddle);
  copy(r->twiddle + m + 1, m - 1, r->twiddle + m);
  return r;
}

template<typename T>
void* rfft_state_memory_ptr(RealState<T>* state)
{
  return fft_state_memory_ptr(state->state);
}

template<typename T>
void rfft(const RealState<T>* state, T* src, T* dst)
{
  fft(state->state, src, state->state->working1);
  state->real_pass(
    state->state->n * 2,
    state->state->working1,
    state->state->n, 
    state->twiddle,
    dst,
    align_size<T>(state->state->n + 1));
}

template<typename T>
struct InverseRealState
{
  State<T>* state;
  T* twiddle;
  void (*real_pass)(Int, T*, Int, T*, T*, Int);
};

template<typename V>
Int inverse_rfft_state_memory_size(Int n)
{
  return rfft_state_memory_size<V>(n);
}

template<typename V, template<typename> class SrcCfT>
InverseRealState<typename V::T>* inverse_rfft_state(Int n, void* ptr)
{
  VEC_TYPEDEFS(V);
  auto r = (InverseRealState<T>*)(Uint(ptr) + real_state_struct_offset<V>(n));

  r->twiddle = (T*)(Uint(ptr) + real_state_twiddle_offset<V>(n));
  r->real_pass = &real_pass<V, SrcCfT, cf::Split, true>;
  r->state = inverse_fft_state<V, cf::Split, cf::Scal>(n / 2, ptr);

  Int m =  n / 2;
  compute_twiddle(m, m, r->twiddle, r->twiddle + m);
  copy(r->twiddle + 1, m - 1, r->twiddle);
  copy(r->twiddle + m + 1, m - 1, r->twiddle + m);
  return r;
}

template<typename T>
void* inverse_rfft_state_memory_ptr(InverseRealState<T>* state)
{
  return fft_state_memory_ptr(state->state);
}

template<typename T>
void inverse_rfft(const InverseRealState<T>* state, T* src, T* dst)
{
  state->real_pass(
    state->state->n * 2,
    src,
    align_size<T>(state->state->n + 1),
    state->twiddle,
    state->state->working1,
    state->state->n);

  inverse_fft(state->state, state->state->working1, dst);
}

#if 0
template<typename V>
void multi_first_two_passes(
  Int n,
  Int m,
  Int dft_size,
  Int start,
  Int end,
  Int im_off,
  typename V::T* src,
  typename V::T* dst)
{
  VEC_TYPEDEFS(V);
  typedef cf::Split<V> Cf;

  T* s0 = src + 0 * n * m / 4;
  T* s1 = src + 1 * n * m / 4;
  T* s2 = src + 2 * n * m / 4;
  T* s3 = src + 3 * n * m / 4;

  T* d0 = dst + 0 * n * m / 4;
  T* d1 = dst + 1 * n * m / 4;
  T* d2 = dst + 2 * n * m / 4;
  T* d3 = dst + 3 * n * m / 4;

  for(Int i = 0; i < n * m / 4; i += V::vec_size)
  {
    C a0 = Cf::load(s0 + i, im_off);
    C a1 = Cf::load(s1 + i, im_off);
    C a2 = Cf::load(s2 + i, im_off);
    C a3 = Cf::load(s3 + i, im_off);

    C b0 = a0 + a2;
    C b2 = a0 - a2;
    C b1 = a1 + a3;
    C b3 = a1 - a3;

    C c0 = b0 + b1;
    C c1 = b0 - b1;
    C c2 = b2 + b3.mul_neg_i();
    C c3 = b2 - b3.mul_neg_i();

    Cf::store(c0, d0 + i, im_off);
    Cf::store(c1, d1 + i, im_off);
    Cf::store(c2, d2 + i, im_off);
    Cf::store(c3, d3 + i, im_off);
  }
}
#endif

template<typename V>
NOINLINE void multi_two_passes_inner(
  typename V::T* a0,
  typename V::T* a1,
  typename V::T* a2,
  typename V::T* a3,
  Complex<typename V::Vec> t0,
  Complex<typename V::Vec> t1,
  Complex<typename V::Vec> t2,
  Int m,
  Int im_off)
{
  VEC_TYPEDEFS(V);
  typedef cf::Split<V> Cf;
  for(Int i = 0; i < m * Cf::idx_ratio; i += Cf::stride)
  {
    C b0 = Cf::load(a0 + i, im_off);
    C b1 = Cf::load(a1 + i, im_off);
    C b2 = Cf::load(a2 + i, im_off);
    C b3 = Cf::load(a3 + i, im_off);
    two_passes_inner(b0, b1, b2, b3, b0, b1, b2, b3, t0, t1, t2);
    Cf::store(b0, a0 + i, im_off);
    Cf::store(b1, a1 + i, im_off);
    Cf::store(b2, a2 + i, im_off);
    Cf::store(b3, a3 + i, im_off);
  }
}

template<typename V>
void multi_two_passes(
  Int n,
  Int m,
  Int dft_size,
  Int start,
  Int end,
  Int im_off,
  typename V::T* twiddle,
  typename V::T* data)
{
  VEC_TYPEDEFS(V);

  Int stride = n >> log2(dft_size);
  auto tw = twiddle + n - 4 * dft_size;
  if(start != 0) tw += 3 * (start >> log2(stride));

  for(Int i = start; i < end; i += stride)
  {
    auto tw0 = {VecCf::vec(tw[0]), VecCf::vec(tw[1])}; 
    auto tw1 = {VecCf::vec(tw[2]), VecCf::vec(tw[3])}; 
    auto tw2 = {VecCf::vec(tw[4]), VecCf::vec(tw[5])}; 
    tw += 6;

    for(Int j = i; j < i + stride / 4; j++)
      multi_two_passes_inner<V>(
        data + (j + 0 * stride / 4) * m,
        data + (j + 1 * stride / 4) * m,
        data + (j + 2 * stride / 4) * m,
        data + (j + 3 * stride / 4) * m,
        tw0, tw1, tw2, m, im_off);
  }
}

template<typename V>
void multi_last_pass(
  Int n,
  Int m,
  Int start,
  Int end,
  Int im_off,
  typename V::T* twiddle,
  typename V::T* data)
{
  VEC_TYPEDEFS(V);
  for(Int i = start; i < end; i += 2)
  {
    auto tw = {VecCf::vec(twiddle[i]), VecCf::vec(tw[i + 1])}; 
    auto p0 = data + i * m;
    auto p1 = data + (i + 1) * m;   
    typedef cf::Split<V> Cf;
    for(Int i = 0; i < m * Cf::idx_ratio; i += Cf::stride)
    {
      C b0 = Cf::load(p0 + i, im_off);
      C mul = Cf::load(p1 + i, im_off) * tw;
      Cf::store(b0 + mul, p0 + i, im_off);
      Cf::store(b0 - mul, p1 + i, im_off);
    }
  }
}

template<typename T>
struct MultiState
{
  Int n;
  Int m;
  T* working;
  T* twiddle;
  typedef void (*fun_ptr)(const MultiState* state, T* data);
};

// The result is bit reversed
template<typename V>
void multi_fft(const MultiState<typename V::T>& s, typename V::T* data, Int im_off)
{
  Int dft_size = 1;
  for(; dft_size <= s.n / 4; dft_size *= 4)
    multi_two_passes<V>(s.n, s.m, dft_size, 0, s.n, im_off, s.twiddle, data);
    
  if(dft_size < s.n)
    multi_last_pass<V>(s.n, s.m, 0, s.n, im_off, s.twiddle, data);
}

template<typename V>
Int multi_state_struct_offset(Int n)
{
  VEC_TYPEDEFS(V);
  return align_size(
    2 * n * sizeof(T) +
    2 * n * sizeof(T));
};

template<typename V>
Int multi_state_memory_size(Int n)
{
  VEC_TYPEDEFS(V);
  return multi_state_struct_offset<V>(n) + sizeof(MultiState<T>);
}

template<typename V>
MultiState<typename V::T> multi_fft_state(Int n, Int m, void* ptr)
{
  VEC_TYPEDEFS(V);
  auto r = (MultiState<T>*) (Uint(ptr) + multi_state_struct_offset<V>(n));
  r->n = n;
  r->m = m;
  r->working = (T*) ptr;
  r->twiddle = r->working + 2 * n;
  init_twiddle<V>(
    [n](Int s){ return (Int(1) << (s + 2)) > n ? 1 : 2; },
    n, r->working, r->twiddle, nullptr);

  r->fun_ptr = &multi_fft<V>;
}

const Int maxdim = 64;

template<typename T>
struct MultidimState
{
  Int ndim;
  Int num_elements;
  T* working;
  State<T>* last_transform;
  MultiState<T>* transforms[maxdim];
};

Int product(PtrRange<Int> pr)
{
  Int r = 1;
  for(auto e : pr) r *= e;
  return r;
}

template<typename V>
Int multidim_state_memory_size(Int ndim, Int* dim)
{
  VEC_TYPEDEFS(V);
  Int r = align_size(sizeof(MultidimState<T>));

  Int working_size = 2 * sizeof(T) * product(ptr_range(dim, ndim));
  r = align_size(working_size);

  for(Int i = 0; i < ndim - 1; i++)
    r = align_size(r + multi_state_memory_size<V>(dim[i]));

  r = align_size(r + fft_state_memory_size<V>(dim[ndim - 1]));

  return r;
}

template<typename V>
MultidimState<typename V::T> multidim_state(Int ndim, Int* dim, void* mem)
{
  VEC_TYPEDEFS(V);
  auto s = (MultidimState<typename V::T>*) mem;
  s->ndim = ndim;
  mem = (void*) align_size(Uint(mem) + sizeof(MultidimState<T>));
  s->working = (T*) mem;
 
  s->num_elements = product(ptr_range(dim, ndim));
  Int working_size = 2 * sizeof(T) * s->num_elements;
  mem = (void*) align_size(Uint(mem) + working_size);
  
  for(Int i = 0; i < ndim - 1; i++)
  {
    Int m = product(ptr_range(dim + i + 1, dim + ndim));
    s->transforms[i] = multi_fft_state<V>(dim[i], m, s->num_elements, mem);
    mem = (void*) align_size(Uint(mem) + multi_state_memory_size<V>(dim[i]));
  }

  s->last_transform = fft_state<V>(dim[ndim - 1], mem);
}

template<typename T>
void multidim_fft_impl(
  Int ndim, MultiState<T>* transforms, State<T>* last_transform,
  Int im_off, T* src, T* dst)
{
  ASSERT(ndim > 0);
  if(ndim == 1) fft_impl(last_transform, im_off, src, dst);
  else
  {
    transforms->fun_ptr(transforms, src);

    Int m = transforms->m;
    Int n = transforms->n;
    for(BitReversed br(n); br.i < n; br.advance())
      multidim_fft_impl(
        ndim - 1, transforms + 1, last_transform, im_off,
        src + br.i * m, dst + br.br * m);
  }
}

template<typename T>
void multidim_fft(MultidimState<T>* state, T* src, T* dst)
{
  copy(src, 2 * state->num_elements, state->working);
  multidim_fft_impl(
    state->ndim,
    state->transforms,
    state->last_transform,
    state->num_elements,
    state->working,
    dst); 
}

