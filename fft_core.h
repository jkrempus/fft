typedef long Int;
typedef unsigned long Uint;

const Int max_int = Int(Uint(-1) >> 1);

#define FORCEINLINE __attribute__((always_inline)) inline
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

Int large_fft_size = 1 << 14;
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

template<typename T>
FORCEINLINE void copy(const T* src, Int n, T* dst)
{
#if defined __GNUC__ || defined __clang__
  __builtin_memmove(dst, src, n * sizeof(T));
#else
  for(Int i = 0; i < n; i++) dst[i] = src[i];
#endif
}

template<typename T>
void interleave(const T* src0, const T* src1, Int n, T* dst)
{
  for(Int i = 0; i < n; i++)
  {
    dst[2 * i] = src0[i];
    dst[2 * i + 1] = src1[i];
  }
}

template<typename T>
void deinterleave(const T* src, Int n, T* dst0, T* dst1)
{
  for(Int i = 0; i < n; i++)
  {
    dst0[i] = src[2 * i];
    dst1[i] = src[2 * i + 1];
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
  struct Split
  {
    static const Int elemsz = 1;

    template<typename V>
    static FORCEINLINE Complex<typename V::Vec>
    load(typename V::Vec* ptr, Int off)
    {
      return {ptr[0], ptr[off]};
    }
    
    template<typename V>
    static FORCEINLINE Complex<typename V::Vec>
    unaligned_load(typename V::T* ptr, Int off)
    {
      return {V::unaligned_load(ptr), V::unaligned_load(ptr + off)};
    }

    template<typename V>
    static FORCEINLINE void
    store(Complex<typename V::Vec> a, typename V::Vec* ptr, Int off)
    {
      ptr[0] = a.re;
      ptr[off] = a.im;
    }

    template<typename V>
    static FORCEINLINE void
    unaligned_store(Complex<typename V::Vec> a, typename V::T* ptr, Int off)
    {
      V::unaligned_store(a.re, ptr);
      V::unaligned_store(a.im, ptr + off);
    }
  };

  struct Vec
  {
    static const Int elemsz = 2;

    template<typename V>
    static FORCEINLINE Complex<typename V::Vec>
    load(typename V::Vec* ptr, Int off)
    {
      return {ptr[0], ptr[1]};
    }

    template<typename V>
    static FORCEINLINE Complex<typename V::Vec>
    unaligned_load(typename V::T* ptr, Int off)
    {
      return {V::unaligned_load(ptr), V::unaligned_load(ptr + V::vec_size)};
    }

    template<typename V>
    static FORCEINLINE void
    store(Complex<typename V::Vec> a, typename V::Vec* ptr, Int off)
    {
      ptr[0] = a.re;
      ptr[1] = a.im;
    }

    template<typename V>
    static FORCEINLINE void
    unaligned_store(Complex<typename V::Vec> a, typename V::T* ptr, Int off)
    {
      V::unaligned_store(a.re, ptr);
      V::unaligned_store(a.im, ptr + V::vec_size);
    }
  };

  struct Scal
  {
    static const Int elemsz = 2;

    template<typename V>
    static FORCEINLINE Complex<typename V::Vec>
    load(typename V::Vec* ptr, Int off)
    {
      Complex<typename V::Vec> r;
      V::deinterleave(ptr[0], ptr[1], r.re, r.im);
      return r;
    }

    template<typename V>
    static FORCEINLINE Complex<typename V::Vec>
    unaligned_load(typename V::T* ptr, Int off)
    {
      Complex<typename V::Vec> r;
      V::deinterleave(
        V::unaligned_load(ptr), V::unaligned_load(ptr + V::vec_size),
        r.re, r.im);

      return r;
    }

    template<typename V>
    static FORCEINLINE void
    store(Complex<typename V::Vec> a, typename V::Vec* ptr, Int off)
    {
      V::interleave(a.re, a.im, ptr[0], ptr[1]);
    }

    template<typename V>
    static FORCEINLINE void
    unaligned_store(Complex<typename V::Vec> a, typename V::T* ptr, Int off)
    {
      V::interleave(a.re, a.im, a.re, a.im);
      V::unaligned_store(a.re, ptr);
      V::unaligned_store(a.im, ptr + V::vec_size);
    }
  };
}

namespace cf = complex_format;

template<typename V>
Complex<typename V::Vec> reverse_complex(Complex<typename V::Vec> a)
{
  return { V::reverse(a.re), V::reverse(a.im) };
}

template<typename T>
struct Arg
{
  Int n;
  Int dft_size;
  T* src;
  T* twiddle;
  T* tiny_twiddle;
  T* dst;
};

template<typename T>
struct Step
{
  typedef void (*pass_fun_t)(const Arg<T>&);
  short npasses;
  pass_fun_t fun_ptr;
};

template<typename T>
struct State
{
  Int n;
  T* working;
  T* twiddle;
  T* tiny_twiddle;
  Step<T> steps[8 * sizeof(Int)];
  Int nsteps;
  Int num_copies;
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

  template<bool interleave_rearrange>
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

  static Vec unaligned_load(T* p) { return *p; }
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

  template<bool interleave_rearrange>
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
  
  template<bool interleave_rearrange>
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
};
#endif

#ifdef __AVX__
#include <immintrin.h>

struct AvxFloat
{
  typedef float T;
  typedef __m256 Vec;
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
  
  // The input matrix has 4 rows and vec_size columns
  // TODO: this needs to be updated to support different element order
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

  template<bool interleave_rearrange>
  static void FORCEINLINE transpose(
    Vec a0, Vec a1, Vec a2, Vec a3,
    Vec a4, Vec a5, Vec a6, Vec a7,
    Vec& r0, Vec& r1, Vec& r2, Vec& r3,
    Vec& r4, Vec& r5, Vec& r6, Vec& r7)
  {
#if 0
    if(interleave_rearrange)
      transpose<false>(
        a0, a1, a4, a5, a2, a3, a6, a7,
        r0, r1, r4, r5, r2, r3, r6, r7);
    else
#endif
    {
      transpose4x4_two(a0, a1, a2, a3);
      transpose4x4_two(a4, a5, a6, a7);

      transpose_128(a0, a4, r0, r4);
      transpose_128(a1, a5, r1, r5);
      transpose_128(a2, a6, r2, r6);
      transpose_128(a3, a7, r3, r7);
    }
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

  static Vec unaligned_load(T* p) { return _mm256_loadu_ps(p); }
  static void unaligned_store(Vec val, T* p) { _mm256_storeu_ps(p, val); }
};
#endif

#define VEC_TYPEDEFS(V) \
  typedef typename V::T T; \
  typedef typename V::Vec Vec; \
  typedef Complex<typename V::Vec> C

Int twiddle_elements(Int npasses)
{
  return
    npasses == 1 ? 1 :
    npasses == 2 ? 3 :
    npasses == 3 ? 5 : max_int;
}

template<typename T>
void store_two_pass_twiddle(
  Complex<T> first,
  Complex<T>* dst)
{
  dst[0] = first;
  auto second = first * first;
  auto third = second * first;
  dst[1] = second;
  dst[2] = third;
}

template<typename T>
void swap(T& a, T& b)
{
  T tmpa = a;
  T tmpb = b;
  a = tmpb;
  b = tmpa;
}

template<typename V, typename SrcCF>
void rearrange_vector_elements_like_load(typename V::Vec* p, Int len)
{
  VEC_TYPEDEFS(V);
  Vec a[2];
  
  for(auto end = p + len; p < end; p += 2)
  {
    T* src = (T*) p;
    T* dst = (T*) a;

    //Get it into the source format
    for(Int i = 0; i < V::vec_size; i++)
    {
      if(SameType<SrcCF, cf::Scal>::value)
      {
        dst[2 * i] = src[i];
        dst[2 * i + 1] = src[i + V::vec_size];
      }
      else
      {
        dst[2 * i] = src[2 * i];
        dst[2 * i + 1] = src[2 * i + 1];
      }
    }

    //load it like we would during fft computation
    Complex<Vec> c = SrcCF::template load<V>(a, 1);
    //store it as cf::Vec
    p[0] = c.re;
    p[1] = c.im; 
  }
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

template<typename V, typename SrcCF, typename DstCF>
void init_twiddle(State<typename V::T>& state)
{
  VEC_TYPEDEFS(V);

  auto dst = state.twiddle;
  auto n = state.n;
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
    auto re = state.tiny_twiddle + 2 * V::vec_size * tiny_log2(size);
    auto im = re + V::vec_size;

    for(Int j = 0; j < V::vec_size; j++)
    {
      re[j] = (end_re - 2 * size)[j & (size - 1)];
      im[j] = (end_im - 2 * size)[j & (size - 1)];
    }
  }

  Int vn = n / V::vec_size;
  interleave((Vec*) dst, (Vec*) dst + vn, vn, (Vec*) state.working);
  copy((Vec*) state.working, vn, (Vec*) dst);

  int dft_size = 1;
  for(Int s = 0; s < state.nsteps; s++)
  {
    auto step = state.steps[s];
    //printf("nsteps %d npasses %d\n", step.nsteps, step.npasses);

    Int vdft_size = dft_size / V::vec_size;

    if(step.npasses == 2 && dft_size >= V::vec_size)
    {
      auto src_row0 = ((Complex<Vec>*) state.working) + vn - 4 * vdft_size;
      auto dst_row0 = ((Complex<Vec>*) dst) + vn - 4 * vdft_size;
      for(Int i = 0; i < vdft_size; i++)
        store_two_pass_twiddle(src_row0[i], dst_row0 + 3 * i);
    }
    else if(step.npasses == 3 && dft_size >= V::vec_size)
    {
      auto src_row0 = ((Complex<Vec>*) state.working) + vn - 4 * vdft_size;
      auto src_row1 = ((Complex<Vec>*) state.working) + vn - 8 * vdft_size;
      auto dst_row1 = ((Complex<Vec>*) dst) + vn - 8 * vdft_size;
      for(Int i = 0; i < vdft_size; i++)
      {
        store_two_pass_twiddle(src_row0[i], dst_row1 + 5 * i);
        dst_row1[5 * i + 3] = src_row1[i];
        dst_row1[5 * i + 4] = src_row1[i + vdft_size];
      }
    } 
    else if(step.npasses == 4 && dft_size >= V::vec_size)
    {
      auto src_row0 = ((Complex<Vec>*) state.working) + vn - 4 * vdft_size;
      auto dst_row0 = ((Complex<Vec>*) dst) + vn - 4 * vdft_size;
      for(Int i = 0; i < vdft_size; i++)
        store_two_pass_twiddle(src_row0[i], dst_row0 + 3 * i);
      
      auto src_row2 = ((Complex<Vec>*) state.working) + vn - 16 * vdft_size;
      auto dst_row2 = ((Complex<Vec>*) dst) + vn - 16 * vdft_size;
      for(Int i = 0; i < 4 * vdft_size; i++)
        store_two_pass_twiddle(src_row2[i], dst_row2 + 3 * i);
    }

    dft_size <<= step.npasses;
  }

  rearrange_vector_elements_like_load<V, SrcCF>((Vec*) dst, 2 * n / V::vec_size);
  rearrange_vector_elements_like_load<V, SrcCF>(
    (Vec*) state.tiny_twiddle, 2 * tiny_log2(V::vec_size));
}

template<typename T> T min(T a, T b){ return a < b ? a : b; }
template<typename T> T max(T a, T b){ return a > b ? a : b; }

template<typename T> T sq(T a){ return a * a; }

template<typename V, Int dft_size, typename SrcCF>
void ct_dft_size_pass(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  typedef Complex<Vec> C;
  Int vn = arg.n / V::vec_size;
  auto vsrc0 = (Vec*) arg.src;
  auto vsrc1 = (Vec*) arg.src + SrcCF::elemsz * vn / 2;
  auto vdst = (Complex<Vec>*) arg.dst;
  C tw = ((C*) arg.tiny_twiddle)[tiny_log2(dft_size)];
  for(auto end = vdst + vn; vdst < end;)
  {
    C a0 = SrcCF::template load<V>(vsrc0, vn);
    C a1 = SrcCF::template load<V>(vsrc1, vn);

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
    V::template interleave_multi<nelem>(b0.re, b1.re, vdst[0].re, vdst[1].re);
    V::template interleave_multi<nelem>(b0.im, b1.im, vdst[0].im, vdst[1].im);

    vsrc0 += SrcCF::elemsz;
    vsrc1 += SrcCF::elemsz;
    vdst += 2;
  }
}

template<typename V, typename SrcCF>
void first_two_passes_impl(Int n, const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  typedef Complex<Vec> C;

  Int vn = n / V::vec_size;
  Vec* vsrc0 = (Vec*) arg.src;
  Vec* vsrc1 = (Vec*) arg.src + SrcCF::elemsz * vn / 4;
  Vec* vsrc2 = (Vec*) arg.src + SrcCF::elemsz * 2 * vn / 4;
  Vec* vsrc3 = (Vec*) arg.src + SrcCF::elemsz * 3 * vn / 4;
  auto vdst = (C*) arg.dst;

  for(Int i = 0; i < vn / 4; i++)
  {
    C a0 = SrcCF::template load<V>(vsrc0 + SrcCF::elemsz * i, vn);
    C a1 = SrcCF::template load<V>(vsrc1 + SrcCF::elemsz * i, vn);
    C a2 = SrcCF::template load<V>(vsrc2 + SrcCF::elemsz * i, vn);
    C a3 = SrcCF::template load<V>(vsrc3 + SrcCF::elemsz * i, vn);

    C b0 = a0 + a2;
    C b1 = a0 - a2;
    C b2 = a1 + a3; 
    C b3 = a1 - a3; 

    C c0 = b0 + b2; 
    C c2 = b0 - b2;
    C c1 = b1 + b3.mul_neg_i();
    C c3 = b1 - b3.mul_neg_i();

    Int j = 4 * i;
    V::transpose(
      c0.re, c1.re, c2.re, c3.re,
      vdst[j].re, vdst[j + 1].re, vdst[j + 2].re, vdst[j + 3].re);
    
    V::transpose(
      c0.im, c1.im, c2.im, c3.im,
      vdst[j].im, vdst[j + 1].im, vdst[j + 2].im, vdst[j + 3].im);
  }
}

template<typename V, typename SrcCF>
void first_two_passes(const Arg<typename V::T>& arg)
{
  first_two_passes_impl<V, SrcCF>(arg.n, arg);
}

template<typename V, typename SrcCF, Int n>
void first_two_passes_ct_size(const Arg<typename V::T>& arg)
{
  first_two_passes_impl<V, SrcCF>(n, arg);
}

template<typename V, typename SrcCF>
FORCEINLINE void first_three_passes_impl(
  Int n,
  typename V::Vec* src,
  Complex<typename V::Vec>* dst)
{
  VEC_TYPEDEFS(V);
  Int l = n / 8;
  Vec invsqrt2 = V::vec(SinCosTable<T>::cos[2]);

  for(auto end = src + l * SrcCF::elemsz;;)
  {
    C c0, c1, c2, c3;
    {
      C a0 = SrcCF::template load<V>(src + 0 * l * SrcCF::elemsz, n);
      C a1 = SrcCF::template load<V>(src + 2 * l * SrcCF::elemsz, n);
      C a2 = SrcCF::template load<V>(src + 4 * l * SrcCF::elemsz, n);
      C a3 = SrcCF::template load<V>(src + 6 * l * SrcCF::elemsz, n);
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
      C a0 = SrcCF::template load<V>(src + 1 * l * SrcCF::elemsz, n);
      C a1 = SrcCF::template load<V>(src + 3 * l * SrcCF::elemsz, n);
      C a2 = SrcCF::template load<V>(src + 5 * l * SrcCF::elemsz, n);
      C a3 = SrcCF::template load<V>(src + 7 * l * SrcCF::elemsz, n);
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

    src += SrcCF::elemsz;

    V::template transpose<SameType<SrcCF, cf::Scal>::value>(
      c0.re + mul0.re, c1.re + mul1.re, c2.re + mul2.re, c3.re + mul3.re,
      c0.re - mul0.re, c1.re - mul1.re, c2.re - mul2.re, c3.re - mul3.re,
      dst[0].re, dst[1].re, dst[2].re, dst[3].re,
      dst[4].re, dst[5].re, dst[6].re, dst[7].re);

    V::template transpose<SameType<SrcCF, cf::Scal>::value>(
      c0.im + mul0.im, c1.im + mul1.im, c2.im + mul2.im, c3.im + mul3.im,
      c0.im - mul0.im, c1.im - mul1.im, c2.im - mul2.im, c3.im - mul3.im,
      dst[0].im, dst[1].im, dst[2].im, dst[3].im,
      dst[4].im, dst[5].im, dst[6].im, dst[7].im);

    dst += 8;
    if(src == end) break;
  }
}

template<typename V, typename SrcCF>
void first_three_passes(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  first_three_passes_impl<V, SrcCF>(
    arg.n / V::vec_size,
    (Vec*) arg.src,
    (Complex<Vec>*) arg.dst);
}

template<typename V, typename SrcCF, Int n>
void first_three_passes_ct_size(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  first_three_passes_impl<V, SrcCF>(
    n / V::vec_size,
    (Vec*) arg.src,
    (Complex<Vec>*) arg.dst);
}

template<typename V, typename DstCF>
FORCEINLINE void last_pass_impl(
  Int dft_size,
  Complex<typename V::Vec>* src,
  Complex<typename V::Vec>* twiddle,
  typename V::Vec* dst)
{
  for(Int i0 = 0, i1 = dft_size; i0 < dft_size; i0++, i1++)
  {
    auto a = src[i0];
    auto mul = src[i1] * twiddle[i0];
    DstCF::template store<V>(a + mul, dst + i0 * DstCF::elemsz, 2 * dft_size);

    DstCF::template store<V>(a - mul, dst + i1 * DstCF::elemsz, 2 * dft_size);
  }
}

template<typename V, typename DstCF>
void last_pass_vec(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  last_pass_impl<V, DstCF>(
    arg.n / V::vec_size / 2,
    (Complex<Vec>*) arg.src,
    (Complex<Vec>*) arg.twiddle,
    (Vec*) arg.dst);
}

template<typename V, typename DstCF, Int n>
void last_pass_vec_ct_size(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  last_pass_impl<V, DstCF>(
    n / V::vec_size / 2,
    (Complex<Vec>*) arg.src,
    (Complex<Vec>*) arg.twiddle,
    (Vec*) arg.dst);
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

template<
  typename T,
  void (*fun0)(const Arg<T>& a),
  void (*fun1)(const Arg<T>& a)>
void two_steps(const Arg<T>& arg_in)
{
  Arg<T> arg = arg_in;
  fun0(arg);
  swap(arg.src, arg.dst);
  fun1(arg); 
}

template<typename V, typename DstCF>
void two_passes(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  typedef Complex<Vec> C;

  Int n = arg.n / V::vec_size;
  Int dft_size = arg.dft_size / V::vec_size;
  auto src = (Complex<Vec>*) arg.src;
  auto tw = (Complex<Vec>*) arg.twiddle + n - 4 * dft_size;
  auto dst = (Vec*) arg.dst;

  Int l1 = n / 4;
  Int l2 = 2 * l1;
  Int l3 = 3 * l1;

  Int m1 = dft_size * DstCF::elemsz;
  Int m2 = 2 * m1;
  Int m3 = 3 * m1;

  for(C* end = src + dft_size; src < end;)
  {
    auto s = src;
    auto d = dst;
    auto tw0 = tw[0];
    auto tw1 = tw[1];
    auto tw2 = tw[2]; 
    src += 1;
    tw += 3;
    dst += DstCF::elemsz;

    for(C* end1 = s + l1;;)
    {
      C d0, d1, d2, d3;
      two_passes_inner(s[0], s[l1], s[l2], s[l3], d0, d1, d2, d3, tw0, tw1, tw2);
      s += dft_size;

      DstCF::template store<V>(d0, d, n);
      DstCF::template store<V>(d1, d + m1, n);
      DstCF::template store<V>(d2, d + m2, n);
      DstCF::template store<V>(d3, d + m3, n);

      d += m1 + m3;
      if(!(s < end1)) break;
    }
  }
}

namespace index_mapping
{
  Int offset(Int off, Int dft_size, Int npasses, bool dft_size_is_large)
  {
    if(dft_size_is_large)
      return ((off & ~(dft_size - 1)) << npasses) + (off & (dft_size - 1));
    else
      return off << npasses;
  }

  Int to_strided_index(
    Int i, Int n, Int stride, Int chunk_size, Int dft_size, Int npasses,
    bool dft_size_is_large)
  {
    if(dft_size_is_large)
    {
      Int ichunk = i / chunk_size;
      Int chunk_offset = i % chunk_size;
      Int dft_size_mul = 1 << npasses;
      
      return
        (ichunk & ~(dft_size_mul - 1)) * stride +
        (ichunk & (dft_size_mul - 1)) * dft_size;
    }
    else
    {
      Int contiguous_size = chunk_size << npasses;
      Int contiguous_multiple = i & ~(contiguous_size - 1);
      Int contiguous_offset = i & (contiguous_size - 1);
      return contiguous_multiple * stride / chunk_size + contiguous_offset;
    }
  }
}

template<
  Int npasses,
  Int chunk_size,
  bool src_is_strided,
  bool dst_is_strided,
  typename DstCF,
  typename V>
FORCEINLINE void two_passes_strided_impl(
  Int n,
  Int nchunks,
  Int initial_dft_size,
  Int offset,
  Complex<typename V::Vec>* src0,
  Complex<typename V::Vec>* twiddle_start,
  typename V::Vec* dst0)
{
  VEC_TYPEDEFS(V);
  namespace im = index_mapping;

  //printf("npasses %d offset %d\n", npasses, offset);
  typedef Complex<Vec> C;
 
  Int l = nchunks * chunk_size / 4;
  Int dft_size = initial_dft_size << npasses;
  Int m = min(initial_dft_size, chunk_size) << npasses;

  C* twiddle = twiddle_start + n - 4 * dft_size;

  bool is_large = initial_dft_size >= chunk_size;
  Int soffset = im::offset(offset, initial_dft_size, npasses, is_large);
  Int doffset = im::offset(offset, initial_dft_size, npasses + 2, is_large);
  Int chunk_stride = n / nchunks;

  Int sstride = 
    src_is_strided ? im::to_strided_index(
      l, n, chunk_stride, chunk_size, initial_dft_size, npasses, is_large)
    : l;

  Int dstride = 
    dst_is_strided ? im::to_strided_index(
      m, n, chunk_stride, chunk_size, initial_dft_size, npasses + 2, is_large)
    : m;

  C* src1 = src0 + sstride;
  C* src2 = src1 + sstride;
  C* src3 = src2 + sstride;
  
  Vec* dst1 = dst0 + dstride * DstCF::elemsz;
  Vec* dst2 = dst1 + dstride * DstCF::elemsz;
  Vec* dst3 = dst2 + dstride * DstCF::elemsz;

  if(is_large)
    for(Int i = 0; i < l; i += m)
    {
      for(Int j = 0; j < m; j += chunk_size)
      {
        Int s = i + j;
        Int d = 4 * i + j;

        Int strided_s = soffset + im::to_strided_index(
          s, n, chunk_stride, chunk_size, initial_dft_size, npasses, true);

        Int strided_d = doffset + im::to_strided_index(
          d, n, chunk_stride, chunk_size, initial_dft_size, npasses + 2, true);

        if(src_is_strided) s = strided_s;
        if(dst_is_strided) d = strided_d; 

        for(Int k = 0; k < chunk_size; k++)
        {
          auto tw = twiddle + 3 * (strided_s & (dft_size - 1));
       
          C d0, d1, d2, d3; 
          two_passes_inner(
            src0[s], src1[s], src2[s], src3[s], d0, d1, d2, d3,
            tw[0], tw[1], tw[2]);

          DstCF::template store<V>(d0, dst0 + d * DstCF::elemsz, n);  
          DstCF::template store<V>(d1, dst1 + d * DstCF::elemsz, n);  
          DstCF::template store<V>(d2, dst2 + d * DstCF::elemsz, n);  
          DstCF::template store<V>(d3, dst3 + d * DstCF::elemsz, n);  

          s++;
          d++;
          strided_s++;
        }
      }
    }
  else
    for(Int i = 0; i < l; i += m)
    {
      Int s = i;
      Int d = 4 * i;

      Int strided_s = soffset + im::to_strided_index(
        s, n, chunk_stride, chunk_size, initial_dft_size, npasses, false);

      Int strided_d = doffset + im::to_strided_index(
        d, n, chunk_stride, chunk_size, initial_dft_size, npasses + 2, false);

      if(src_is_strided) s = strided_s;
      if(dst_is_strided) d = strided_d;

      for(Int j = 0; j < m; j++)
      {
        auto tw = twiddle + 3 * (strided_s & (dft_size - 1));

        C d0, d1, d2, d3; 
        two_passes_inner(
          src0[s], src1[s], src2[s], src3[s], d0, d1, d2, d3,
          tw[0], tw[1], tw[2]);

        DstCF::template store<V>(d0, dst0 + d * DstCF::elemsz, n);  
        DstCF::template store<V>(d1, dst1 + d * DstCF::elemsz, n);  
        DstCF::template store<V>(d2, dst2 + d * DstCF::elemsz, n);  
        DstCF::template store<V>(d3, dst3 + d * DstCF::elemsz, n);  

        s++;
        d++;
        strided_s++;
      }
    }
}

template<typename V, typename DstCF>
void four_passes(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  Int n = arg.n / V::vec_size;
  Int dft_size = arg.dft_size / V::vec_size;
  auto src = (Complex<Vec>*) arg.src;
  auto twiddle = (Complex<Vec>*) arg.twiddle;
  auto dst = (Vec*) arg.dst;
  
  typedef Complex<Vec> C;

  const Int chunk_size = 16;
  const Int nchunks = 16;
  Int stride = n / nchunks;
  Vec working[2 * chunk_size * nchunks];

  for(Int offset = 0; offset < stride; offset += chunk_size)
  {
    two_passes_strided_impl<0, chunk_size, true, false, cf::Vec, V>(
      n, nchunks, dft_size, offset, src, twiddle, working);
    two_passes_strided_impl<2, chunk_size, false, true, DstCF, V>(
      n, nchunks, dft_size, offset, (C*) working, twiddle, dst);
  }
}

template<typename V, typename DstCF>
FORCEINLINE void last_three_passes_impl(
  Int n,
  Complex<typename V::Vec>* src,
  Complex<typename V::Vec>* twiddle,
  typename V::Vec* dst)
{
  typedef Complex<typename V::Vec> C;
  Int l1 = n / 8;
  Int l2 = 2 * l1;
  Int l3 = 3 * l1;
  Int l4 = 4 * l1;
  Int l5 = 5 * l1;
  Int l6 = 6 * l1;
  Int l7 = 7 * l1;

  for(auto end = src + l1;;)
  {
    C a[8];
    C a0, a1, a2, a3, a4, a5, a6, a7;
    {
      C tw0 = twiddle[0];
      C tw1 = twiddle[1];
      C tw2 = twiddle[2];

      {
        C mul0 =       src[0];
        C mul1 = tw0 * src[l2];
        C mul2 = tw1 * src[l4];
        C mul3 = tw2 * src[l6];

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
        C mul0 =       src[l1];
        C mul1 = tw0 * src[l3];
        C mul2 = tw1 * src[l5];
        C mul3 = tw2 * src[l7];

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
      C tw3 = twiddle[3];
      {
        auto mul = tw3 * a4;
        DstCF::template store<V>(a0 + mul, dst + 0, n);
        DstCF::template store<V>(a0 - mul, dst + l4 * DstCF::elemsz, n);
      }

      {
        auto mul = tw3.mul_neg_i() * a6;
        DstCF::template store<V>(a2 + mul, dst + l2 * DstCF::elemsz, n);
        DstCF::template store<V>(a2 - mul, dst + l6 * DstCF::elemsz, n);
      }
    }

    {
      C tw4 = twiddle[4];
      {
        auto mul = tw4 * a5;
        DstCF::template store<V>(a1 + mul, dst + l1 * DstCF::elemsz, n);
        DstCF::template store<V>(a1 - mul, dst + l5 * DstCF::elemsz, n);
      }

      {
        auto mul = tw4.mul_neg_i() * a7;
        DstCF::template store<V>(a3 + mul, dst + l3 * DstCF::elemsz, n);
        DstCF::template store<V>(a3 - mul, dst + l7 * DstCF::elemsz, n);
      }
    }

    src += 1;
    dst += DstCF::elemsz;
    twiddle += 5;
    if(src == end) break;
  }
}

template<typename V, typename DstCF>
void last_three_passes_vec(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  last_three_passes_impl<V, DstCF>(
    arg.n / V::vec_size,
    (Complex<Vec>*) arg.src,
    (Complex<Vec>*) arg.twiddle,
    (Vec*) arg.dst);
}

template<typename V, typename DstCF, Int n>
void last_three_passes_vec_ct_size(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  last_three_passes_impl<V, DstCF>(
    n / V::vec_size,
    (Complex<Vec>*) arg.src,
    (Complex<Vec>*) arg.twiddle,
    (Vec*) arg.dst);
}

template<typename V, typename SrcCF, typename DstCF>
void init_steps(State<typename V::T>& state)
{
  VEC_TYPEDEFS(V);
  Int step_index = 0;
  state.num_copies = 0;

  for(Int dft_size = 1; dft_size < state.n; step_index++)
  {
    Step<T> step;
    if(dft_size == 1 && state.n >= 8 * V::vec_size)
    {
      if(state.n == 8 * V::vec_size)
        step.fun_ptr = &first_three_passes_ct_size<V, SrcCF, 8 * V::vec_size>;
      else
        step.fun_ptr = &first_three_passes<V, SrcCF>;

      step.npasses = 3;
    }
    else if(dft_size == 1 && state.n >= 4 * V::vec_size)
    {
      if(state.n == 4 * V::vec_size)
        step.fun_ptr = &first_two_passes_ct_size<V, SrcCF, 4 * V::vec_size>;
      else if(state.n == 8 * V::vec_size)
        step.fun_ptr = &first_two_passes_ct_size<V, SrcCF, 8 * V::vec_size>;
      else
        step.fun_ptr = &first_two_passes<V, SrcCF>;

      step.npasses = 2;
    }
    else if(dft_size >= V::vec_size)
    {
      if(dft_size * 8 == state.n)
      {
        if(state.n == V::vec_size * 8)
          step.fun_ptr = &last_three_passes_vec_ct_size<V, DstCF, V::vec_size * 8>;
        else
          step.fun_ptr = &last_three_passes_vec<V, DstCF>;

        step.npasses = 3;
      }
      else if(state.n >= large_fft_size && dft_size * 16 == state.n)
      {
        step.fun_ptr = &four_passes<V, DstCF>;
        step.npasses = 4;
      }
      else if(state.n >= large_fft_size && dft_size * 16 < state.n)
      {
        step.fun_ptr = &four_passes<V, cf::Vec>;
        step.npasses = 4;
      }
      else if(dft_size * 4 == state.n)
      {
        step.fun_ptr = &two_passes<V, DstCF>;
        step.npasses = 2;
      }
      else if(dft_size * 4 < state.n)
      {
        step.fun_ptr = &two_passes<V, cf::Vec>;
        step.npasses = 2;
      }
      else
      {
        if(state.n == 2 * V::vec_size)
          step.fun_ptr = &last_pass_vec_ct_size<V, DstCF, 2 * V::vec_size>;
        else if(state.n == 4 * V::vec_size)
          step.fun_ptr = &last_pass_vec_ct_size<V, DstCF, 4 * V::vec_size>;
        else if(state.n == 8 * V::vec_size)
          step.fun_ptr = &last_pass_vec_ct_size<V, DstCF, 8 * V::vec_size>;
        else
          step.fun_ptr = &last_pass_vec<V, DstCF>;

        step.npasses = 1;
      }
    }
    else
    {
      if(V::vec_size > 1 && dft_size == 1)
        step.fun_ptr = &ct_dft_size_pass<V, 1, SrcCF>;
      else if(V::vec_size > 2 && dft_size == 2)
        step.fun_ptr = &ct_dft_size_pass<V, 2, cf::Vec>;
      else if(V::vec_size > 4 && dft_size == 4)
        step.fun_ptr = &ct_dft_size_pass<V, 4, cf::Vec>;
      else if(V::vec_size > 8 && dft_size == 8)
        step.fun_ptr = &ct_dft_size_pass<V, 8, cf::Vec>;

      step.npasses = 1;
    }

    state.steps[step_index] = step;
    dft_size <<= step.npasses;
    state.num_copies++;
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
Int state_struct_offset(Int n)
{
  VEC_TYPEDEFS(V);
  return align_size(
    sizeof(T) * 2 * n +                                     //working
    sizeof(T) * 2 * n +                                     //twiddle
    sizeof(Vec) * 2 * tiny_log2(V::vec_size));              //tiny_twiddle
}

template<typename V>
Int fft_state_memory_size(Int n)
{
  VEC_TYPEDEFS(V);
  return state_struct_offset<V>(n) + sizeof(State<T>);
}

template<typename V, typename SrcCF, typename DstCF>
State<typename V::T>* fft_state(Int n, void* ptr)
{
  VEC_TYPEDEFS(V);
  auto state = (State<T>*)(Uint(ptr) + Uint(state_struct_offset<V>(n)));
  state->n = n;
  state->working = (T*) ptr;
  state->twiddle = state->working + 2 * n;
  state->tiny_twiddle = state->twiddle + 2 * n;
  init_steps<V, SrcCF, DstCF>(*state);
  init_twiddle<V, SrcCF, DstCF>(*state);
  return state;
}

template<typename T>
void* fft_state_memory_ptr(State<T>* state) { return state->working; }

template<typename T>
void fft(const State<T>* state, T* src, T* dst)
{
  Arg<T> arg;
  arg.n = state->n;
  arg.dft_size = 1;
  arg.src = src;
  arg.twiddle = state->twiddle;
  arg.tiny_twiddle = state->tiny_twiddle;
  
  auto is_odd = bool(state->num_copies & 1);
  arg.dst = is_odd ? dst : state->working;
  auto next_dst = is_odd ? state->working : dst;

  for(Int step = 0; step < state->nsteps; step++)
  {
    auto next_dft_size = arg.dft_size << state->steps[step].npasses;
    if(!state->steps[step].fun_ptr) break;
    state->steps[step].fun_ptr(arg);
    arg.dft_size = next_dft_size;

    swap(next_dst, arg.dst);
    arg.src = next_dst;
  }
}

template<typename V, typename DstCF>
void real_last_pass(
  Int n, typename V::T* src, typename V::T* twiddle, typename V::T* dst)
{
  VEC_TYPEDEFS(V);
  typedef Complex<Vec> C;
  Int src_off = n / 2;
  Int vsrc_off = src_off / V::vec_size;
  Int dst_off = align_size<T>(n / 2 + 1);
  Int vdst_off = dst_off / V::vec_size;

  const Int dst_es = DstCF::elemsz;

  Vec half = V::vec(0.5);

  Complex<T> middle = {src[n / 4], src[src_off + n / 4]};

  for(
    Int i0 = 1, i1 = n / 2 - V::vec_size, iw = 0; 
    i0 <= i1; 
    i0 += V::vec_size, i1 -= V::vec_size, iw += V::vec_size)
  {
    C w = cf::Split::load<V>((Vec*)(twiddle + iw), vsrc_off);
    C s0 = cf::Split::unaligned_load<V>(src +  i0, src_off);
    C s1 = reverse_complex<V>(
        cf::Split::load<V>((Vec*)(src + i1), vsrc_off));

    //printf("%f %f %f %f %f %f\n", w.re, w.im, s0.re, s0.im, s1.re, s1.im);

    C a = (s0 + s1.adj()) * half;
    C b = ((s0 - s1.adj()) * w) * half;

    C d0 = a + b.mul_neg_i();
    C d1 = a.adj() + b.adj().mul_neg_i();

    DstCF::template unaligned_store<V>(d0, dst + i0 * dst_es, dst_off);
    DstCF::template store<V>(
      reverse_complex<V>(d1), (Vec*)(dst + i1 * dst_es), vdst_off);
  }

  // fixes the aliasing bug
  DstCF::template store<Scalar<T>>(middle.adj(), dst + n / 4, dst_off);

  {
    Complex<T> r0 = {src[0], src[src_off]};
    DstCF::template store<Scalar<T>>({r0.re + r0.im, 0}, dst, dst_off);
    DstCF::template store<Scalar<T>>({r0.re - r0.im, 0}, dst + n / 2, dst_off);
  }
}

template<typename T>
struct RealState
{
  State<T>* state;
  T* twiddle;
  void (*last_pass)(Int, T*, T*, T*);
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

template<typename V, typename DstCF>
RealState<typename V::T>* rfft_state(Int n, void* ptr)
{
  VEC_TYPEDEFS(V);
  RealState<T>* r = (RealState<T>*)(Uint(ptr) + real_state_struct_offset<V>(n));
  r->state = fft_state<V, cf::Scal, cf::Split>(n / 2, ptr);

  r->state->num_copies++; // causes fft() to put the result in state->working
  r->twiddle = (T*)(Uint(ptr) + real_state_twiddle_offset<V>(n));
  r->last_pass = &real_last_pass<V, DstCF>;
  
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
  fft(state->state, src, dst); // the intermediate result is now in state->working
  state->last_pass(
    state->state->n * 2,
    state->state->working,
    state->twiddle,
    dst);
}

