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

template<typename V>
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

  for(Int si = 0, di = 0; si < n;
    si += cf::Split<V>::stride, di += cf::Vec<V>::stride)
  {
    cf::Vec<V>::store(cf::Split<V>::load(dst + si, n), state.working + di, 0);
  }

  copy(state.working, 2 * n, dst);

  // It's all in Vec format after this point
  typedef cf::Vec<V> CF;

  Int dft_size = 1;
  for(Int s = 0; s < state.nsteps; s++)
  {
    auto step = state.steps[s];
    //printf("nsteps %d npasses %d\n", step.nsteps, step.npasses);

    if(step.npasses == 2 && dft_size >= V::vec_size)
    {
      auto src_row0 = state.working + (n - 4 * dft_size) * CF::idx_ratio;
      auto dst_row0 = dst + (n - 4 * dft_size) * CF::idx_ratio;
      for(Int i = 0; i < dft_size * CF::idx_ratio; i += CF::stride)
        store_two_pass_twiddle<V>(CF::load(src_row0 + i, 0), dst_row0 + 3 * i);
    }
    else if(step.npasses == 3 && dft_size >= V::vec_size)
    {
      auto src_row0 = state.working + (n - 4 * dft_size) * CF::idx_ratio;
      auto src_row1 = state.working + (n - 8 * dft_size) * CF::idx_ratio;
      auto dst_row1 = dst + (n - 8 * dft_size) * CF::idx_ratio;
      for(Int i = 0; i < dft_size * CF::idx_ratio; i += CF::stride)
      {
        store_two_pass_twiddle<V>(CF::load(src_row0 + i, 0), dst_row1 + 5 * i);
        CF::store(
          CF::load(src_row1 + i, 0),
          dst_row1 + 5 * i + 3 * CF::stride, 0);
        
        CF::store(
          CF::load(src_row1 + i + dft_size * CF::idx_ratio, 0),
          dst_row1 + 5 * i + 4 * CF::stride, 0);
      }
    } 
    else if(step.npasses == 4 && dft_size >= V::vec_size)
    {
      auto src_row0 = state.working + (n - 4 * dft_size) * CF::idx_ratio;
      auto dst_row0 = dst + (n - 4 * dft_size) * CF::idx_ratio;
      for(Int i = 0; i < dft_size * CF::idx_ratio; i += CF::stride)
        store_two_pass_twiddle<V>(CF::load(src_row0 + i, 0), dst_row0 + 3 * i);
      
      auto src_row2 = state.working + (n - 16 * dft_size) * CF::idx_ratio;
      auto dst_row2 = dst + (n - 16 * dft_size) * CF::idx_ratio;
      for(Int i = 0; i < 4 * dft_size * CF::idx_ratio; i += CF::stride)
        store_two_pass_twiddle<V>(CF::load(src_row2 + i, 0), dst_row2 + 3 * i);
    }

    dft_size <<= step.npasses;
  }
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
    C a0 = SrcCf::load(src0, n);
    C a1 = SrcCf::load(src1, n);
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
    C a0 = SrcCf::load(src0, n);
    C a1 = SrcCf::load(src1, n);
    C a2 = SrcCf::load(src2, n);
    C a3 = SrcCf::load(src3, n);
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
  typename V::T* src,
  typename V::T* dst)
{
  VEC_TYPEDEFS(V);
  Int l = n / 8 * SrcCf::idx_ratio;
  Vec invsqrt2 = V::vec(SinCosTable<T>::cos[2]);

  for(T* end = dst + n * VecCf::idx_ratio; dst < end;)
  {
    C c0, c1, c2, c3;
    {
      C a0 = SrcCf::load(src + 0 * l, n);
      C a1 = SrcCf::load(src + 2 * l, n);
      C a2 = SrcCf::load(src + 4 * l, n);
      C a3 = SrcCf::load(src + 6 * l, n);
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
      C a0 = SrcCf::load(src + 1 * l, n);
      C a1 = SrcCf::load(src + 3 * l, n);
      C a2 = SrcCf::load(src + 5 * l, n);
      C a3 = SrcCf::load(src + 7 * l, n);
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

    C d[8];
    V::transpose(
      c0.re + mul0.re, c1.re + mul1.re, c2.re + mul2.re, c3.re + mul3.re,
      c0.re - mul0.re, c1.re - mul1.re, c2.re - mul2.re, c3.re - mul3.re,
      d[0].re, d[1].re, d[2].re, d[3].re, d[4].re, d[5].re, d[6].re, d[7].re);

    V::transpose(
      c0.im + mul0.im, c1.im + mul1.im, c2.im + mul2.im, c3.im + mul3.im,
      c0.im - mul0.im, c1.im - mul1.im, c2.im - mul2.im, c3.im - mul3.im,
      d[0].im, d[1].im, d[2].im, d[3].im, d[4].im, d[5].im, d[6].im, d[7].im);

    for(Int i = 0; i < 8; i++) VecCf::store(d[i], dst + i * VecCf::stride, 0);

    dst += 8 * VecCf::stride;
    if(src == end) break;
  }
}

template<typename V, typename SrcCf>
void first_three_passes(const Arg<typename V::T>& arg)
{
  first_three_passes_impl<V, SrcCf>(arg.n, arg.src, arg.dst);
}

template<typename V, typename SrcCf, Int n>
void first_three_passes_ct_size(const Arg<typename V::T>& arg)
{
  first_three_passes_impl<V, SrcCf>(n, arg.src, arg.dst);
}

template<typename V, typename DstCf>
FORCEINLINE void last_pass_impl(
  Int n,
  typename V::T* src,
  typename V::T* twiddle,
  typename V::T* dst)
{
  VEC_TYPEDEFS(V);
  Int vdft_size = n / 2 / V::vec_size;
  for(Int i0 = 0, i1 = vdft_size; i0 < vdft_size; i0++, i1++)
  {
    auto a = VecCf::load(src + i0 * VecCf::stride, 0);
    auto b = VecCf::load(src + i1 * VecCf::stride, 0);
    auto mul = b * VecCf::load(twiddle + i0 * VecCf::stride, 0);
    DstCf::store(a + mul, dst + i0 * DstCf::stride, n);
    DstCf::store(a - mul, dst + i1 * DstCf::stride, n);
  }
}

template<typename V, typename DstCf>
void last_pass_vec(const Arg<typename V::T>& arg)
{
  last_pass_impl<V, DstCf>(arg.n, arg.src, arg.twiddle, arg.dst);
}

template<typename V, typename DstCf, Int n>
void last_pass_vec_ct_size(const Arg<typename V::T>& arg)
{
  last_pass_impl<V, DstCf>(n, arg.src, arg.twiddle, arg.dst);
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

#if 0 //TODO
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
#endif

template<typename V, typename DstCf>
void two_passes(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  typedef Complex<Vec> C;

  Int n = arg.n;
  Int dft_size = arg.dft_size;
  auto src = arg.src;
  auto tw = arg.twiddle + (VecCf::idx_ratio) * (n - 4 * dft_size);
  auto dst = arg.dst;

  Int l1 = n / 4 * VecCf::idx_ratio;
  Int l2 = 2 * l1;
  Int l3 = 3 * l1;

  Int m1 = dft_size * DstCf::idx_ratio;
  Int m2 = 2 * m1;
  Int m3 = 3 * m1;

  for(T* end = src + dft_size * VecCf::idx_ratio; src < end;)
  {
    auto s = src;
    auto d = dst;
    auto tw0 = VecCf::load(tw, 0);
    auto tw1 = VecCf::load(tw + VecCf::stride, 0);
    auto tw2 = VecCf::load(tw + 2 * VecCf::stride, 0);
    src += VecCf::stride;
    tw += 3 * VecCf::stride;
    dst += DstCf::stride;

    for(T* end1 = s + l1;;)
    {
      C d0, d1, d2, d3;
      two_passes_inner(
        VecCf::load(s, 0), VecCf::load(s + l1, 0),
        VecCf::load(s + l2, 0), VecCf::load(s + l3, 0),
        d0, d1, d2, d3, tw0, tw1, tw2);

      s += dft_size * VecCf::idx_ratio;

      DstCf::store(d0, d, n);
      DstCf::store(d1, d + m1, n);
      DstCf::store(d2, d + m2, n);
      DstCf::store(d3, d + m3, n);

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

#if 1 //TODO
template<
  Int npasses,
  Int chunk_size,
  bool src_is_strided,
  bool dst_is_strided,
  typename DstCf,
  typename V>
FORCEINLINE void two_passes_strided_impl(
  Int vn,
  Int nchunks,
  Int initial_dft_size,
  Int offset,
  typename V::T* src0,
  typename V::T* twiddle_start,
  typename V::T* dst0)
{
  VEC_TYPEDEFS(V);
  namespace im = index_mapping;

  Int n_ = vn * V::vec_size;
  //printf("npasses %d offset %d\n", npasses, offset);
 
  Int l = nchunks * chunk_size / 4;
  Int dft_size = initial_dft_size << npasses;
  Int m = min(initial_dft_size, chunk_size) << npasses;

  T* twiddle = twiddle_start + (vn - 4 * dft_size) * VecCf::stride;

  bool is_large = initial_dft_size >= chunk_size;
  Int soffset = im::offset(offset, initial_dft_size, npasses, is_large);
  Int doffset = im::offset(offset, initial_dft_size, npasses + 2, is_large);
  Int chunk_stride = vn / nchunks;

  Int sstride = 
    src_is_strided ? im::to_strided_index(
      l, vn, chunk_stride, chunk_size, initial_dft_size, npasses, is_large)
    : l;

  Int dstride = 
    dst_is_strided ? im::to_strided_index(
      m, vn, chunk_stride, chunk_size, initial_dft_size, npasses + 2, is_large)
    : m;

  T* src1 = src0 + sstride * VecCf::stride;
  T* src2 = src1 + sstride * VecCf::stride;
  T* src3 = src2 + sstride * VecCf::stride;
  
  T* dst1 = dst0 + dstride * DstCf::stride;
  T* dst2 = dst1 + dstride * DstCf::stride;
  T* dst3 = dst2 + dstride * DstCf::stride;

  if(is_large)
    for(Int i = 0; i < l; i += m)
    {
      for(Int j = 0; j < m; j += chunk_size)
      {
        Int s = i + j;
        Int d = 4 * i + j;

        Int strided_s = soffset + im::to_strided_index(
          s, vn, chunk_stride, chunk_size, initial_dft_size, npasses, true);

        Int strided_d = doffset + im::to_strided_index(
          d, vn, chunk_stride, chunk_size, initial_dft_size, npasses + 2, true);

        if(src_is_strided) s = strided_s;
        if(dst_is_strided) d = strided_d; 

        for(Int k = 0; k < chunk_size; k++)
        {
          auto tw = twiddle + 3 * (strided_s & (dft_size - 1)) * VecCf::stride;
          
          C tw0 = VecCf::load(tw, 0);
          C tw1 = VecCf::load(tw + VecCf::stride, 0);
          C tw2 = VecCf::load(tw + 2 * VecCf::stride, 0);

          C s0 = VecCf::load(src0 + s * VecCf::stride, 0);
          C s1 = VecCf::load(src1 + s * VecCf::stride, 0);
          C s2 = VecCf::load(src2 + s * VecCf::stride, 0);
          C s3 = VecCf::load(src3 + s * VecCf::stride, 0);

          C d0, d1, d2, d3; 
          two_passes_inner(s0, s1, s2, s3, d0, d1, d2, d3, tw0, tw1, tw2);

          DstCf::store(d0, dst0 + d * DstCf::stride, n_);  
          DstCf::store(d1, dst1 + d * DstCf::stride, n_);  
          DstCf::store(d2, dst2 + d * DstCf::stride, n_);  
          DstCf::store(d3, dst3 + d * DstCf::stride, n_);  

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
        s, vn, chunk_stride, chunk_size, initial_dft_size, npasses, false);

      Int strided_d = doffset + im::to_strided_index(
        d, vn, chunk_stride, chunk_size, initial_dft_size, npasses + 2, false);

      if(src_is_strided) s = strided_s;
      if(dst_is_strided) d = strided_d;

      for(Int j = 0; j < m; j++)
      {
        auto tw = twiddle + 3 * (strided_s & (dft_size - 1)) * VecCf::stride;
        
        C tw0 = VecCf::load(tw, 0);
        C tw1 = VecCf::load(tw + VecCf::stride, 0);
        C tw2 = VecCf::load(tw + 2 * VecCf::stride, 0);

        C s0 = VecCf::load(src0 + s * VecCf::stride, 0);
        C s1 = VecCf::load(src1 + s * VecCf::stride, 0);
        C s2 = VecCf::load(src2 + s * VecCf::stride, 0);
        C s3 = VecCf::load(src3 + s * VecCf::stride, 0);

        C d0, d1, d2, d3; 
        two_passes_inner(s0, s1, s2, s3, d0, d1, d2, d3, tw0, tw1, tw2);

        DstCf::store(d0, dst0 + d * DstCf::stride, n_);
        DstCf::store(d1, dst1 + d * DstCf::stride, n_);
        DstCf::store(d2, dst2 + d * DstCf::stride, n_);
        DstCf::store(d3, dst3 + d * DstCf::stride, n_);

        s++;
        d++;
        strided_s++;
      }
    }
}
#endif

template<typename V, typename DstCf>
void four_passes(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  Int n = arg.n / V::vec_size;
  Int dft_size = arg.dft_size / V::vec_size;
  auto src = arg.src;
  auto twiddle = arg.twiddle;
  auto dst = arg.dst;
  
  typedef Complex<Vec> C;

  const Int chunk_size = 16;
  const Int nchunks = 16;
  Int stride = n / nchunks;
  
  Uint bitmask = align_bytes - 1;
  char mem[2 * chunk_size * nchunks * sizeof(Vec) + bitmask];
  T* working = (T*)((Uint(mem) + bitmask) & ~bitmask);

  for(Int offset = 0; offset < stride; offset += chunk_size)
  {
    two_passes_strided_impl<0, chunk_size, true, false, cf::Vec<V>, V>(
      n, nchunks, dft_size, offset, src, twiddle, working);
    two_passes_strided_impl<2, chunk_size, false, true, DstCf, V>(
      n, nchunks, dft_size, offset, working, twiddle, dst);
  }
}

template<typename V, typename DstCf>
FORCEINLINE void last_three_passes_impl(
  Int n,
  typename V::T* src,
  typename V::T* twiddle,
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

  for(auto end = src + l1 * VecCf::stride;;)
  {
    C a[8];
    C a0, a1, a2, a3, a4, a5, a6, a7;
    {
      C tw0 = VecCf::load(twiddle, 0);
      C tw1 = VecCf::load(twiddle + VecCf::stride, 0);
      C tw2 = VecCf::load(twiddle + 2 * VecCf::stride, 0);

      {
        C mul0 =       VecCf::load(src, 0);
        C mul1 = tw0 * VecCf::load(src + l2 * VecCf::stride, 0);
        C mul2 = tw1 * VecCf::load(src + l4 * VecCf::stride, 0);
        C mul3 = tw2 * VecCf::load(src + l6 * VecCf::stride, 0);

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
        C mul0 =       VecCf::load(src + l1 * VecCf::stride, 0);
        C mul1 = tw0 * VecCf::load(src + l3 * VecCf::stride, 0);
        C mul2 = tw1 * VecCf::load(src + l5 * VecCf::stride, 0);
        C mul3 = tw2 * VecCf::load(src + l7 * VecCf::stride, 0);

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
        DstCf::store(a0 + mul, dst + 0, n);
        DstCf::store(a0 - mul, dst + l4 * DstCf::stride, n);
      }

      {
        auto mul = tw3.mul_neg_i() * a6;
        DstCf::store(a2 + mul, dst + l2 * DstCf::stride, n);
        DstCf::store(a2 - mul, dst + l6 * DstCf::stride, n);
      }
    }

    {
      C tw4 = VecCf::load(twiddle + 4 * VecCf::stride, 0);
      {
        auto mul = tw4 * a5;
        DstCf::store(a1 + mul, dst + l1 * DstCf::stride, n);
        DstCf::store(a1 - mul, dst + l5 * DstCf::stride, n);
      }

      {
        auto mul = tw4.mul_neg_i() * a7;
        DstCf::store(a3 + mul, dst + l3 * DstCf::stride, n);
        DstCf::store(a3 - mul, dst + l7 * DstCf::stride, n);
      }
    }

    src += VecCf::stride;
    dst += DstCf::stride;
    twiddle += 5 * VecCf::stride;
    if(src == end) break;
  }
}

template<typename V, typename DstCf>
void last_three_passes_vec(const Arg<typename V::T>& arg)
{
  last_three_passes_impl<V, DstCf>(arg.n, arg.src, arg.twiddle, arg.dst);
}

template<typename V, typename DstCf, Int n>
void last_three_passes_vec_ct_size(const Arg<typename V::T>& arg)
{
  last_three_passes_impl<V, DstCf>(n, arg.src, arg.twiddle, arg.dst);
}

template<typename V, typename SrcCf, typename DstCf>
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
        step.fun_ptr = &first_three_passes_ct_size<V, SrcCf, 8 * V::vec_size>;
      else
        step.fun_ptr = &first_three_passes<V, SrcCf>;

      step.npasses = 3;
    }
    else if(dft_size == 1 && state.n >= 4 * V::vec_size)
    {
      if(state.n == 4 * V::vec_size)
        step.fun_ptr = &first_two_passes_ct_size<V, SrcCf, 4 * V::vec_size>;
      else if(state.n == 8 * V::vec_size)
        step.fun_ptr = &first_two_passes_ct_size<V, SrcCf, 8 * V::vec_size>;
      else
        step.fun_ptr = &first_two_passes<V, SrcCf>;

      step.npasses = 2;
    }
    else if(dft_size >= V::vec_size)
    {
      if(dft_size * 8 == state.n)
      {
        if(state.n == V::vec_size * 8)
          step.fun_ptr = &last_three_passes_vec_ct_size<V, DstCf, V::vec_size * 8>;
        else
          step.fun_ptr = &last_three_passes_vec<V, DstCf>;

        step.npasses = 3;
      }
      else if(state.n >= large_fft_size && dft_size * 16 == state.n)
      {
        step.fun_ptr = &four_passes<V, DstCf>;
        step.npasses = 4;
      }
      else if(state.n >= large_fft_size && dft_size * 16 < state.n)
      {
        step.fun_ptr = &four_passes<V, cf::Vec<V>>;
        step.npasses = 4;
      }
      else if(dft_size * 4 == state.n)
      {
        step.fun_ptr = &two_passes<V, DstCf>;
        step.npasses = 2;
      }
      else if(dft_size * 4 < state.n)
      {
        step.fun_ptr = &two_passes<V, cf::Vec<V>>;
        step.npasses = 2;
      }
      else
      {
        if(state.n == 2 * V::vec_size)
          step.fun_ptr = &last_pass_vec_ct_size<V, DstCf, 2 * V::vec_size>;
        else if(state.n == 4 * V::vec_size)
          step.fun_ptr = &last_pass_vec_ct_size<V, DstCf, 4 * V::vec_size>;
        else if(state.n == 8 * V::vec_size)
          step.fun_ptr = &last_pass_vec_ct_size<V, DstCf, 8 * V::vec_size>;
        else
          step.fun_ptr = &last_pass_vec<V, DstCf>;

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

template<
  typename V,
  template<typename> class SrcCfT,
  template<typename> class DstCfT>
State<typename V::T>* fft_state(Int n, void* ptr)
{
  VEC_TYPEDEFS(V);
  auto state = (State<T>*)(Uint(ptr) + Uint(state_struct_offset<V>(n)));
  state->n = n;
  state->working = (T*) ptr;
  state->twiddle = state->working + 2 * n;
  state->tiny_twiddle = state->twiddle + 2 * n;
  init_steps<V, SrcCfT<V>, DstCfT<V>>(*state);
  init_twiddle<V>(*state);
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

template<typename V, template<typename> class DstCfT>
void real_last_pass(
  Int n, typename V::T* src, typename V::T* twiddle, typename V::T* dst)
{
  static_assert(!SameType<DstCfT<V>, cf::Vec<V>>::value, "");
  VEC_TYPEDEFS(V);
  Int src_off = n / 2;
  Int dst_off = align_size<T>(n / 2 + 1);

  const Int dst_ratio = DstCfT<V>::idx_ratio;

  Vec half = V::vec(0.5);

  Complex<T> middle = {src[n / 4], src[src_off + n / 4]};

  for(
    Int i0 = 1, i1 = n / 2 - V::vec_size, iw = 0; 
    i0 <= i1; 
    i0 += V::vec_size, i1 -= V::vec_size, iw += V::vec_size)
  {
    C w = cf::Split<V>::load(twiddle + iw, src_off);
    C s0 = cf::Split<V>::unaligned_load(src + i0, src_off);
    C s1 = reverse_complex<V>(cf::Split<V>::load(src + i1, src_off));

    //printf("%f %f %f %f %f %f\n", w.re, w.im, s0.re, s0.im, s1.re, s1.im);

    C a = (s0 + s1.adj()) * half;
    C b = ((s0 - s1.adj()) * w) * half;

    C d0 = a + b.mul_neg_i();
    C d1 = a.adj() + b.adj().mul_neg_i();

    DstCfT<V>::unaligned_store(d0, dst + i0 * dst_ratio, dst_off);
    DstCfT<V>::store(reverse_complex<V>(d1), dst + i1 * dst_ratio, dst_off);
  }

  // fixes the aliasing bug
  DstCfT<Scalar<T>>::store(middle.adj(), dst + n / 4 * dst_ratio, dst_off);

  {
    Complex<T> r0 = {src[0], src[src_off]};
    DstCfT<Scalar<T>>::store({r0.re + r0.im, 0}, dst, dst_off);
    DstCfT<Scalar<T>>::store({r0.re - r0.im, 0}, dst + n / 2 * dst_ratio, dst_off);
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

template<typename V, template<typename> class DstCfT>
RealState<typename V::T>* rfft_state(Int n, void* ptr)
{
  VEC_TYPEDEFS(V);
  RealState<T>* r = (RealState<T>*)(Uint(ptr) + real_state_struct_offset<V>(n));
  r->state = fft_state<V, cf::Scal, cf::Split>(n / 2, ptr);

  r->state->num_copies++; // causes fft() to put the result in state->working
  r->twiddle = (T*)(Uint(ptr) + real_state_twiddle_offset<V>(n));
  r->last_pass = &real_last_pass<V, DstCfT>;
  
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
    state->state->n * 2, state->state->working, state->twiddle, dst);
}
