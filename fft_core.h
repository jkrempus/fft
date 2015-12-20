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

    template<typename V>
    static FORCEINLINE Complex<V> load(typename V::T* ptr, Int off)
    {
      return { V::load(ptr), V::load(ptr + off)};
    }

    template<typename V>
    static FORCEINLINE Complex<V> unaligned_load(typename V::T* ptr, Int off)
    {
      return {
        V::unaligned_load(ptr),
        V::unaligned_load(ptr + off)};
    }

    template<typename V>
    static FORCEINLINE void store(Complex<V> a, typename V::T* ptr, Int off)
    {
      V::store(a.re, ptr);
      V::store(a.im, ptr + off);
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

    template<typename V>
    static FORCEINLINE Complex<V> load(typename V::T* ptr, Int off)
    {
      return {V::load(ptr), V::load(ptr + V::vec_size)};
    }

    template<typename V>
    static FORCEINLINE Complex<V> unaligned_load(typename V::T* ptr, Int off)
    {
      return {V::unaligned_load(ptr), V::unaligned_load(ptr + V::vec_size)};
    }

    template<typename V>
    static FORCEINLINE void store(Complex<V> a, typename V::T* ptr, Int off)
    {
      V::store(a.re, ptr);
      V::store(a.im, ptr + V::vec_size);
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

    template<typename V>
    static FORCEINLINE Complex<V> load(typename V::T* ptr, Int off)
    {
      Complex<V> r;
      V::deinterleave(
        V::load(ptr), V::load(ptr + V::vec_size),
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

    template<typename V>
    static FORCEINLINE void store(Complex<V> a, typename V::T* ptr, Int off)
    {
      V::interleave(a.re, a.im, a.re, a.im);
      V::store(a.re, ptr);
      V::store(a.im, ptr + V::vec_size);
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

    template<typename V>
    static FORCEINLINE Complex<V> load(typename V::T* ptr, Int off)
    {
      auto a = InputCf::template load<V>(ptr, off);
      return {a.im, a.re};
    }

    template<typename V>
    static FORCEINLINE Complex<V> unaligned_load(typename V::T* ptr, Int off)
    {
      auto a = InputCf::template unaligned_load<V>(ptr, off);
      return {a.im, a.re};
    }

    template<typename V>
    static FORCEINLINE void store(Complex<V> a, typename V::T* ptr, Int off)
    {
      InputCf::template store<V>({a.im, a.re}, ptr, off);
    }

    template<typename V>
    static FORCEINLINE void unaligned_store(
      Complex<V> a, typename V::T* ptr, Int off)
    {
      InputCf::template unaligned_store<V>({a.im, a.re}, ptr, off);
    }
  };
}

template<typename V, typename Cf>
FORCEINLINE Complex<V> load(typename V::T* ptr, Int off)
{
  return Cf::template load<V>(ptr, off);
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
  for(Int i = 0; i < n; i++)
  {
    auto c = load<V, SrcCf>(src + i * stride<V, SrcCf>(), src_off);
    DstCf::store(c, dst + i * stride<V, DstCf>(), dst_off);
  }
}

template<typename V>
FORCEINLINE Complex<V> reverse_complex(Complex<V> a)
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

  static Vec load(T* p) { return _mm256_load_ps(p); }
  static Vec unaligned_load(T* p) { return _mm256_loadu_ps(p); }
  static void store(Vec val, T* p) { _mm256_store_ps(p, val); }
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

namespace onedim
{
template<typename T>
struct Fft
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

template<typename T> struct Ifft;

template<typename V, typename DstCf>
void init_br_table(Fft<typename V::T>& state)
{
  if(state.steps[state.nsteps - 1].npasses == 0) return;
  auto len = (state.n / V::vec_size) >> state.steps[state.nsteps - 1].npasses;
  for(BitReversed br(len); br.i < len; br.advance())
    state.br_table[br.i] = br.br * stride<V, DstCf>();
}

template<typename V, Int dft_size, typename SrcCf>
void ct_dft_size_pass(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  Int n = arg.n;
  auto src0 = arg.src;
  auto src1 = arg.src + n * SrcCf::idx_ratio / 2;
  auto dst = arg.dst;
  auto tw = load<V, cf::Vec>(
    arg.tiny_twiddle + tiny_log2(dft_size) * stride<V, cf::Vec>(), 0);

  for(auto end = src1; src0 < end;)
  {
    C a0 = load<V, SrcCf>(src0, arg.im_off);
    C a1 = load<V, SrcCf>(src1, arg.im_off);
    src0 += stride<V, SrcCf>();
    src1 += stride<V, SrcCf>();

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
    cf::Vec::store(d0, dst, 0);
    cf::Vec::store(d1, dst + stride<V, cf::Vec>(), 0);

    dst += 2 * stride<V, cf::Vec>();
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
    C a0 = load<V, SrcCf>(src0, arg.im_off);
    C a1 = load<V, SrcCf>(src1, arg.im_off);
    C a2 = load<V, SrcCf>(src2, arg.im_off);
    C a3 = load<V, SrcCf>(src3, arg.im_off);
    src0 += stride<V, SrcCf>();
    src1 += stride<V, SrcCf>();
    src2 += stride<V, SrcCf>();
    src3 += stride<V, SrcCf>();

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

    cf::Vec::store(d0, dst, 0); 
    cf::Vec::store(d1, dst + stride<V, cf::Vec>(), 0); 
    cf::Vec::store(d2, dst + 2 * stride<V, cf::Vec>(), 0); 
    cf::Vec::store(d3, dst + 3 * stride<V, cf::Vec>(), 0); 
    dst += 4 * stride<V, cf::Vec>();
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
    auto a0 = load<V, SrcCf>(src + i0 * stride<V, SrcCf>(), arg.im_off);
    auto a1 = load<V, SrcCf>(src + i1 * stride<V, SrcCf>(), arg.im_off);
    cf::Vec::store(a0 + a1, dst + i0 * stride<V, cf::Vec>(), 0);
    cf::Vec::store(a0 - a1, dst + i1 * stride<V, cf::Vec>(), 0);
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

  for(T* end = dst + dst_chunk_size * cf::Vec::idx_ratio; dst < end;)
  {
    C c0, c1, c2, c3;
    {
      C a0 = load<V, SrcCf>(src + 0 * l, im_off);
      C a1 = load<V, SrcCf>(src + 2 * l, im_off);
      C a2 = load<V, SrcCf>(src + 4 * l, im_off);
      C a3 = load<V, SrcCf>(src + 6 * l, im_off);
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
      C a0 = load<V, SrcCf>(src + 1 * l, im_off);
      C a1 = load<V, SrcCf>(src + 3 * l, im_off);
      C a2 = load<V, SrcCf>(src + 5 * l, im_off);
      C a3 = load<V, SrcCf>(src + 7 * l, im_off);
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

    src += stride<V, SrcCf>();

    {
      Vec d[8];
      V::transpose(
        c0.re + mul0.re, c1.re + mul1.re, c2.re + mul2.re, c3.re + mul3.re,
        c0.re - mul0.re, c1.re - mul1.re, c2.re - mul2.re, c3.re - mul3.re,
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);

      for(Int i = 0; i < 8; i++) V::store(d[i], dst + i * stride<V, cf::Vec>());
    }

    {
      Vec d[8];
      V::transpose(
        c0.im + mul0.im, c1.im + mul1.im, c2.im + mul2.im, c3.im + mul3.im,
        c0.im - mul0.im, c1.im - mul1.im, c2.im - mul2.im, c3.im - mul3.im,
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);

      for(Int i = 0; i < 8; i++)
        V::store(d[i], dst + i * stride<V, cf::Vec>() + V::vec_size);
    }

    dst += 8 * stride<V, cf::Vec>();
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

template<typename V>
FORCEINLINE void two_passes_inner(
  Complex<V> src0, Complex<V> src1, Complex<V> src2, Complex<V> src3,
  Complex<V>& dst0, Complex<V>& dst1, Complex<V>& dst2, Complex<V>& dst3,
  Complex<V> tw0, Complex<V> tw1, Complex<V> tw2)
{
  typedef Complex<V> C;
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

  auto off1 = (arg.n >> log2(dft_size)) / 4 * stride<V, cf::Vec>();
  auto off2 = off1 + off1;
  auto off3 = off2 + off1;
  
  auto start = arg.start_offset * cf::Vec::idx_ratio;
  auto end = arg.end_offset * cf::Vec::idx_ratio;

  auto tw = arg.twiddle + cf::Vec::idx_ratio * (n - 4 * dft_size);
  if(start != 0)
    tw += 3 * stride<V, cf::Vec>() * (start >> log2(off1 + off3));

  for(auto p = src + start; p < src + end;)
  {
    auto tw0 = load<V, cf::Vec>(tw, 0);
    auto tw1 = load<V, cf::Vec>(tw + stride<V, cf::Vec>(), 0);
    auto tw2 = load<V, cf::Vec>(tw + 2 * stride<V, cf::Vec>(), 0);
    tw += 3 * stride<V, cf::Vec>();

    for(auto end1 = p + off1;;)
    {
      ASSERT(p >= arg.src);
      ASSERT(p + off3 < arg.src + arg.n * cf::Vec::idx_ratio);

      C d0, d1, d2, d3;
      two_passes_inner(
        load<V, cf::Vec>(p, 0), load<V, cf::Vec>(p + off1, 0),
        load<V, cf::Vec>(p + off2, 0), load<V, cf::Vec>(p + off3, 0),
        d0, d1, d2, d3, tw0, tw1, tw2);

      cf::Vec::store(d0, p, 0);
      cf::Vec::store(d2, p + off1, 0);
      cf::Vec::store(d1, p + off2, 0);
      cf::Vec::store(d3, p + off3, 0);

      p += stride<V, cf::Vec>();
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
  auto dst1 = dst0 + vn / 4 * stride<V, DstCf>(); 
  auto dst2 = dst1 + vn / 4 * stride<V, DstCf>(); 
  auto dst3 = dst2 + vn / 4 * stride<V, DstCf>(); 

  for(auto end = src + vn * stride<V, cf::Vec>(); src < end; )
  {
    auto tw0 = load<V, cf::Vec>(tw, 0);
    auto tw1 = load<V, cf::Vec>(tw + stride<V, cf::Vec>(), 0);
    auto tw2 = load<V, cf::Vec>(tw + 2 * stride<V, cf::Vec>(), 0);
    tw += 3 * stride<V, cf::Vec>();

    C d0, d1, d2, d3;
    two_passes_inner(
      load<V, cf::Vec>(src, 0),
      load<V, cf::Vec>(src + stride<V, cf::Vec>(), 0),
      load<V, cf::Vec>(src + 2 * stride<V, cf::Vec>(), 0),
      load<V, cf::Vec>(src + 3 * stride<V, cf::Vec>(), 0),
      d0, d1, d2, d3, tw0, tw1, tw2);

    src += 4 * stride<V, cf::Vec>();

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
    br_table[br.i] = br.br * stride<V, DstCf>();

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
  auto dst1 = dst0 + vn / 2 * stride<V, DstCf>(); 

  for(auto end = src + vn * stride<V, cf::Vec>(); src < end; )
  {
    C a0 = load<V, cf::Vec>(src, 0);
    C mul = load<V, cf::Vec>(src + stride<V, cf::Vec>(), 0) * load<V, cf::Vec>(tw, 0);
    tw += stride<V, cf::Vec>();
    src += 2 * stride<V, cf::Vec>();

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
    br_table[br.i] = br.br * stride<V, DstCf>();

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
        load<V, cf::Vec>(arg.src + br.i * stride<V, cf::Vec>(), 0),
        arg.dst + br.br * stride<V, DstCf>(),
        im_off);
  }
  else
  {
    const Int br_table[m] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
    const Int memsize = m * m * cf::Vec::idx_ratio * V::vec_size * sizeof(T);
    AlignedMemory<memsize, align_bytes> mem;
    auto working = (T*) mem.get();
    Int stride_ = vn / m;

    for(BitReversed br(vn / (m * m)); br.i < vn / (m * m); br.advance())
    {
      T* src = arg.src + br.i * m * stride<V, cf::Vec>();
      for(Int i0 = 0; i0 < m; i0++)
        for(Int i1 = 0; i1 < m; i1++)
          cf::Vec::store(
            load<V, cf::Vec>(src + (i0 * stride_ + i1) * stride<V, cf::Vec>(), 0),
            working + (i0 * m + i1) * stride<V, cf::Vec>(), 0);

      T* dst = arg.dst + br.br * m * stride<V, DstCf>();
      for(Int i0 = 0; i0 < m; i0++)
      {
        T* s = working + br_table[i0] * stride<V, cf::Vec>();
        T* d = dst + i0 * stride_ * stride<V, DstCf>();
        for(Int i1 = 0; i1 < m; i1++)
          DstCf::store(
            load<V, cf::Vec>(s + (br_table[i1] << 4) * stride<V, cf::Vec>(), 0),
            d + i1 * stride<V, DstCf>(), im_off);
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
      C tw0 = load<V, cf::Vec>(twiddle, 0);
      C tw1 = load<V, cf::Vec>(twiddle + stride<V, cf::Vec>(), 0);
      C tw2 = load<V, cf::Vec>(twiddle + 2 * stride<V, cf::Vec>(), 0);

      {
        C mul0 =       load<V, cf::Vec>(s, 0);
        C mul1 = tw0 * load<V, cf::Vec>(s + 2 * stride<V, cf::Vec>(), 0);
        C mul2 = tw1 * load<V, cf::Vec>(s + 4 * stride<V, cf::Vec>(), 0);
        C mul3 = tw2 * load<V, cf::Vec>(s + 6 * stride<V, cf::Vec>(), 0);

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
        C mul0 =       load<V, cf::Vec>(s + 1 * stride<V, cf::Vec>(), 0);
        C mul1 = tw0 * load<V, cf::Vec>(s + 3 * stride<V, cf::Vec>(), 0);
        C mul2 = tw1 * load<V, cf::Vec>(s + 5 * stride<V, cf::Vec>(), 0);
        C mul3 = tw2 * load<V, cf::Vec>(s + 7 * stride<V, cf::Vec>(), 0);

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
      C tw3 = load<V, cf::Vec>(twiddle + 3 * stride<V, cf::Vec>(), 0);
      {
        auto mul = tw3 * a4;
        DstCf::store(a0 + mul, d + 0, im_off);
        DstCf::store(a0 - mul, d + l4 * stride<V, DstCf>(), im_off);
      }

      {
        auto mul = tw3.mul_neg_i() * a6;
        DstCf::store(a2 + mul, d + l2 * stride<V, DstCf>(), im_off);
        DstCf::store(a2 - mul, d + l6 * stride<V, DstCf>(), im_off);
      }
    }

    {
      C tw4 = load<V, cf::Vec>(twiddle + 4 * stride<V, cf::Vec>(), 0);
      {
        auto mul = tw4 * a5;
        DstCf::store(a1 + mul, d + l1 * stride<V, DstCf>(), im_off);
        DstCf::store(a1 - mul, d + l5 * stride<V, DstCf>(), im_off);
      }

      {
        auto mul = tw4.mul_neg_i() * a7;
        DstCf::store(a3 + mul, d + l3 * stride<V, DstCf>(), im_off);
        DstCf::store(a3 - mul, d + l7 * stride<V, DstCf>(), im_off);
      }
    }

    s += 8 * stride<V, cf::Vec>();
    twiddle += 5 * stride<V, cf::Vec>();
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
  
  auto start = arg.start_offset * cf::Vec::idx_ratio;
  auto end = arg.end_offset * cf::Vec::idx_ratio;
  
  auto src = arg.src;
  T* twiddle = arg.twiddle + start * 5 / 8;

  for(auto p = src + start; p < src + end;)
  {
    C a0, a1, a2, a3, a4, a5, a6, a7;
    {
      C tw0 = load<V, cf::Vec>(twiddle, 0);
      C tw1 = load<V, cf::Vec>(twiddle + stride<V, cf::Vec>(), 0);
      C tw2 = load<V, cf::Vec>(twiddle + 2 * stride<V, cf::Vec>(), 0);

      {
        C mul0 =       load<V, cf::Vec>(p, 0);
        C mul1 = tw0 * load<V, cf::Vec>(p + 2 * stride<V, cf::Vec>(), 0);
        C mul2 = tw1 * load<V, cf::Vec>(p + 4 * stride<V, cf::Vec>(), 0);
        C mul3 = tw2 * load<V, cf::Vec>(p + 6 * stride<V, cf::Vec>(), 0);

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
        C mul0 =       load<V, cf::Vec>(p + 1 * stride<V, cf::Vec>(), 0);
        C mul1 = tw0 * load<V, cf::Vec>(p + 3 * stride<V, cf::Vec>(), 0);
        C mul2 = tw1 * load<V, cf::Vec>(p + 5 * stride<V, cf::Vec>(), 0);
        C mul3 = tw2 * load<V, cf::Vec>(p + 7 * stride<V, cf::Vec>(), 0);

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
      C tw3 = load<V, cf::Vec>(twiddle + 3 * stride<V, cf::Vec>(), 0);
      {
        auto mul = tw3 * a4;
        cf::Vec::store(a0 + mul, p + 0, 0);
        cf::Vec::store(a0 - mul, p + 1 * stride<V, cf::Vec>(), 0);
      }

      {
        auto mul = tw3.mul_neg_i() * a6;
        cf::Vec::store(a2 + mul, p + 2 * stride<V, cf::Vec>(), 0);
        cf::Vec::store(a2 - mul, p + 3 * stride<V, cf::Vec>(), 0);
      }
    }

    {
      C tw4 = load<V, cf::Vec>(twiddle + 4 * stride<V, cf::Vec>(), 0);
      {
        auto mul = tw4 * a5;
        cf::Vec::store(a1 + mul, p + 4 * stride<V, cf::Vec>(), 0);
        cf::Vec::store(a1 - mul, p + 5 * stride<V, cf::Vec>(), 0);
      }

      {
        auto mul = tw4.mul_neg_i() * a7;
        cf::Vec::store(a3 + mul, p + 6 * stride<V, cf::Vec>(), 0);
        cf::Vec::store(a3 - mul, p + 7 * stride<V, cf::Vec>(), 0);
      }
    }

    p += 8 * stride<V, cf::Vec>();
    twiddle += 5 * stride<V, cf::Vec>();
  }
}

template<typename V, typename SrcCf, typename DstCf, Int n>
void tiny_transform(typename V::T* src, typename V::T* dst, Int im_off)
{
  VEC_TYPEDEFS(V);
  if(n == 1)
  {
    DstCf::store(load<V, SrcCf>(src, im_off), dst, im_off);
  }
  else if(n == 2)
  {
    auto a0 = load<V, SrcCf>(src, im_off);
    auto a1 = load<V, SrcCf>(src + stride<V, SrcCf>(), im_off);
    DstCf::store(a0 + a1, dst, im_off);
    DstCf::store(a0 - a1, dst + stride<V, DstCf>(), im_off);
  }
}

template<typename V, typename SrcCf, typename DstCf>
void init_steps(Fft<typename V::T>& state)
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
        step.fun_ptr = &ct_dft_size_pass<V, 2, cf::Vec>;
      else if(V::vec_size > 4 && dft_size == 4)
        step.fun_ptr = &ct_dft_size_pass<V, 4, cf::Vec>;
      else if(V::vec_size > 8 && dft_size == 8)
        step.fun_ptr = &ct_dft_size_pass<V, 8, cf::Vec>;

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

template<typename V>
Int fft_memsize(Int n)
{
  VEC_TYPEDEFS(V);
  if(n <= V::vec_size && V::vec_size != 1)
    return fft_memsize<Scalar<T>>(n);

  Int sz = 0;
  sz = aligned_increment(sz, sizeof(Fft<T>));
  sz = aligned_increment(sz, sizeof(T) * 2 * n);
  sz = aligned_increment(sz, sizeof(T) * 2 * n);
  sz = aligned_increment(sz, sizeof(T) * 2 * n);
  sz = aligned_increment(sz, tiny_twiddle_bytes<V>() * n);
  sz = aligned_increment(sz, sizeof(Int) * n / V::vec_size / 2);
  return sz;
}

template<
  typename V,
  typename SrcCf,
  typename DstCf>
Fft<typename V::T>* fft_create(Int n, void* ptr)
{
  VEC_TYPEDEFS(V);
  if(n <= V::vec_size && V::vec_size != 1)
    return fft_create<Scalar<T>, SrcCf, DstCf>(n, ptr);

  auto state = (Fft<T>*) ptr;
  state->n = n;
  state->im_off = n;
  ptr = aligned_increment(ptr, sizeof(Fft<T>));

  state->working0 = (T*) ptr;
  ptr = aligned_increment(ptr, sizeof(T) * 2 * n);

  state->working1 = (T*) ptr;
  ptr = aligned_increment(ptr, sizeof(T) * 2 * n);

  state->twiddle = (T*) ptr;
  ptr = aligned_increment(ptr, sizeof(T) * 2 * n);

  state->tiny_twiddle = (T*) ptr;
  ptr = aligned_increment(ptr, tiny_twiddle_bytes<V>());

  state->br_table = (Int*) ptr;

  init_steps<V, SrcCf, DstCf>(*state);

  init_twiddle<V>([state](Int s, Int){ return state->steps[s].npasses; },
    n, state->working0, state->twiddle, state->tiny_twiddle);

  init_br_table<V, DstCf>(*state);

  return state;
}

template<typename V>
Int ifft_memsize(Int n){ return fft_memsize<V>(n); }

template<typename V, typename SrcCf, typename DstCf>
Ifft<typename V::T>* ifft_create(Int n, void* ptr)
{
  return (Ifft<typename V::T>*) fft_create<
    V, cf::Swapped<SrcCf>, cf::Swapped<DstCf>>(n, ptr);
}

template<typename T>
NOINLINE void recursive_passes(
  const Fft<T>* state, Int step, T* p, Int start, Int end)
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
FORCEINLINE void fft_impl(const Fft<T>* state, Int im_off, T* src, T* dst)
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
void fft(const Fft<T>* state, T* src, T* dst)
{
  fft_impl(state, state->n, src, dst);
}

template<typename T>
void ifft(const Ifft<T>* state, T* src, T* dst)
{
  fft_impl((Fft<T>*) state, ((Fft<T>*) state)->n, src, dst);
}

template<
  typename V,
  typename SrcCf,
  typename DstCf,
  bool inverse>
void real_pass(
  Int n,
  typename V::T* src,
  Int src_off,
  typename V::T* twiddle,
  typename V::T* dst,
  Int dst_off)
{
  if(SameType<DstCf, cf::Vec>::value) ASSERT(0);
  VEC_TYPEDEFS(V);

  const Int src_ratio = SrcCf::idx_ratio;
  const Int dst_ratio = DstCf::idx_ratio;

  Vec half = V::vec(0.5);

  typedef Scalar<T> S;
  typedef Complex<S> SC; 
  SC middle = load<S, SrcCf>(src + n / 4 * src_ratio, src_off);

  for(
    Int i0 = 1, i1 = n / 2 - V::vec_size, iw = 0; 
    i0 <= i1; 
    i0 += V::vec_size, i1 -= V::vec_size, iw += V::vec_size)
  {
    C w = load<V, cf::Split>(twiddle + iw, n / 2);
    C s0 = SrcCf::template unaligned_load<V>(src + i0 * src_ratio, src_off);
    C s1 = reverse_complex<V>(load<V, SrcCf>(src + i1 * src_ratio, src_off));

    //printf("%f %f %f %f %f %f\n", s0.re, s0.im, s1.re, s1.im, w.re, w.im);

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

    DstCf::unaligned_store(d0, dst + i0 * dst_ratio, dst_off);
    DstCf::store(reverse_complex<V>(d1), dst + i1 * dst_ratio, dst_off);
  }

  // fixes the aliasing bug
  DstCf::store(
    middle.adj() * (inverse ? 2.0f : 1.0f), dst + n / 4 * dst_ratio, dst_off);

  if(inverse)
  {
    T r0 = load<S, SrcCf>(src, src_off).re;
    T r1 = load<S, SrcCf>(src + n / 2 * src_ratio, src_off).re;
    DstCf::template store<S>({r0 + r1, r0 - r1}, dst, dst_off);
  }
  else
  {
    SC r0 = load<S, SrcCf>(src, src_off);
    DstCf::template store<S>({r0.re + r0.im, 0}, dst, dst_off);
    DstCf::template store<S>(
      {r0.re - r0.im, 0}, dst + n / 2 * dst_ratio, dst_off);
  }
}

template<typename T>
struct Rfft
{
  Fft<T>* state;
  T* twiddle;
  void (*real_pass)(Int, T*, Int, T*, T*, Int);
};

template<typename V>
Int rfft_memsize(Int n)
{
  VEC_TYPEDEFS(V);
  if(V::vec_size != 1 && n <= 2 * V::vec_size)
    return rfft_memsize<Scalar<T>>(n);

  Int sz = 0;
  sz = aligned_increment(sz, sizeof(Rfft<T>));
  sz = aligned_increment(sz, sizeof(T) * n);
  sz = aligned_increment(sz, fft_memsize<V>(n));
  return sz;
}

template<typename V, typename DstCf>
Rfft<typename V::T>* rfft_create(Int n, void* ptr)
{
  VEC_TYPEDEFS(V);
  if(V::vec_size != 1 && n <= 2 * V::vec_size)
    return rfft_create<Scalar<T>, DstCf>(n, ptr);

  Rfft<T>* r = (Rfft<T>*) ptr;
  ptr = aligned_increment(ptr, sizeof(Rfft<T>));

  r->twiddle = (T*) ptr;
  ptr = aligned_increment(ptr, sizeof(T) * n);
 
  r->real_pass = &real_pass<V, cf::Split, DstCf, false>;
  
  Int m =  n / 2;
  compute_twiddle(m, m, r->twiddle, r->twiddle + m);
  copy(r->twiddle + 1, m - 1, r->twiddle);
  copy(r->twiddle + m + 1, m - 1, r->twiddle + m);
  
  r->state = fft_create<V, cf::Scal, cf::Split>(n / 2, ptr);
  return r;
}

template<typename T>
void rfft(const Rfft<T>* state, T* src, T* dst)
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
struct Irfft
{
  Ifft<T>* state;
  T* twiddle;
  void (*real_pass)(Int, T*, Int, T*, T*, Int);
};

template<typename V> Int irfft_memsize(Int n) { return rfft_memsize<V>(n); }

template<typename V, typename SrcCf>
Irfft<typename V::T>* irfft_create(Int n, void* ptr)
{
  VEC_TYPEDEFS(V);
  VEC_TYPEDEFS(V);
  if(V::vec_size != 1 && n <= 2 * V::vec_size)
    return irfft_create<Scalar<T>, SrcCf>(n, ptr);

  auto r = (Irfft<T>*) ptr;
  ptr = aligned_increment(ptr, sizeof(Irfft<T>));

  r->twiddle = (T*) ptr;
  ptr = aligned_increment(ptr, sizeof(T) * n);

  r->real_pass = &real_pass<V, SrcCf, cf::Split, true>;

  Int m =  n / 2;
  compute_twiddle(m, m, r->twiddle, r->twiddle + m);
  copy(r->twiddle + 1, m - 1, r->twiddle);
  copy(r->twiddle + m + 1, m - 1, r->twiddle + m);

  r->state = ifft_create<V, cf::Split, cf::Scal>(n / 2, ptr);
  return r;
}

template<typename T>
void irfft(const Irfft<T>* state, T* src, T* dst)
{
  state->real_pass(
    state->state->n * 2,
    src,
    align_size<T>(state->state->n + 1),
    state->twiddle,
    state->state->working1,
    state->state->n);

  ifft(state->state, state->state->working1, dst);
}
}

namespace multi
{
template<typename SrcRows, typename DstRows>
void first_pass(Int n, const SrcRows& src, const DstRows& dst)
{
  typedef typename SrcRows::V V; VEC_TYPEDEFS(V);

  for(auto i = 0; i < n / 2; i++)
  {
    auto s0 = src.row(i);
    auto s1 = src.row(i + n / 2);
    auto d0 = dst.row(i);
    auto d1 = dst.row(i + n / 2);
    for(auto end = s0 + dst.m * SrcRows::Cf::idx_ratio; s0 < end;)
    {
      C a0 = load<V, typename SrcRows::Cf>(s0, src.im_off);
      C a1 = load<V, typename SrcRows::Cf>(s1, src.im_off);
      DstRows::Cf::store(a0 + a1, d0, dst.im_off);
      DstRows::Cf::store(a0 - a1, d1, dst.im_off);
      s0 += stride<V, typename SrcRows::Cf>();
      s1 += stride<V, typename SrcRows::Cf>();
      d0 += stride<V, typename DstRows::Cf>();
      d1 += stride<V, typename DstRows::Cf>();
    }
  }
}

template<typename SrcRows, typename DstRows>
void first_two_passes(
  Int n, const SrcRows& src , const DstRows& dst)
{
  typedef typename SrcRows::V V; VEC_TYPEDEFS(V);

  for(auto i = 0; i < n / 4; i++)
  {
    auto s0 = src.row(i);
    auto s1 = src.row(i + n / 4);
    auto s2 = src.row(i + 2 * n / 4);
    auto s3 = src.row(i + 3 * n / 4);
    auto d0 = dst.row(i);
    auto d1 = dst.row(i + n / 4);
    auto d2 = dst.row(i + 2 * n / 4);
    auto d3 = dst.row(i + 3 * n / 4);

    for(auto end = s0 + dst.m * SrcRows::Cf::idx_ratio; s0 < end;)
    {
      C a0 = load<V, typename SrcRows::Cf>(s0, src.im_off);
      C a1 = load<V, typename SrcRows::Cf>(s1, src.im_off);
      C a2 = load<V, typename SrcRows::Cf>(s2, src.im_off);
      C a3 = load<V, typename SrcRows::Cf>(s3, src.im_off);

      C b0 = a0 + a2;
      C b2 = a0 - a2;
      C b1 = a1 + a3;
      C b3 = a1 - a3;

      C c0 = b0 + b1;
      C c1 = b0 - b1;
      C c2 = b2 + b3.mul_neg_i();
      C c3 = b2 - b3.mul_neg_i();

      DstRows::Cf::store(c0, d0, dst.im_off);
      DstRows::Cf::store(c1, d1, dst.im_off);
      DstRows::Cf::store(c2, d2, dst.im_off);
      DstRows::Cf::store(c3, d3, dst.im_off);

      s0 += stride<V, typename SrcRows::Cf>();
      s1 += stride<V, typename SrcRows::Cf>();
      s2 += stride<V, typename SrcRows::Cf>();
      s3 += stride<V, typename SrcRows::Cf>();
      d0 += stride<V, typename DstRows::Cf>();
      d1 += stride<V, typename DstRows::Cf>();
      d2 += stride<V, typename DstRows::Cf>();
      d3 += stride<V, typename DstRows::Cf>();
    }
  }
}

template<typename V, typename Cf>
NOINLINE void two_passes_inner(
  typename V::T* a0,
  typename V::T* a1,
  typename V::T* a2,
  typename V::T* a3,
  Complex<V> t0,
  Complex<V> t1,
  Complex<V> t2,
  Int m,
  Int im_off)
{
  VEC_TYPEDEFS(V);
  for(Int i = 0; i < m * Cf::idx_ratio; i += stride<V, Cf>())
  {
    C b0 = load<V, Cf>(a0 + i, im_off);
    C b1 = load<V, Cf>(a1 + i, im_off);
    C b2 = load<V, Cf>(a2 + i, im_off);
    C b3 = load<V, Cf>(a3 + i, im_off);
    onedim::two_passes_inner(b0, b1, b2, b3, b0, b1, b2, b3, t0, t1, t2);
    Cf::store(b0, a0 + i, im_off);
    Cf::store(b2, a1 + i, im_off);
    Cf::store(b1, a2 + i, im_off);
    Cf::store(b3, a3 + i, im_off);
  }
}

template<typename Rows>
void two_passes(
  Int n,
  Int dft_size,
  Int start,
  Int end,
  typename Rows::V::T* twiddle,
  const Rows& data)
{
  typedef typename Rows::V V; VEC_TYPEDEFS(V);

  Int stride = end - start;
  ASSERT(stride * dft_size == n);
  auto tw = twiddle + 2 * (n - 4 * dft_size) + 6 * (start >> log2(stride));

  C tw0 = {V::vec(tw[0]), V::vec(tw[1])}; 
  C tw1 = {V::vec(tw[2]), V::vec(tw[3])}; 
  C tw2 = {V::vec(tw[4]), V::vec(tw[5])}; 

  for(Int j = start; j < start + stride / 4; j++)
    two_passes_inner<typename Rows::V, typename Rows::Cf>(
      data.row(j + 0 * stride / 4),
      data.row(j + 1 * stride / 4),
      data.row(j + 2 * stride / 4),
      data.row(j + 3 * stride / 4),
      tw0, tw1, tw2, data.m, data.im_off);
}

template<typename Rows>
void last_pass(
  Int n, Int start, Int end,
  typename Rows::V::T* twiddle, const Rows& rows)
{
  typedef typename Rows::V V; VEC_TYPEDEFS(V);
  typedef typename Rows::Cf Cf;
  ASSERT(end - start == 2);

  C tw = {V::vec(twiddle[start]), V::vec(twiddle[start + 1])}; 
  auto p0 = rows.row(start);
  auto p1 = rows.row(start + 1);
  for(Int i = 0; i < rows.m * Rows::Cf::idx_ratio; i += stride<V, Cf>())
  {
    C b0 = load<V, Cf>(p0 + i, rows.im_off);
    C mul = load<V, Cf>(p1 + i, rows.im_off) * tw;
    Rows::Cf::store(b0 + mul, p0 + i, rows.im_off);
    Rows::Cf::store(b0 - mul, p1 + i, rows.im_off);
  }
}

template<typename T>
struct Fft
{
  Int n;
  Int m;
  T* working;
  T* twiddle;
  void (*fun_ptr)(
    const Fft<T>* state,
    T* src,
    T* dst,
    Int im_off,
    bool interleaved_src_rows,
    bool interleaved_dst_rows);
};

template<typename V_, typename Cf_>
struct Rows
{
  typedef Cf_ Cf;
  typedef V_ V;
  typename V::T* ptr_;
  Int m;
  Int row_stride;
  Int im_off;
  typename V::T* row(Int i) const
  {
    return ptr_ + i * row_stride * Cf::idx_ratio;
  }
};

template<typename V_, typename Cf_>
struct BrRows
{
  typedef V_ V;
  typedef Cf_ Cf;
  Int log2n;
  typename V::T* ptr_;
  Int m;
  Int row_stride;
  Int im_off;

  BrRows(Int n, typename V::T* ptr_, Int m, Int row_stride, Int im_off)
  : log2n(log2(n)), ptr_(ptr_), m(m), row_stride(row_stride), im_off(im_off) {}

  typename V::T* row(Int i) const
  {
    return ptr_ + reverse_bits(i, log2n) * row_stride * Cf::idx_ratio;
  }
};

template<typename Rows>
void fft_recurse(
  Int n,
  Int start,
  Int end,
  Int dft_size,
  typename Rows::V::T* twiddle,
  const Rows& rows)
{
  if(4 * dft_size <= n)
  {
    two_passes(n, dft_size, start, end, twiddle, rows);

    Int l = (end - start) / 4;
    if(4 * dft_size < n)
      for(Int i = start; i < end; i += l)
        fft_recurse(n, i, i + l, dft_size * 4, twiddle, rows);
  }
  else
    last_pass(n, start, end, twiddle, rows);
}

// The result is bit reversed
template<typename SrcRows, typename DstRows>
void fft_impl(
  Int n,
  typename SrcRows::V::T* twiddle,
  const SrcRows& src,
  const DstRows& dst)
{
  if(n == 1)
    complex_copy<typename SrcRows::V, typename SrcRows::Cf, typename DstRows::Cf>(
      src.row(0), src.im_off, src.m, dst.row(0), dst.im_off);
  else
  {
    if(n == 2)
      first_pass(n, src, dst);
    else
      first_two_passes(n, src, dst);

    if(n > 4) 
      for(Int i = 0; i < n; i += n / 4)
        fft_recurse(n, i, i + n / 4, 4, twiddle, dst);
  }
}

template<typename V, typename SrcCf, typename DstCf, bool br_dst_rows>
void fft(
  const Fft<typename V::T>* s,
  typename V::T* src,
  typename V::T* dst,
  Int im_off,
  bool interleaved_src_rows,
  bool interleaved_dst_rows)
{
  auto src_rows = interleaved_src_rows
    ? Rows<V, SrcCf>({src, s->m, 2 * s->m, s->m})
    : Rows<V, SrcCf>({src, s->m, s->m, im_off});

  Int dst_off = interleaved_dst_rows ? s->m : im_off;
  Int dst_stride = interleaved_dst_rows ? 2 * s->m : s->m;

  if(br_dst_rows)
    fft_impl(
      s->n, s->twiddle, src_rows,
      Rows<V, DstCf>({dst, s->m, dst_stride, dst_off}));
  else
    fft_impl(
      s->n, s->twiddle, src_rows, 
      BrRows<V, DstCf>({s->n, dst, s->m, dst_stride, dst_off}));
}


template<typename V>
Int fft_memsize(Int n)
{
  VEC_TYPEDEFS(V);
  Int r = 0;
  r = aligned_increment(r, sizeof(Fft<T>));
  r = aligned_increment(r, 2 * n * sizeof(T));
  r = aligned_increment(r, 2 * n * sizeof(T));
  return r;
}

template<typename V, typename SrcCf, typename DstCf, bool br_dst_rows>
Fft<typename V::T>* fft_create(Int n, Int m, void* ptr)
{
  VEC_TYPEDEFS(V);
  auto r = (Fft<T>*) ptr;
  ptr = aligned_increment(ptr, sizeof(Fft<T>));
  r->n = n;
  r->m = m;
  r->working = (T*) ptr;
  ptr = aligned_increment(ptr, 2 * n * sizeof(T));
  r->twiddle = (T*) ptr;
  ptr = aligned_increment(ptr, 2 * n * sizeof(T));
  
  init_twiddle<Scalar<T>>(
    [n](Int s, Int dft_size){ return 4 * dft_size <= n ? 2 : 1; },
    n, r->working, r->twiddle, nullptr);

  r->fun_ptr = &fft<V, SrcCf, DstCf, br_dst_rows>;
  return r;
}
}

const Int maxdim = 64;

template<typename T>
struct Fft
{
  Int ndim;
  Int num_elements;
  Int working_idx_ratio;
  Int dst_idx_ratio;
  T* working;
  onedim::Fft<T>* last_transform;
  multi::Fft<T>* transforms[maxdim];
};

Int product(const Int* b, const Int* e)
{
  Int r = 1;
  for(; b < e; b++) r *= *b;
  return r;
}

Int product(const Int* p, Int n) { return product(p, p + n); }

template<typename V>
Int fft_memsize(Int ndim, const Int* dim)
{
  VEC_TYPEDEFS(V);
  if(V::vec_size > 1 && dim[ndim - 1] < 2 * V::vec_size)
    return fft_memsize<Scalar<T>>(ndim, dim);

  Int r = align_size(sizeof(Fft<T>));

  Int working_size = 2 * sizeof(T) * product(dim, ndim);
  r = align_size(r + working_size);

  for(Int i = 0; i < ndim - 1; i++)
    r = align_size(r + multi::fft_memsize<V>(dim[i]));

  r = align_size(r + onedim::fft_memsize<V>(dim[ndim - 1]));

  return r;
}

template<typename V, typename SrcCf, typename DstCf>
Fft<typename V::T>* fft_create(Int ndim, const Int* dim, void* mem)
{
  VEC_TYPEDEFS(V);
  if(V::vec_size > 1 && dim[ndim - 1] < 2 * V::vec_size)
    return fft_create<Scalar<T>, SrcCf, DstCf>(ndim, dim, mem);

  auto s = (Fft<typename V::T>*) mem;
  s->ndim = ndim;
  s->working_idx_ratio = cf::Vec::idx_ratio;
  s->dst_idx_ratio = DstCf::idx_ratio;
  mem = (void*) align_size(Uint(mem) + sizeof(Fft<T>));
  s->working = (T*) mem;
 
  s->num_elements = product(dim, ndim);
  Int working_size = 2 * sizeof(T) * s->num_elements;
  mem = (void*) align_size(Uint(mem) + working_size);

  if(ndim == 1)
    s->last_transform = onedim::fft_create<V, SrcCf, DstCf>(dim[ndim - 1], mem);
  else
  {
    for(Int i = 0; i < ndim - 1; i++)
    {
      Int m = product(dim + i + 1, dim + ndim);

      if(i == 0)
        s->transforms[i] = multi::fft_create<V, SrcCf, cf::Vec, true>(
          dim[i], m, mem);
      else
        s->transforms[i] = multi::fft_create<V, cf::Vec, cf::Vec, true>(
          dim[i], m, mem);

      mem = (void*) align_size(Uint(mem) + multi::fft_memsize<V>(dim[i]));
    }

    s->last_transform = onedim::fft_create<V, cf::Vec, DstCf>(dim[ndim - 1], mem);
  }

  return s;
}

template<typename T>
void fft_impl(
  Int idim, Fft<T>* s, Int im_off, T* src, T* working, T* dst,
  bool interleaved_src_rows)
{
  ASSERT(idim < s->ndim);
  if(idim == s->ndim - 1) fft_impl(s->last_transform, im_off, src, dst);
  else
  {
    s->transforms[idim]->fun_ptr(
      s->transforms[idim], src, working, im_off, interleaved_src_rows, false);

    Int m = s->transforms[idim]->m;
    Int n = s->transforms[idim]->n;
    for(BitReversed br(n); br.i < n; br.advance())
    {
      auto next_src = working + br.i * m * s->working_idx_ratio;
      auto next_dst = dst + br.br * m * s->dst_idx_ratio;
      fft_impl(idim + 1, s, im_off, next_src, next_src, next_dst,
        interleaved_src_rows);
    }
  }
}

template<typename T>
void fft(Fft<T>* state, T* src, T* dst)
{
  fft_impl<T>(0, state, state->num_elements, src, state->working, dst, false);
}

template<typename T>
struct Ifft
{
  Fft<T> state;
};

template<typename T>
void ifft(Ifft<T>* state, T* src, T* dst)
{
  auto s = &state->state;
  fft_impl<T>(0, s, s->num_elements, src, s->working, dst, false);
}

template<typename V>
Int ifft_memsize(Int ndim, const Int* dim)
{
  return fft_memsize<V>(ndim, dim); 
}

template<typename V, typename SrcCf, typename DstCf>
Ifft<typename V::T>* ifft_create(Int ndim, const Int* dim, void* mem)
{
  VEC_TYPEDEFS(V);
  return (Ifft<T>*) fft_create<V, cf::Swapped<SrcCf>, cf::Swapped<DstCf>>(
    ndim, dim, mem);
}

// src and dst must not be the same
// does not work for inverse yet
template<typename V, typename DstCf, bool inverse>
void multi_real_pass(
  Int n, Int m, typename V::T* twiddle, typename V::T* dst, Int dst_im_off)
{
  VEC_TYPEDEFS(V);

  Vec half = V::vec(0.5);
  Int nbits = log2(n / 2);

  for(Int i = 1; i <= n / 4; i++)
  {
    C w = { V::vec(twiddle[i]), V::vec(twiddle[i + n / 2]) };

    auto d0 = dst + i * m * DstCf::idx_ratio; 
    auto d1 = dst + (n / 2 - i) * m * DstCf::idx_ratio; 

    for(auto end = d0 + m * DstCf::idx_ratio; d0 < end;)
    {
      C sval0 = load<V, DstCf>(d0, dst_im_off);
      C sval1 = load<V, DstCf>(d1, dst_im_off);

      C a, b;

      if(inverse)
      {
        a = sval0 + sval1.adj();
        b = (sval1.adj() - sval0) * w.adj();
      }
      else
      {
        a = (sval0 + sval1.adj()) * half;
        b = ((sval0 - sval1.adj()) * w) * half;
      }

      C dval0 = a + b.mul_neg_i();
      C dval1 = a.adj() + b.adj().mul_neg_i();

      DstCf::store(dval0, d0, dst_im_off);
      DstCf::store(dval1, d1, dst_im_off);

      d0 += stride<V, DstCf>();
      d1 += stride<V, DstCf>();
    }
  }

  if(inverse)
  {
    auto s0 = dst; 
    auto s1 = dst + n / 2 * m * DstCf::idx_ratio; 
    auto d = dst; 

    for(auto end = s0 + m * DstCf::idx_ratio; s0 < end;)
    {
      Vec r0 = load<V, DstCf>(s0, dst_im_off).re;
      Vec r1 = load<V, DstCf>(s1, dst_im_off).re;
      DstCf::template store<V>({r0 + r1, r0 - r1}, d, dst_im_off);

      s0 += stride<V, DstCf>();
      s1 += stride<V, DstCf>();
      d += stride<V, DstCf>();
    }
  }
  else
  {
    auto d0 = dst; 
    auto d1 = dst + n / 2 * m * DstCf::idx_ratio; 
    auto s = dst; 

    for(auto end = s + m * DstCf::idx_ratio; s < end;)
    {
      C r0 = load<V, DstCf>(s, dst_im_off);
      DstCf::template store<V>({r0.re + r0.im, V::vec(0)}, d0, dst_im_off);
      DstCf::template store<V>({r0.re - r0.im, V::vec(0)}, d1, dst_im_off);
      
      s += stride<V, DstCf>();
      d0 += stride<V, DstCf>();
      d1 += stride<V, DstCf>();
    }
  }
}

template<typename T>
struct Rfft
{
  T* twiddle;
  T* working0;
  Int outer_n;
  Int inner_n;
  Int im_off;
  Int dst_idx_ratio;
  onedim::Rfft<T>* onedim_transform;
  Fft<T>* multidim_transform;
  multi::Fft<T>* first_transform;
  void (*real_pass)(Int n, Int m, T* twiddle, T* dst, Int dst_im_off);
};

template<typename T>
Int rfft_im_off(Int ndim, const Int* dim)
{
  return align_size<T>(product(dim + 1, dim + ndim) * (dim[0] / 2 + 1));
}

template<typename V>
Int rfft_memsize(Int ndim, const Int* dim)
{
  VEC_TYPEDEFS(V)
  if(V::vec_size != 1 && dim[ndim - 1] < 2 * V::vec_size)
    return rfft_memsize<Scalar<T>>(ndim, dim);

  Int r = 0;
  r = align_size(r + sizeof(Rfft<T>));
  if(ndim == 1)
    r = align_size(r + onedim::rfft_memsize<V>(dim[0]));
  else
  {
    r = align_size(r + sizeof(T) * dim[0]);
    r = align_size(r + sizeof(T) * 2 * rfft_im_off<T>(ndim, dim));
    r = align_size(r + multi::fft_memsize<V>(dim[0] / 2));
    r = align_size(r + fft_memsize<V>(ndim - 1, dim + 1));
  }
  return r;
}

template<typename V, typename DstCf>
Rfft<typename V::T>* rfft_create(Int ndim, const Int* dim, void* mem)
{
  VEC_TYPEDEFS(V)
  if(V::vec_size != 1 && dim[ndim - 1] < 2 * V::vec_size)
    return rfft_create<Scalar<T>, DstCf>(ndim, dim, mem);

  auto r = (Rfft<T>*) mem;
  mem = (void*) align_size(Uint(mem) + sizeof(Rfft<T>)); 
  if(ndim == 1)
  {
     r->onedim_transform = onedim::rfft_create<V, DstCf>(dim[0], mem);
     r->dst_idx_ratio = 0;
     r->working0 = nullptr;
     r->twiddle = nullptr;
     r->outer_n = 0;
     r->inner_n = 0;
     r->im_off = 0;
     r->multidim_transform = nullptr;
     r->first_transform = nullptr;
     r->real_pass = nullptr;
  }
  else
  {
    r->dst_idx_ratio = DstCf::idx_ratio;
    r->outer_n = dim[0];
    r->inner_n = product(dim + 1, dim + ndim);
    r->onedim_transform = nullptr;
    r->im_off = rfft_im_off<T>(ndim, dim);
    r->twiddle = (T*) mem;
    mem = (void*) align_size(Uint(mem) + sizeof(T) * dim[0]);
    r->working0 = (T*) mem;
    mem = (void*) align_size(Uint(mem) + 2 * sizeof(T) * r->im_off);
    r->first_transform = multi::fft_create<V, cf::Split, cf::Vec, false>(
      r->outer_n / 2, r->inner_n, mem);

    mem = (void*) align_size(Uint(mem) + multi::fft_memsize<V>(dim[0] / 2));
    r->multidim_transform = fft_create<V, cf::Vec, DstCf>(ndim - 1, dim + 1, mem);
    r->real_pass = &multi_real_pass<V, cf::Vec, false>;
  
    Int m =  r->outer_n / 2;
    compute_twiddle(m, m, r->twiddle, r->twiddle + m);
  }
  
  return r;
}

template<typename T>
void rfft(Rfft<T>* s, T* src, T* dst)
{
  if(s->onedim_transform) return onedim::rfft(s->onedim_transform, src, dst);

  s->first_transform->fun_ptr(
    s->first_transform,
    src,
    s->working0, s->outer_n / 2 * s->inner_n,
    true,
    false);

  s->real_pass(s->outer_n, s->inner_n, s->twiddle, s->working0, 0);

  const Int working_idx_ratio = 2; // because we have cf::Vec in working
  const Int nbits = log2(s->outer_n / 2);
  for(Int i = 0; i < s->outer_n / 2 + 1 ; i++)
    fft_impl(
      0,
      s->multidim_transform,
      s->im_off,
      s->working0 + i * s->inner_n * working_idx_ratio,
      s->multidim_transform->working,
      dst + i * s->inner_n * s->dst_idx_ratio,
      false);
}

template<typename T>
struct Irfft
{
  T* twiddle;
  T* working0;
  T* working1;
  Int outer_n;
  Int inner_n;
  Int im_off;
  Int src_idx_ratio;
  onedim::Rfft<T>* onedim_transform;
  Fft<T>* multidim_transform;
  multi::Fft<T>* last_transform;
  void (*real_pass)(Int n, Int m, T* twiddle, T* dst, Int dst_im_off);
};

template<typename V>
Int irfft_memsize(Int ndim, const Int* dim)
{
  return rfft_memsize<V>(ndim, dim);
}

template<typename V, typename SrcCf>
Irfft<typename V::T>* irfft_create(Int ndim, const Int* dim, void* mem)
{
  VEC_TYPEDEFS(V)
  auto r = (Irfft<T>*) mem;
  mem = (void*) align_size(Uint(mem) + sizeof(Irfft<T>));
  
  if(ndim == 1)
  {
     r->onedim_transform = onedim::rfft_create<V, SrcCf>(dim[0], mem);
     r->src_idx_ratio = 0;
     r->working0 = nullptr;
     r->twiddle = nullptr;
     r->outer_n = 0;
     r->inner_n = 0;
     r->im_off = 0;
     r->multidim_transform = nullptr;
     r->last_transform = nullptr;
     r->real_pass = nullptr;
  }
  else
  {
    r->src_idx_ratio = SrcCf::idx_ratio;
    r->outer_n = dim[0];
    r->inner_n = product(dim + 1, dim + ndim);
    r->onedim_transform = nullptr;
    r->im_off = rfft_im_off<T>(ndim, dim);
    r->twiddle = (T*) mem;
    mem = (void*) align_size(Uint(mem) + sizeof(T) * dim[0]);
    r->working0 = (T*) mem;
    mem = (void*) align_size(Uint(mem) + 2 * sizeof(T) * r->im_off);

    r->last_transform = multi::fft_create<
      V, cf::Swapped<cf::Vec>, cf::Swapped<cf::Split>, false>(
        r->outer_n / 2, r->inner_n, mem);

    mem = (void*) align_size(Uint(mem) + multi::fft_memsize<V>(dim[0] / 2));
    r->multidim_transform = fft_create<
      V, cf::Swapped<SrcCf>, cf::Swapped<cf::Vec>>(
        ndim - 1, dim + 1, mem);

    r->real_pass = &multi_real_pass<V, cf::Vec, true>;
  
    Int m =  r->outer_n / 2;
    compute_twiddle(m, m, r->twiddle, r->twiddle + m);
  }
  
  return r; 
}

template<typename T>
void irfft(Irfft<T>* s, T* src, T* dst)
{
  if(s->onedim_transform) return onedim::rfft(s->onedim_transform, src, dst);
  
  const Int working_idx_ratio = 2; // because we have cf::Vec in working
  const Int nbits = log2(s->outer_n / 2);
  for(Int i = 0; i < s->outer_n / 2 + 1 ; i++)
    fft_impl(
      0,
      s->multidim_transform,
      s->im_off,
      src + i * s->inner_n * s->src_idx_ratio,
      s->multidim_transform->working,
      s->working0 + i * s->inner_n * working_idx_ratio,
      false);

  s->real_pass(s->outer_n, s->inner_n, s->twiddle, s->working0, 0);

  s->last_transform->fun_ptr(
    s->last_transform, s->working0, dst, s->outer_n / 2 * s->inner_n,
    false, true);
}
