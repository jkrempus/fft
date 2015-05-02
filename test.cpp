#include <immintrin.h>

typedef long Int;
typedef unsigned long Uint;

const Int max_int = Int(Uint(-1) >> 1);

#define FORCEINLINE __attribute__((always_inline)) inline
#define HOT __attribute__((hot))
#define NOINLINE __attribute__((noinline))

#define ASSERT(condition) ((condition) || *((volatile int*) 0))

#if 1
#include <cstdio>
template<typename T>
void print_vec(T a)
{
  for(Int i = 0; i < sizeof(T) / sizeof(float); i++)
    printf("%f ", ((float*)&a)[i]);

  printf("\n"); 
}
#endif

template<typename T>
FORCEINLINE void copy(const T* src, Int n, T* dst)
{
#if defined __GNUC__ || defined __clang__
  __builtin_memcpy(dst, src, n * sizeof(T));
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

template<typename T_>
struct ComplexPtrs
{
  T_* re;
  T_* im;

  ComplexPtrs& operator+=(Int offset)
  {
    re += offset;
    im += offset;
    return *this;
  }
  
  ComplexPtrs& operator-=(Int offset)
  {
    re -= offset;
    im -= offset;
    return *this;
  }

  ComplexPtrs operator+(Int offset) const { return {re + offset, im + offset}; }
  ComplexPtrs operator-(Int offset) const { return {re - offset, im - offset}; }
};

template<typename T>
struct Complex
{
  T re;
  T im;
  FORCEINLINE Complex mul_neg_i() { return {im, -re}; }
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

  static FORCEINLINE Complex load(ComplexPtrs<T> ptr)
  {
    return { ptr.re[0], ptr.im[0] };
  }

  static FORCEINLINE Complex load(ComplexPtrs<T> ptr, Int offset)
  {
    return { ptr.re[offset], ptr.im[offset] };
  }

  FORCEINLINE void store(ComplexPtrs<T> ptr)
  {
    ptr.re[0] = re;
    ptr.im[0] = im;
  }

  FORCEINLINE void store(ComplexPtrs<T> ptr, Int offset)
  {
    ptr.re[offset] = re;
    ptr.im[offset] = im;
  }
};

template<typename T>
struct Arg
{
  Int n;
  Int dft_size;
  ComplexPtrs<T> src;
  ComplexPtrs<T> twiddle;
  ComplexPtrs<T> dst;
};

template<typename T>
struct Step
{
  short npasses;
  bool out_of_place;
  void (*fun_ptr)(const Arg<T>&);
};

template<typename T>
struct State
{
  Int n;
  ComplexPtrs<T> twiddle;
  ComplexPtrs<T> working;
  ComplexPtrs<T> copied_working0;
  ComplexPtrs<T> copied_working1;
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

template<typename First, typename Second>
struct pair
{
  First first;
  Second second;
};

template<typename T_>
struct Scalar
{
  typedef T_ T;
  typedef T_ Vec;
  const static Int vec_size = 1;
  
  template<Int elements_per_vec>
  static FORCEINLINE void interleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    r0 = a0;
    r1 = a1;
  }
  
  template<Int elements>
  static Vec FORCEINLINE load_repeated(T* ptr) { return *ptr; }
  
  static void FORCEINLINE transpose(
    Vec a0, Vec a1, Vec a2, Vec a3,
    Vec& r0, Vec& r1, Vec& r2, Vec& r3)
  {
    r0 = a0;
    r1 = a2;
    r2 = a1;
    r3 = a3;
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

  static Vec FORCEINLINE vec(T a){ return a; }
};

struct SseFloat
{
  typedef float T;
  typedef __m128 Vec;
  const static Int vec_size = 4;
  
  template<Int elements_per_vec>
  static FORCEINLINE void interleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
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
  
  template<Int elements>
  static Vec FORCEINLINE load_repeated(T* ptr)
  {
    if(elements == 1) return _mm_load1_ps(ptr);
    else if(elements == 2)
    {
      Vec a = _mm_loadl_pi(_mm_setzero_ps(), (__m64*) ptr);
      return _mm_movelh_ps(a, a);
    }

    return Vec();
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
};

#ifdef __AVX__

struct AvxFloat
{
  typedef float T;
  typedef __m256 Vec;
  const static Int vec_size = 8;

  template<Int elements_per_vec>
  static FORCEINLINE void interleave(
    Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    if(elements_per_vec == 8)
      transpose_128(
        _mm256_unpacklo_ps(a0, a1), _mm256_unpackhi_ps(a0, a1), r0, r1);
    else if(elements_per_vec == 4)
      transpose_128(
        _mm256_shuffle_ps(a0, a1, _MM_SHUFFLE(1, 0, 1, 0)), 
        _mm256_shuffle_ps(a0, a1, _MM_SHUFFLE(3, 2, 3, 2)),
        r0, r1);
    else if (elements_per_vec == 2)
      transpose_128(a0, a1, r0, r1);
  }

  template<Int elements>
  static Vec FORCEINLINE load_repeated(T* ptr)
  {
    if(elements == 1) return _mm256_broadcast_ss(ptr);
    else if(elements == 2)
    {
      // can probably improve this
      Vec a = _mm256_broadcast_ss(ptr);
      Vec b = _mm256_broadcast_ss(ptr + 1);
      return _mm256_blend_ps(a, b, 0xaa);
    }
    else if(elements == 4) return _mm256_broadcast_ps((__m128*) ptr);
    
    return Vec(); // get rid of warnings
  }


  // The input matrix has 4 rows and vec_size columns
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
  ComplexPtrs<T> dst)
{
  first.store(dst);
  auto second = first * first;
  auto third = second * first;
  second.store(dst + 1);
  third.store(dst + 2);
}

template<typename V>
void init_twiddle(State<typename V::T>& state)
{
  VEC_TYPEDEFS(V);
  auto dst = state.twiddle;
  auto n = state.n;
  auto end = dst + n;
  Int table_index = 0;
  end.re[-1] = T(0);
  end.im[-1] = T(0);
  end.re[-2] = T(1);
  end.im[-2] = T(0);

  for(Int size = 2; size < n; size *= 2)
  {
    table_index++;
    auto c = SinCosTable<T>::cos[table_index];
    auto s = SinCosTable<T>::sin[table_index];

    auto prev = end - size;
    auto current = end - 2 * size;
    for(Int j = 0; j < size / 2; j++)
    {
      T prev_re = prev.re[j];
      T prev_im = prev.im[j];
      current.re[2 * j] = prev_re;
      current.im[2 * j] = prev_im;
      current.re[2 * j + 1] = prev_re * c + prev_im * s;
      current.im[2 * j + 1] = prev_im * c - prev_re * s;
    }
  }

  copy(dst.re, n, state.working.re);
  copy(dst.im, n, state.working.im);

  int dft_size = 1;
  for(Int s = 0; s < state.nsteps; s++)
  {
    auto step = state.steps[s];
    //printf("nsteps %d npasses %d\n", step.nsteps, step.npasses);

    Int vn = n / V::vec_size;
    Int vdft_size = dft_size / V::vec_size;

    if(step.npasses == 2 && dft_size >= V::vec_size)
    {
      auto src_row0 = ((ComplexPtrs<Vec>&) state.working) + vn - 4 * vdft_size;
      auto dst_row0 = ((ComplexPtrs<Vec>&) dst) + vn - 4 * vdft_size;
      for(Int i = 0; i < vdft_size; i++)
        store_two_pass_twiddle(
          Complex<Vec>::load(src_row0, i), dst_row0 + 3 * i);
    }
    else if(step.npasses == 3 && dft_size >= V::vec_size)
    {
      auto src_row0 = ((ComplexPtrs<Vec>&) state.working) + vn - 4 * vdft_size;
      auto src_row1 = ((ComplexPtrs<Vec>&) state.working) + vn - 8 * vdft_size;
      auto dst_row1 = ((ComplexPtrs<Vec>&) dst) + vn - 8 * vdft_size;
      for(Int i = 0; i < vdft_size; i++)
      {
        store_two_pass_twiddle(
          Complex<Vec>::load(src_row0, i), dst_row1 + 5 * i);

        Complex<Vec>::load(src_row1, i).store(dst_row1, 5 * i + 3);
        Complex<Vec>::load(src_row1, i + vdft_size).store(dst_row1, 5 * i + 4);
      }
    } 
    else if(step.npasses == 4 && dft_size >= V::vec_size)
    {
#if 0
      auto src_row0 = ((ComplexPtrs<Vec>&) state.working) + vn - 4 * vdft_size;
      auto src_row2 = ((ComplexPtrs<Vec>&) state.working) + vn - 16 * vdft_size;
      auto dst_row2 = ((ComplexPtrs<Vec>&) dst) + vn - 16 * vdft_size;
      for(Int i = 0; i < vdft_size; i++)
      {
        store_two_pass_twiddle(
          Complex<Vec>::load(src_row0, i), dst_row2 + 15 * i);

        for(Int j = 0; j < 4; j++)
          store_two_pass_twiddle(
            Complex<Vec>::load(src_row2 + j * vdft_size, i),
            dst_row2 + 15 * i + 3 * (j + 1));
      }
#else
      auto src_row0 = ((ComplexPtrs<Vec>&) state.working) + vn - 4 * vdft_size;
      auto dst_row0 = ((ComplexPtrs<Vec>&) dst) + vn - 4 * vdft_size;
      for(Int i = 0; i < vdft_size; i++)
        store_two_pass_twiddle(Complex<Vec>::load(src_row0, i), dst_row0 + 3 * i);
      
      auto src_row2 = ((ComplexPtrs<Vec>&) state.working) + vn - 16 * vdft_size;
      auto dst_row2 = ((ComplexPtrs<Vec>&) dst) + vn - 16 * vdft_size;
      for(Int i = 0; i < 4 * vdft_size; i++)
        store_two_pass_twiddle(Complex<Vec>::load(src_row2, i), dst_row2 + 3 * i);
#endif
    }

    dft_size <<= step.npasses;
  }

  copy(dst.re, n, state.working.re);
  copy(dst.im, n, state.working.im);

  interleave(
    (Vec*) state.working.re, (Vec*) state.working.im,
    n / V::vec_size,
    (Vec*) dst.re);
}

template<typename T>
void swap(T& a, T& b)
{
  T tmpa = a;
  T tmpb = b;
  a = tmpb;
  b = tmpa;
}

extern "C" int sprintf(char* s, const char* fmt, ...);
template<typename T_> void dump(T_* ptr, Int n, const char* name, ...);

template<typename T> T min(T a, T b){ return a < b ? a : b; }
template<typename T> T max(T a, T b){ return a > b ? a : b; }

template<typename T>
FORCEINLINE T cmul_re(T a_re, T a_im, T b_re, T b_im)
{
  return a_re * b_re - a_im * b_im;
}

template<typename T>
FORCEINLINE T cmul_im(T a_re, T a_im, T b_re, T b_im)
{
  return a_re * b_im + a_im * b_re;
}

template<typename V, Int dft_size>
void ct_dft_size_pass(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  ComplexPtrs<T> tw = arg.twiddle;
  Int vn = arg.n / V::vec_size;
  auto vsrc0_re = (Vec*) arg.src.re;
  auto vsrc0_im = (Vec*) arg.src.im;
  auto vsrc1_re = (Vec*) arg.src.re + vn / 2;
  auto vsrc1_im = (Vec*) arg.src.im + vn / 2;
  auto vdst_re = (Vec*) arg.dst.re;
  auto vdst_im = (Vec*) arg.dst.im;
  Vec tw_re = V::template load_repeated<dft_size>(tw.re);
  Vec tw_im = V::template load_repeated<dft_size>(tw.im);
  for(Int i = 0; i < vn / 2; i++)
    if(dft_size == 1)
    {
      Vec re0 = vsrc0_re[i]; 
      Vec im0 = vsrc0_im[i]; 
      Vec re1 = vsrc1_re[i]; 
      Vec im1 = vsrc1_im[i]; 
      V::template interleave<V::vec_size>(
        re0 + re1, re0 - re1, vdst_re[2 * i], vdst_re[2 * i + 1]);

      V::template interleave<V::vec_size>(
        im0 + im1, im0 - im1, vdst_im[2 * i], vdst_im[2 * i + 1]);
    }
    else
    {
      Vec re0 = vsrc0_re[i]; 
      Vec im0 = vsrc0_im[i]; 
      Vec re1 = vsrc1_re[i]; 
      Vec im1 = vsrc1_im[i];
      Vec mul_re = cmul_re(re1, im1, tw_re, tw_im);
      Vec mul_im = cmul_im(re1, im1, tw_re, tw_im);
      V::template interleave<V::vec_size / dft_size>(
        re0 + mul_re, re0 - mul_re, vdst_re[2 * i], vdst_re[2 * i + 1]);

      V::template interleave<V::vec_size / dft_size>(
        im0 + mul_im, im0 - mul_im, vdst_im[2 * i], vdst_im[2 * i + 1]);
    }
}

template<typename V>
void first_two_passes_impl(Int n, const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  Int vn = n / V::vec_size;
  Vec* vsrc0_re = (Vec*) arg.src.re;
  Vec* vsrc1_re = (Vec*) arg.src.re + vn / 4;
  Vec* vsrc2_re = (Vec*) arg.src.re + 2 * vn / 4;
  Vec* vsrc3_re = (Vec*) arg.src.re + 3 * vn / 4;
  
  Vec* vsrc0_im = (Vec*) arg.src.im;
  Vec* vsrc1_im = (Vec*) arg.src.im + vn / 4;
  Vec* vsrc2_im = (Vec*) arg.src.im + 2 * vn / 4;
  Vec* vsrc3_im = (Vec*) arg.src.im + 3 * vn / 4;

  Vec* vdst_re = (Vec*) arg.dst.re;
  Vec* vdst_im = (Vec*) arg.dst.im;

  for(Int i = 0; i < vn / 4; i++)
  {
    Complex<Vec> a0 = {vsrc0_re[i], vsrc0_im[i]};
    Complex<Vec> a1 = {vsrc1_re[i], vsrc1_im[i]};
    Complex<Vec> a2 = {vsrc2_re[i], vsrc2_im[i]};
    Complex<Vec> a3 = {vsrc3_re[i], vsrc3_im[i]};

    Complex<Vec> b0 = a0 + a2;
    Complex<Vec> b1 = a0 - a2;
    Complex<Vec> b2 = a1 + a3; 
    Complex<Vec> b3 = a1 - a3; 

    Complex<Vec> c0 = b0 + b2; 
    Complex<Vec> c2 = b0 - b2;
    Complex<Vec> c1 = b1 + b3.mul_neg_i();
    Complex<Vec> c3 = b1 - b3.mul_neg_i();

    Int j = 4 * i;
    V::transpose(
      c0.re, c1.re, c2.re, c3.re,
      vdst_re[j], vdst_re[j + 1], vdst_re[j + 2], vdst_re[j + 3]);
    
    V::transpose(
      c0.im, c1.im, c2.im, c3.im,
      vdst_im[j], vdst_im[j + 1], vdst_im[j + 2], vdst_im[j + 3]);
  }
}

template<typename V>
void first_two_passes(const Arg<typename V::T>& arg)
{
  first_two_passes_impl<V>(arg.n, arg);
}

template<typename V, Int n>
void first_two_passes_ct_size(const Arg<typename V::T>& arg)
{
  first_two_passes_impl<V>(n, arg);
}

template<typename V>
FORCEINLINE void first_three_passes_impl(Int n, const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  
  Int vn = n / V::vec_size;
  Int l = vn / 8;
  
  Vec invsqrt2 = V::vec(SinCosTable<T>::cos[2]);
  
  Vec* sre = (Vec*) arg.src.re;
  Vec* sim = (Vec*) arg.src.im;

  auto d = (Complex<Vec>*) arg.dst.re;

  for(auto end = sre + l;;)
  {
    C c0, c1, c2, c3;
    {
      C a0 = {sre[0],     sim[0]};
      C a1 = {sre[2 * l], sim[2 * l]};
      C a2 = {sre[4 * l], sim[4 * l]};
      C a3 = {sre[6 * l], sim[6 * l]};
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
      C a0 = {sre[l],     sim[l]};
      C a1 = {sre[3 * l], sim[3 * l]};
      C a2 = {sre[5 * l], sim[5 * l]};
      C a3 = {sre[7 * l], sim[7 * l]};
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

    sre++;
    sim++;

    V::transpose(
      c0.re + mul0.re, c1.re + mul1.re, c2.re + mul2.re, c3.re + mul3.re,
      c0.re - mul0.re, c1.re - mul1.re, c2.re - mul2.re, c3.re - mul3.re,
      d[0].re, d[1].re, d[2].re, d[3].re, d[4].re, d[5].re, d[6].re, d[7].re);

    V::transpose(
      c0.im + mul0.im, c1.im + mul1.im, c2.im + mul2.im, c3.im + mul3.im,
      c0.im - mul0.im, c1.im - mul1.im, c2.im - mul2.im, c3.im - mul3.im,
      d[0].im, d[1].im, d[2].im, d[3].im, d[4].im, d[5].im, d[6].im, d[7].im);

    d += 8;
    if(sre == end) break;
  }
}

template<typename V>
void first_three_passes(const Arg<typename V::T>& arg)
{
  first_three_passes_impl<V>(arg.n, arg);
}

template<typename V, Int n>
void first_three_passes_ct_size(const Arg<typename V::T>& arg)
{
  first_three_passes_impl<V>(n, arg);
}

template<typename T>
FORCEINLINE void last_pass_impl(
  Int dft_size,
  ComplexPtrs<T> src_ptrs,
  ComplexPtrs<T> twiddle_ptrs,
  ComplexPtrs<T> dst)
{
  typedef Complex<T> C;
  auto src = (C*) src_ptrs.re;
  auto twiddle = (C*) twiddle_ptrs.re;

  for(Int i0 = 0, i1 = dft_size; i0 < dft_size; i0++, i1++)
  {
    C a = src[i0];
    C mul = src[i1] * twiddle[i0];
    (a + mul).store(dst, i0);
    (a - mul).store(dst, i1);
  }
}

template<typename V>
void last_pass_vec(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  last_pass_impl(
    arg.n / V::vec_size / 2,
    (ComplexPtrs<Vec>&) arg.src,
    (ComplexPtrs<Vec>&) arg.twiddle,
    (ComplexPtrs<Vec>&) arg.dst);
}

template<typename V, Int n>
void last_pass_vec_ct_size(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  last_pass_impl(
    n / V::vec_size / 2,
    (ComplexPtrs<Vec>&) arg.src,
    (ComplexPtrs<Vec>&) arg.twiddle,
    (ComplexPtrs<Vec>&) arg.dst);
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

template<typename T>
FORCEINLINE void two_passes_impl(
  Int n, Int dft_size,
  Complex<T>* src,
  Complex<T>* twiddle,
  Complex<T>* dst)
{
  typedef Complex<T> C;

  auto twiddle_row = twiddle + n - 4 * dft_size;
  Int l = n / 4;

  Int s = 0;
  Int d = 0;

  if(dft_size == 16) printf("two_passes_impl\n");
  for(C* end = src + l; src < end;)
  {
    auto tw = twiddle_row;
    for(C* end1 = src + dft_size;;)
    {
      if(dft_size == 16)
        printf("%-4d%-4d%-4d%-4d    %-4d%-4d%-4d%-4d\n",
               s, s + l, s + 2 * l, s + 3 * l,
               d, d + dft_size, d + 2 * dft_size, d + 3 * dft_size);

      two_passes_inner(
        src[0], src[l], src[2 * l], src[3 * l],
        dst[0], dst[dft_size], dst[2 * dft_size], dst[3 * dft_size],
        tw[0], tw[1], tw[2]);

      src += 1;
      dst += 1;
      s++;
      d++;
      tw += 3;
      if(!(src < end1)) break;
    }

    d += 3 * dft_size;
    dst += 3 * dft_size;
  }
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

Int to_strided_index(
  Int i, Int n, Int nchunks, Int chunk_size, Int dft_size, Int npasses,
  Int offset)
{
  if(chunk_size < dft_size)
  {
    Int ichunk = i / chunk_size;
    Int chunk_offset = i % chunk_size;
    Int dft_size_mul = 1 << npasses;

    return
      (ichunk & ~(dft_size_mul - 1)) * n / nchunks +
      (ichunk & (dft_size_mul - 1)) * dft_size +
      chunk_offset +
      dft_size_mul * (offset & ~(dft_size - 1)) +
      (offset & (dft_size - 1));
  }
  else
  {
    Int contiguous_size = max(dft_size, chunk_size) << npasses;

    Int icontiguous = i / contiguous_size;
    Int contiguous_offset = i % contiguous_size;

    Int stride = n / (nchunks * chunk_size / contiguous_size);

    return
      icontiguous * stride +
      contiguous_offset +
      offset / chunk_size * contiguous_size;
  }
}


template<
  Int npasses,
  Int chunk_size,
  bool strided_src,
  bool strided_dst,
  typename T>
FORCEINLINE void two_passes_strided_impl(
  Int n,
  Int nchunks,
  Int initial_dft_size,
  Int offset,
  Complex<T>* src,
  Complex<T>* twiddle_start,
  Complex<T>* dst)
{
  //printf("npasses %d offset %d\n", npasses, offset);
  typedef Complex<T> C;
 
  Int l = nchunks * chunk_size / 4;
  Int dft_size = initial_dft_size << npasses;
  Int m = min(initial_dft_size, chunk_size) << npasses;

  C* twiddle = twiddle_start + n - 4 * dft_size;

  Int sstride =
    strided_src ?
      to_strided_index(
        l, n, nchunks, chunk_size, initial_dft_size, npasses, offset) -
      to_strided_index(
        0, n, nchunks, chunk_size, initial_dft_size, npasses, offset) 
    : l;

  Int dstride =
    strided_dst ?
      to_strided_index(
        m, n, nchunks, chunk_size, initial_dft_size, npasses + 2, offset) -
      to_strided_index(
        0, n, nchunks, chunk_size, initial_dft_size, npasses + 2, offset) 
    : m;

  printf(
    "dft_size %d n %d npasses %d offset %d l %d m %d sstride %d dstride %d\n",
    dft_size, n, npasses, offset, l, m, sstride, dstride);

  for(Int i = 0; i < l; i += m)
  {
    for(Int j = 0; j < m; j++)
    {
      Int s = i + j;
      Int d = 4 * i + j;

      Int ss = to_strided_index(
        s, n, nchunks, chunk_size, initial_dft_size, npasses, offset);

      Int sd = to_strided_index(
        d, n, nchunks, chunk_size, initial_dft_size, npasses + 2, offset);

      if(strided_src) s = ss;
      if(strided_dst) d = sd; 

      auto tw = twiddle + 3 * (s & (dft_size - 1));
    
#if 0  
      printf("%-4d%-4d%-4d%-4d    %-4d%-4d%-4d%-4d\n",
             s, s + sstride, s + 2 * sstride, s + 3 * sstride,
             d, d + dstride, d + 2 * dstride, d + 3 * dstride);
#endif

      two_passes_inner(
        src[s], src[s + sstride], src[s + 2 * sstride], src[s + 3 * sstride],
        dst[d], dst[d + dstride], dst[d + 2 * dstride], dst[d + 3 * dstride],
        tw[0], tw[1], tw[2]);

      s++;
      d++;
    }
  }
}

template<typename T>
FORCEINLINE void four_passes_impl(
  Int n, Int dft_size,
  ComplexPtrs<T> src_ptrs,
  ComplexPtrs<T> twiddle_ptrs,
  ComplexPtrs<T> dst_ptrs)
{
  typedef Complex<T> C;

  auto src = (C*) src_ptrs.re;
  auto dst = (C*) dst_ptrs.re;
  auto twiddle = (C*) twiddle_ptrs.re;

#if 1
  const Int chunk_size = 1;
  //const Int nchunks = 16;
  const Int nchunks = 16;
  Int stride = n / nchunks;
  //C working[chunk_size * nchunks];
  C working[1024];

#if 0
  two_passes_impl(n, dft_size, src, twiddle, working);
#else
  for(Int offset = 0; offset < stride; offset += chunk_size)
    two_passes_strided_impl<0, chunk_size, true, true>(
      n, nchunks, dft_size, offset, src, twiddle, working);
#endif

#if 0
  two_passes_impl(n, 4 * dft_size, working, twiddle, dst);
#else
  for(Int offset = 0; offset < stride; offset += chunk_size)
    two_passes_strided_impl<2, chunk_size, true, true>(
      n, nchunks, dft_size, offset, working, twiddle, dst);
#endif

#else
  C working[1024];
  two_passes_impl(n, dft_size, src, twiddle, working);
  two_passes_impl(n, 4 * dft_size, working, twiddle, dst);
#endif
}

template<typename T>
FORCEINLINE void last_three_passes_impl(
  Int n,
  ComplexPtrs<T> src,
  ComplexPtrs<T> twiddle,
  ComplexPtrs<T> dst)
{
  typedef Complex<T> C;
  Int l1 = n / 8;
  Int l2 = 2 * l1;
  Int l3 = 3 * l1;
  Int l4 = 4 * l1;
  Int l5 = 5 * l1;
  Int l6 = 6 * l1;
  Int l7 = 7 * l1;

  auto csrc = (C*) src.re;
  auto ctwiddle = (C*) twiddle.re;

  for(auto end = csrc + l1;;)
  {
    C a[8];
    C a0, a1, a2, a3, a4, a5, a6, a7;
    {
      C tw0 = ctwiddle[0];
      C tw1 = ctwiddle[1];
      C tw2 = ctwiddle[2];

      {
        C mul0 =       csrc[0];
        C mul1 = tw0 * csrc[l2];
        C mul2 = tw1 * csrc[l4];
        C mul3 = tw2 * csrc[l6];

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
        C mul0 =       csrc[l1];
        C mul1 = tw0 * csrc[l3];
        C mul2 = tw1 * csrc[l5];
        C mul3 = tw2 * csrc[l7];

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
      C tw3 = ctwiddle[3];
      {
        auto mul = tw3 * a4;
        (a0 + mul).store(dst, 0);
        (a0 - mul).store(dst, l4);
      }

      {
        auto mul = tw3.mul_neg_i() * a6;
        (a2 + mul).store(dst, l2);
        (a2 - mul).store(dst, l6);
      }
    }

    {
      C tw4 = ctwiddle[4];
      {
        auto mul = tw4 * a5;
        (a1 + mul).store(dst, l1);
        (a1 - mul).store(dst, l5);
      }

      {
        auto mul = tw4.mul_neg_i() * a7;
        (a3 + mul).store(dst, l3);
        (a3 - mul).store(dst, l7);
      }
    }

    csrc += 1;
    dst += 1;
    ctwiddle += 5;
    if(csrc == end) break;
  }
}

template<typename V>
void two_passes_vec(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  two_passes_impl(
    arg.n / V::vec_size, arg.dft_size / V::vec_size,
    (Complex<Vec>*) arg.src.re,
    (Complex<Vec>*) arg.twiddle.re,
    (Complex<Vec>*) arg.dst.re);
}

template<typename V>
void last_three_passes_vec(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  last_three_passes_impl(
    arg.n / V::vec_size,
    (ComplexPtrs<Vec>&) arg.src,
    (ComplexPtrs<Vec>&) arg.twiddle,
    (ComplexPtrs<Vec>&) arg.dst);
}

template<typename V>
void four_passes_vec(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  four_passes_impl(
    arg.n / V::vec_size, arg.dft_size / V::vec_size,
    (ComplexPtrs<Vec>&) arg.src,
    (ComplexPtrs<Vec>&) arg.twiddle,
    (ComplexPtrs<Vec>&) arg.dst);
}

template<typename V, Int n>
void last_three_passes_vec_ct_size(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  last_three_passes_impl(
    n / V::vec_size,
    (ComplexPtrs<Vec>&) arg.src,
    (ComplexPtrs<Vec>&) arg.twiddle,
    (ComplexPtrs<Vec>&) arg.dst);
}

template<typename V>
void init_steps(State<typename V::T>& state)
{
  VEC_TYPEDEFS(V);
  Int step_index = 0;
  state.num_copies = 0;
  for(Int dft_size = 1; dft_size < state.n; step_index++)
  {
    Step<T> step;
    step.out_of_place = true;
    if(dft_size >= V::vec_size)
    {
      if(dft_size * 8 == state.n)
      {
        if(state.n == V::vec_size * 8)
          step.fun_ptr = &last_three_passes_vec_ct_size<V, V::vec_size * 8>;
        else
          step.fun_ptr = &last_three_passes_vec<V>;

        step.npasses = 3;
      }
#if 1
      else if(dft_size > V::vec_size && dft_size * 16 <= state.n)
      {
        step.fun_ptr = &four_passes_vec<V>;
        step.npasses = 4;
      }
#endif
      else if(dft_size * 4 <= state.n)
      {
        step.fun_ptr = &two_passes_vec<V>;
        step.npasses = 2;
      }
      else
      {
        if(state.n == 2 * V::vec_size)
          step.fun_ptr = &last_pass_vec_ct_size<V, 2 * V::vec_size>;
        else if(state.n == 4 * V::vec_size)
          step.fun_ptr = &last_pass_vec_ct_size<V, 4 * V::vec_size>;
        else if(state.n == 8 * V::vec_size)
          step.fun_ptr = &last_pass_vec_ct_size<V, 8 * V::vec_size>;
        else
          step.fun_ptr = &last_pass_vec<V>;

        step.npasses = 1;
      }
    }
    else if(dft_size == 1 && state.n >= 8 * V::vec_size)
    {
      if(state.n == 8 * V::vec_size)
        step.fun_ptr = &first_three_passes_ct_size<V, 8 * V::vec_size>;
      else
        step.fun_ptr = &first_three_passes<V>;

      step.npasses = 3;
    }
    else if(dft_size == 1 && state.n >= 4 * V::vec_size)
    {
      if(state.n == 4 * V::vec_size)
        step.fun_ptr = &first_two_passes_ct_size<V, 4 * V::vec_size>;
      else if(state.n == 8 * V::vec_size)
        step.fun_ptr = &first_two_passes_ct_size<V, 8 * V::vec_size>;
      else
        step.fun_ptr = &first_two_passes<V>;

      step.npasses = 2;
    }
    else
    {
      if(V::vec_size > 1 && dft_size == 1)
        step.fun_ptr = &ct_dft_size_pass<V, 1>;
      else if(V::vec_size > 2 && dft_size == 2)
        step.fun_ptr = &ct_dft_size_pass<V, 2>;
      else if(V::vec_size > 4 && dft_size == 4)
        step.fun_ptr = &ct_dft_size_pass<V, 4>;
      else if(V::vec_size > 8 && dft_size == 8)
        step.fun_ptr = &ct_dft_size_pass<V, 8>;

      step.npasses = 1;
    }

    state.steps[step_index] = step;
    dft_size <<= step.npasses;
    if(step.out_of_place)
      state.num_copies++;
  }

  state.nsteps = step_index;

#if 1
  for(Int i = 0; i < state.nsteps; i++)
    printf("npasses %d\n", state.steps[i].npasses);
#endif
}

template<typename V>
void fft(
  const State<typename V::T>& state,
  ComplexPtrs<typename V::T> src,
  ComplexPtrs<typename V::T> dst)
{
  VEC_TYPEDEFS(V);

  Arg<T> arg;
  arg.n = state.n;
  arg.dft_size = 1;
  arg.src = src;
  
  //auto twiddle_end = state.twiddle + state.n;
  
  auto is_odd = bool(state.num_copies & 1);
  arg.dst = is_odd ? dst : state.working;
  auto next_dst = is_odd ? state.working : dst;

  for(Int step = 0; step < state.nsteps; step++)
  {
    auto next_dft_size = arg.dft_size << state.steps[step].npasses;
    arg.twiddle = state.twiddle;
    state.steps[step].fun_ptr(arg);
    arg.dft_size = next_dft_size;

    if(state.steps[step].out_of_place)
    {
      swap(next_dst, arg.dst);
      arg.src = next_dst;
    }
  }

#if 0
  deinterleave(
    (Vec*) arg.src.re,
    arg.n / V::vec_size,
    (Vec*) arg.dst.re,
    (Vec*) arg.dst.im);
#endif
}

#include <string>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include <fstream>
#include <unistd.h>
#include <random>

double get_time()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

template<typename T_>
void dump(T_* ptr, Int n, const char* name, ...)
{
  char buf[1 << 10];

  va_list args;
  va_start(args, name);
  vsprintf(buf, name, args);
  va_end(args); 

  std::ofstream(buf, std::ios_base::binary).write((char*) ptr, sizeof(T_) * n);
}

template<typename T>
ComplexPtrs<T> alloc_complex_ptrs(Int n)
{
  ComplexPtrs<T> r;
  r.re = (T*) valloc(2 * n * sizeof(T));
  r.im = r.re + n;
  return r;
}

int main(int argc, char** argv)
{
  typedef AvxFloat V;
  //typedef SseFloat V;
  //typedef Scalar<float> V;
  VEC_TYPEDEFS(V);

  Int log2n = atoi(argv[1]);
  Int n = 1 << log2n;

  ComplexPtrs<T> src = alloc_complex_ptrs<T>(n);
  ComplexPtrs<T> working = alloc_complex_ptrs<T>(n);
  ComplexPtrs<T> dst = alloc_complex_ptrs<T>(n);

  State<T> state;
  state.n = n;
  state.working = alloc_complex_ptrs<T>(n);
  state.twiddle = alloc_complex_ptrs<T>(n);
  init_steps<V>(state);
  init_twiddle<V>(state);

  std::fill_n(dst.re, n, T(0));
  std::fill_n(dst.im, n, T(0));

  std::mt19937 mt;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for(Int i = 0; i < n; i++)
  {
    //src.im[i] = 0;
    //T x = std::min(i, n - i) / T(n);
    //src.re[i] = T(1) / (1 + 100 * x * x);
    src.re[i] = dist(mt);
    src.im[i] = dist(mt);
  }

  dump(src.re, n, "src_re.float32");
  dump(src.im, n, "src_im.float32");

  double t = get_time();
  //for(int i = 0; i < 100LL*1000*1000*1000 / (5 * n * log2n); i++)
    fft<V>(state, src, dst);

  printf("time %f\n", get_time() - t);

  dump(dst.re, n, "dst_re.float32");
  dump(dst.im, n, "dst_im.float32");

  return 0;
}
