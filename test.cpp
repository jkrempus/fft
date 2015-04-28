#include <immintrin.h>

typedef long Int;
typedef unsigned long Uint;

const Int max_int = Int(Uint(-1) >> 1);

#define FORCEINLINE __attribute__((always_inline)) inline
#define NOINLINE __attribute__((noinline))

#define ASSERT(condition) ((condition) || *((volatile int*) 0))

template<typename T>
FORCEINLINE void copy(const T* src, Int n, T* dst)
{
#if defined __GNUC__ || defined __clang__
  __builtin_memcpy(dst, src, n * sizeof(T));
#else
  for(Int i = 0; i < n; i++) dst[i] = src[i];
#endif
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

const Int max_passes_per_copy = 5;
const Int copy_chunk_size = 8;

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

#include <cstdio>

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
    }

    dft_size <<= step.npasses;
  }
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
template<typename T_> void dump(T_* ptr, Int n, const char* name);

template<typename T> T min(T a, T b){ a < b ? a : b; }
template<typename T> T max(T a, T b){ a > b ? a : b; }

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

  Vec* dre = (Vec*) arg.dst.re;
  Vec* dim = (Vec*) arg.dst.im;

  //for(Int i = 0; i < l; i++)
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

    C c4, c5, c6, c7;
    {
      C a0 = {sre[l],     sim[l]};
      C a1 = {sre[3 * l], sim[3 * l]};
      C a2 = {sre[5 * l], sim[5 * l]};
      C a3 = {sre[7 * l], sim[7 * l]};
      C b0 = a0 + a2;
      C b1 = a0 - a2;
      C b2 = a1 + a3; 
      C b3 = a1 - a3; 
      c4 = b0 + b2; 
      c6 = b0 - b2;
      c5 = b1 + b3.mul_neg_i();
      c7 = b1 - b3.mul_neg_i();
    }

    sre++;
    sim++;

    C mul0 = c4;
    C d0 = c0 + mul0;
    C d4 = c0 - mul0;

    C mul1 = {invsqrt2 * (c5.re + c5.im), invsqrt2 * (c5.im - c5.re)};
    C d1 = c1 + mul1;
    C d5 = c1 - mul1;

    C mul2 = c6.mul_neg_i();
    C d2 = c2 + mul2;
    C d6 = c2 - mul2;

    C mul3 = {invsqrt2 * (c7.im - c7.re), invsqrt2 * (-c7.im - c7.re)};
    C d3 = c3 + mul3;
    C d7 = c3 - mul3;

    V::transpose(
      d0.re, d1.re, d2.re, d3.re, d4.re, d5.re, d6.re, d7.re,
      dre[0], dre[1], dre[2], dre[3], dre[4], dre[5], dre[6], dre[7]);

    V::transpose(
      d0.im, d1.im, d2.im, d3.im, d4.im, d5.im, d6.im, d7.im,
      dim[0], dim[1], dim[2], dim[3], dim[4], dim[5], dim[6], dim[7]);

    dre += 8;
    dim += 8;
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
  ComplexPtrs<T> data,
  ComplexPtrs<T> twiddle)
{
  for(Int i0 = 0, i1 = dft_size; i0 < dft_size; i0++, i1++)
  {
    T tw_re = twiddle.re[i0];
    T tw_im = twiddle.im[i0];
    T re0 = data.re[i0];
    T im0 = data.im[i0];
    T re1 = data.re[i1];
    T im1 = data.im[i1];
    T mul_re = tw_re * re1 - tw_im * im1;
    T mul_im = tw_re * im1 + tw_im * re1;
    data.re[i0] = re0 + mul_re;
    data.re[i1] = re0 - mul_re;
    data.im[i0] = im0 + mul_im;
    data.im[i1] = im0 - mul_im;
  }
}

template<typename V>
void last_pass_vec(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  last_pass_impl(
    arg.n / V::vec_size / 2,
    (ComplexPtrs<Vec>&) arg.src,
    (ComplexPtrs<Vec>&) arg.twiddle);
}

template<typename V, Int n>
void last_pass_vec_ct_size(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  last_pass_impl(
    n / V::vec_size / 2,
    (ComplexPtrs<Vec>&) arg.src,
    (ComplexPtrs<Vec>&) arg.twiddle);
}

template<typename T>
FORCEINLINE void two_passes_inner(
  ComplexPtrs<T> src_ptrs,
  ComplexPtrs<T> dst_ptrs,
  Int src_stride,
  Int dst_stride,
  Complex<T> tw0,
  Complex<T> tw1,
  Complex<T> tw2)
{
  typedef Complex<T> C;
  C mul0 =       C::load(src_ptrs);
  C mul1 = tw0 * C::load(src_ptrs + src_stride);
  C mul2 = tw1 * C::load(src_ptrs + 2 * src_stride);
  C mul3 = tw2 * C::load(src_ptrs + 3 * src_stride);

  C sum02 = mul0 + mul2;
  C dif02 = mul0 - mul2;
  C sum13 = mul1 + mul3;
  C dif13 = mul1 - mul3;

  (sum02 + sum13).store(dst_ptrs); 
  (sum02 - sum13).store(dst_ptrs + 2 * dst_stride);
  (dif02 + dif13.mul_neg_i()).store(dst_ptrs + dst_stride);
  (dif02 - dif13.mul_neg_i()).store(dst_ptrs + 3 * dst_stride); 
}

template<typename T>
FORCEINLINE void two_passes_impl(
  Int n, Int dft_size,
  ComplexPtrs<T> src,
  ComplexPtrs<T> twiddle,
  ComplexPtrs<T> dst)
{
  typedef Complex<T> C;
  ComplexPtrs<T> src_ptrs = src;
  ComplexPtrs<T> dst_ptrs = dst;
  Int l = n / 4;

  for(T* end = src_ptrs.re + l; src_ptrs.re < end;)
  {
    for(T* end1 = src_ptrs.re + dft_size;;)
    {
      two_passes_inner(src_ptrs, dst_ptrs, l, dft_size,
        C::load(twiddle), C::load(twiddle + 1), C::load(twiddle + 2));

      src_ptrs += 1;
      dst_ptrs += 1;
      twiddle += 3;
      if(!(src_ptrs.re < end1)) break;
    }

    dst_ptrs += 3 * dft_size;
    twiddle -= 3 * dft_size;
  }
}

template<typename T>
FORCEINLINE void four_passes_impl(
  Int n, Int dft_size,
  ComplexPtrs<T> src,
  ComplexPtrs<T> twiddle,
  ComplexPtrs<T> dst)
{
  typedef Complex<T> C;
  Int l = n / 16;

  T mem_re[16];
  T mem_im[16];
  ComplexPtrs<T> working = {mem_re, mem_im};

  for(T* end = src.re + l; src.re < end;)
  {
    for(T* end1 = src.re + dft_size;;)
    {
#if 0
      const int a = 0;
      __builtin_prefetch(dst.re + a);
      __builtin_prefetch(dst.im + a);
      __builtin_prefetch(src.re + a);
      __builtin_prefetch(src.im + a);
      __builtin_prefetch(twiddle.re + 15 * a);
      __builtin_prefetch(twiddle.im + 15 * a);
#endif

      {
        C tw0 = C::load(twiddle + 0);
        C tw1 = C::load(twiddle + 1);
        C tw2 = C::load(twiddle + 2);
        for(Int i = 0; i < 4; i++)
          two_passes_inner(
            src + i * l, working + 4 * i, 4 * l, 1, tw0, tw1, tw2);
      }

      for(int i = 0; i < 4; i++)
      {
        C tw0 = C::load(twiddle + 3 * (i + 1) + 0);
        C tw1 = C::load(twiddle + 3 * (i + 1) + 1);
        C tw2 = C::load(twiddle + 3 * (i + 1) + 2);
        two_passes_inner(
          working + i, dst + i * dft_size, 4, dft_size * 4, tw0, tw1, tw2);
      }

      src += 1;
      dst += 1;
      twiddle += 15;
      if(!(src.re < end1)) break;
    }

    dst += 15 * dft_size;
    twiddle -= 15 * dft_size;
  }
}

template<typename T>
FORCEINLINE void last_three_passes_impl(
  Int n,
  ComplexPtrs<T> data,
  ComplexPtrs<T> twiddle)
{
  typedef Complex<T> C;
  Int l1 = n / 8;
  Int l2 = 2 * l1;
  Int l3 = 3 * l1;
  Int l4 = 4 * l1;
  Int l5 = 5 * l1;
  Int l6 = 6 * l1;
  Int l7 = 7 * l1;

  for(auto end = data.re + l1;;)
  {
    C a0, a1, a2, a3, a4, a5, a6, a7;

    {
      C tw0 = C::load(twiddle, 0);
      C tw1 = C::load(twiddle, 1);
      C tw2 = C::load(twiddle, 2);

      {
        C mul0 =       C::load(data, 0);
        C mul1 = tw0 * C::load(data, l2);
        C mul2 = tw1 * C::load(data, l4);
        C mul3 = tw2 * C::load(data, l6);

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
        C mul0 =       C::load(data, l1);
        C mul1 = tw0 * C::load(data, l3);
        C mul2 = tw1 * C::load(data, l5);
        C mul3 = tw2 * C::load(data, l7);

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
      C tw3 = C::load(twiddle, 3);
      C tw4 = C::load(twiddle, 4);
      {
        auto mul = tw3 * a4;
        (a0 + mul).store(data, 0);
        (a0 - mul).store(data, l4);
      }

      {
        auto mul = tw4 * a5;
        (a1 + mul).store(data, l1);
        (a1 - mul).store(data, l5);
      }

      {
        auto mul = tw3.mul_neg_i() * a6;
        (a2 + mul).store(data, l2);
        (a2 - mul).store(data, l6);
      }

      {
        auto mul = tw4.mul_neg_i() * a7;
        (a3 + mul).store(data, l3);
        (a3 - mul).store(data, l7);
      }
    }

    data += 1;
    twiddle += 5;
    if(data.re == end) break;
  }
}

template<typename V>
void two_passes_vec(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  two_passes_impl(
    arg.n / V::vec_size, arg.dft_size / V::vec_size,
    (ComplexPtrs<Vec>&) arg.src,
    (ComplexPtrs<Vec>&) arg.twiddle,
    (ComplexPtrs<Vec>&) arg.dst);
}

template<typename V>
void last_three_passes_vec(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  last_three_passes_impl(
    arg.n / V::vec_size,
    (ComplexPtrs<Vec>&) arg.src,
    (ComplexPtrs<Vec>&) arg.twiddle);
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
    (ComplexPtrs<Vec>&) arg.twiddle);
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
        step.out_of_place = false;
      }
      /*else if(dft_size * 16 <= state.n)
      {
        step.fun_ptr = &four_passes_vec<V>;
        step.npasses = 4;
      }*/
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
        step.out_of_place = false;
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
  
  auto twiddle_end = state.twiddle + state.n;
  
  auto is_odd = bool(state.num_copies & 1);
  arg.dst = is_odd ? dst : state.working;
  auto next_dst = is_odd ? state.working : dst;

  for(Int step = 0; step < state.nsteps; step++)
  {
    auto next_dft_size = arg.dft_size << state.steps[step].npasses;
    arg.twiddle = twiddle_end - next_dft_size;
    state.steps[step].fun_ptr(arg);
    arg.dft_size = next_dft_size;

    if(state.steps[step].out_of_place)
    {
      swap(next_dst, arg.dst);
      arg.src = next_dst;
    }
  }
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
void dump(T_* ptr, Int n, const char* name)
{
  std::ofstream(name, std::ios_base::binary).write((char*) ptr, sizeof(T_) * n);
}

template<typename T>
void print_vec(T a)
{
  for(Int i = 0; i < sizeof(T) / sizeof(float); i++)
    printf("%f ", ((float*)&a)[i]);

  printf("\n"); 
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
  for(int i = 0; i < 100LL*1000*1000*1000 / (5 * n * log2n); i++)
    fft<V>(state, src, dst);

  printf("time %f\n", get_time() - t);

  dump(dst.re, n, "dst_re.float32");
  dump(dst.im, n, "dst_im.float32");

  return 0;
}
