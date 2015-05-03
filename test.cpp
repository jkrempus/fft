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

Int large_fft_size = 1 << 14;
Int max_vec_size = 8;

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
};

enum class ComplexFormat { split, scalar, vec };

template<ComplexFormat cf, typename V>
FORCEINLINE Complex<typename V::Vec> load_complex(typename V::Vec* ptr, Int n)
{
  if(cf == ComplexFormat::split)
    return {ptr[0], ptr[n]};
  else if(cf == ComplexFormat::vec)
    return {ptr[0], ptr[1]};
  else if (cf == ComplexFormat::scalar)
  {
    Complex<typename V::Vec> r;
    V::deinterleave(ptr[0], ptr[1], r.re, r.im);
    return r;
  }
}

template<ComplexFormat cf, typename V>
void store_complex(Complex<typename V::Vec> a, typename V::Vec* ptr, Int n)
{
  if(cf == ComplexFormat::split)
  {
    ptr[0] = a.re;
    ptr[n] = a.im;
  }
  else if(cf == ComplexFormat::vec)
  {
    ptr[0] = a.re;
    ptr[1] = a.im;
  }
  else if (cf == ComplexFormat::scalar)
  {
    V::interleave(a.re, a.im, ptr[0], ptr[1]);
  }
}

template<ComplexFormat cf>
FORCEINLINE Int complex_element_size()
{
  if(cf == ComplexFormat::split)
    return 1;
  else if(cf == ComplexFormat::vec)
    return 2;
  else if (cf == ComplexFormat::scalar)
    return 2;
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
  short npasses;
  bool out_of_place;
  void (*fun_ptr)(const Arg<T>&);
};

template<typename T>
struct State
{
  Int n;
  T* twiddle;
  T* tiny_twiddle;
  T* working;
  T* copied_working0;
  T* copied_working1;
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
};

#ifdef __AVX__

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
  }

  static FORCEINLINE void deinterleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
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

  static void FORCEINLINE transpose_impl(
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

  static void FORCEINLINE transpose(
    Vec a0, Vec a1, Vec a2, Vec a3,
    Vec a4, Vec a5, Vec a6, Vec a7,
    Vec& r0, Vec& r1, Vec& r2, Vec& r3,
    Vec& r4, Vec& r5, Vec& r6, Vec& r7)
  {
    transpose_impl(
      a0, a1, a4, a5, a2, a3, a6, a7,
      r0, r1, r4, r5, r2, r3, r6, r7);
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

template<typename V, ComplexFormat cf>
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
      if(cf == ComplexFormat::scalar)
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
    Complex<Vec> c = load_complex<cf, V>(a, 1);
    //store it as ComplexFormat::vec
    p[0] = c.re;
    p[1] = c.im; 
  }
}

template<typename V, ComplexFormat cf>
void init_twiddle(State<typename V::T>& state)
{
  VEC_TYPEDEFS(V);

  auto dst = state.twiddle;
  auto n = state.n;
  auto end_re = dst + n;
  auto end_im = dst + 2 * n;
  Int table_index = 0;
  end_re[-1] = T(0);
  end_im[-1] = T(0);
  end_re[-2] = T(1);
  end_im[-2] = T(0);

  for(Int size = 2; size < n; size *= 2)
  {
    table_index++;
    auto c = SinCosTable<T>::cos[table_index];
    auto s = SinCosTable<T>::sin[table_index];

    auto prev_re = end_re - size;
    auto prev_im = end_im - size;
    auto current_re = end_re - 2 * size;
    auto current_im = end_im - 2 * size;
    for(Int j = 0; j < size / 2; j++)
    {
      T re = prev_re[j];
      T im = prev_im[j];
      current_re[2 * j] = re;
      current_im[2 * j] = im;
      current_re[2 * j + 1] = re * c + im * s;
      current_im[2 * j + 1] = im * c - re * s;
    }

    if(size < V::vec_size)
    {
      auto re = state.tiny_twiddle + 2 * V::vec_size * tiny_log2(size);
      auto im = re + V::vec_size;

      for(Int j = 0; j < V::vec_size; j++)
      {
        re[j] = current_re[j & (size - 1)];
        im[j] = current_im[j & (size - 1)];
      }
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

  rearrange_vector_elements_like_load<V, cf>((Vec*) dst, 2 * n / V::vec_size);
  rearrange_vector_elements_like_load<V, cf>(
    (Vec*) state.tiny_twiddle, 2 * tiny_log2(V::vec_size));
}

extern "C" int sprintf(char* s, const char* fmt, ...);
template<typename T_> void dump(T_* ptr, Int n, const char* name, ...);

template<typename T> T min(T a, T b){ return a < b ? a : b; }
template<typename T> T max(T a, T b){ return a > b ? a : b; }

template<typename V, Int dft_size, ComplexFormat src_cf>
void ct_dft_size_pass(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  typedef Complex<Vec> C;
  const Int src_elem_size = complex_element_size<src_cf>();
  Int vn = arg.n / V::vec_size;
  auto vsrc0 = (Vec*) arg.src;
  auto vsrc1 = (Vec*) arg.src + src_elem_size * vn / 2;
  auto vdst = (Complex<Vec>*) arg.dst;
  C tw = ((C*) arg.tiny_twiddle)[tiny_log2(dft_size)];
  for(auto end = vdst + vn; vdst < end;)
  {
    C a0 = load_complex<src_cf, V>(vsrc0, vn);
    C a1 = load_complex<src_cf, V>(vsrc1, vn);

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

    vsrc0 += src_elem_size;
    vsrc1 += src_elem_size;
    vdst += 2;
  }
}

template<typename V, ComplexFormat src_cf>
void first_two_passes_impl(Int n, const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  typedef Complex<Vec> C;

  const Int src_elem_size = complex_element_size<src_cf>();

  Int vn = n / V::vec_size;
  Vec* vsrc0 = (Vec*) arg.src;
  Vec* vsrc1 = (Vec*) arg.src + src_elem_size * vn / 4;
  Vec* vsrc2 = (Vec*) arg.src + src_elem_size * 2 * vn / 4;
  Vec* vsrc3 = (Vec*) arg.src + src_elem_size * 3 * vn / 4;
  auto vdst = (C*) arg.dst;

  for(Int i = 0; i < vn / 4; i++)
  {
    C a0 = load_complex<src_cf, V>(vsrc0 + src_elem_size * i, vn);
    C a1 = load_complex<src_cf, V>(vsrc1 + src_elem_size * i, vn);
    C a2 = load_complex<src_cf, V>(vsrc2 + src_elem_size * i, vn);
    C a3 = load_complex<src_cf, V>(vsrc3 + src_elem_size * i, vn);

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

template<typename V, ComplexFormat src_cf>
void first_two_passes(const Arg<typename V::T>& arg)
{
  first_two_passes_impl<V, src_cf>(arg.n, arg);
}

template<typename V, ComplexFormat src_cf, Int n>
void first_two_passes_ct_size(const Arg<typename V::T>& arg)
{
  first_two_passes_impl<V, src_cf>(n, arg);
}

template<typename V, ComplexFormat src_cf>
FORCEINLINE void first_three_passes_impl(
  Int n,
  typename V::Vec* src,
  Complex<typename V::Vec>* dst)
{
  VEC_TYPEDEFS(V);
  Int l = n / 8;
  Vec invsqrt2 = V::vec(SinCosTable<T>::cos[2]);
  const Int src_elem_size = complex_element_size<src_cf>();

  for(auto end = src + l * src_elem_size;;)
  {
    C c0, c1, c2, c3;
    {
      C a0 = load_complex<src_cf, V>(src + 0 * l * src_elem_size, n);
      C a1 = load_complex<src_cf, V>(src + 2 * l * src_elem_size, n);
      C a2 = load_complex<src_cf, V>(src + 4 * l * src_elem_size, n);
      C a3 = load_complex<src_cf, V>(src + 6 * l * src_elem_size, n);
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
      C a0 = load_complex<src_cf, V>(src + 1 * l * src_elem_size, n);
      C a1 = load_complex<src_cf, V>(src + 3 * l * src_elem_size, n);
      C a2 = load_complex<src_cf, V>(src + 5 * l * src_elem_size, n);
      C a3 = load_complex<src_cf, V>(src + 7 * l * src_elem_size, n);
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

    src += src_elem_size;

    V::transpose(
      c0.re + mul0.re, c1.re + mul1.re, c2.re + mul2.re, c3.re + mul3.re,
      c0.re - mul0.re, c1.re - mul1.re, c2.re - mul2.re, c3.re - mul3.re,
      dst[0].re, dst[1].re, dst[2].re, dst[3].re,
      dst[4].re, dst[5].re, dst[6].re, dst[7].re);

    V::transpose(
      c0.im + mul0.im, c1.im + mul1.im, c2.im + mul2.im, c3.im + mul3.im,
      c0.im - mul0.im, c1.im - mul1.im, c2.im - mul2.im, c3.im - mul3.im,
      dst[0].im, dst[1].im, dst[2].im, dst[3].im,
      dst[4].im, dst[5].im, dst[6].im, dst[7].im);

    dst += 8;
    if(src == end) break;
  }
}

template<typename V, ComplexFormat src_cf>
void first_three_passes(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  first_three_passes_impl<V, src_cf>(
    arg.n / V::vec_size,
    (Vec*) arg.src,
    (Complex<Vec>*) arg.dst);
}

template<typename V, ComplexFormat src_cf, Int n>
void first_three_passes_ct_size(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  first_three_passes_impl<V, src_cf>(
    n / V::vec_size,
    (Vec*) arg.src,
    (Complex<Vec>*) arg.dst);
}

template<typename V, ComplexFormat dst_cf>
FORCEINLINE void last_pass_impl(
  Int dft_size,
  Complex<typename V::Vec>* src,
  Complex<typename V::Vec>* twiddle,
  typename V::Vec* dst)
{
  const Int dst_elem_size = complex_element_size<dst_cf>();
  for(Int i0 = 0, i1 = dft_size; i0 < dft_size; i0++, i1++)
  {
    auto a = src[i0];
    auto mul = src[i1] * twiddle[i0];
    store_complex<dst_cf, V>(a + mul, dst + i0 * dst_elem_size, 2 * dft_size);
    store_complex<dst_cf, V>(a - mul, dst + i1 * dst_elem_size, 2 * dft_size);
  }
}

template<typename V, ComplexFormat dst_cf>
void last_pass_vec(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  last_pass_impl<V, dst_cf>(
    arg.n / V::vec_size / 2,
    (Complex<Vec>*) arg.src,
    (Complex<Vec>*) arg.twiddle,
    (Vec*) arg.dst);
}

template<typename V, ComplexFormat dst_cf, Int n>
void last_pass_vec_ct_size(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  last_pass_impl<V, dst_cf>(
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

template<typename V, ComplexFormat dst_cf>
void two_passes(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  Int n = arg.n / V::vec_size;
  Int dft_size = arg.dft_size / V::vec_size;
  auto src = (Complex<Vec>*) arg.src;
  auto twiddle = (Complex<Vec>*) arg.twiddle;
  auto dst = (Vec*) arg.dst;
  
  typedef Complex<Vec> C;

  auto twiddle_row = twiddle + n - 4 * dft_size;
  Int l = n / 4;
  Int m = dft_size * complex_element_size<dst_cf>();

  for(C* end = src + l; src < end;)
  {
    auto tw = twiddle_row;
    for(C* end1 = src + dft_size;;)
    {
      C d0, d1, d2, d3;
      two_passes_inner(
        src[0], src[l], src[2 * l], src[3 * l], d0, d1, d2, d3,
        tw[0], tw[1], tw[2]);

      store_complex<dst_cf, V>(d0, dst, n);
      store_complex<dst_cf, V>(d1, dst + m, n);
      store_complex<dst_cf, V>(d2, dst + 2 * m, n);
      store_complex<dst_cf, V>(d3, dst + 3 * m, n);

      src += 1;
      dst += complex_element_size<dst_cf>();
      tw += 3;
      if(!(src < end1)) break;
    }

    dst += 3 * m;
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
  ComplexFormat dst_cf,
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
  
  const Int dst_elem_size = complex_element_size<dst_cf>();
  Vec* dst1 = dst0 + dstride * dst_elem_size;
  Vec* dst2 = dst1 + dstride * dst_elem_size;
  Vec* dst3 = dst2 + dstride * dst_elem_size;

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

          store_complex<dst_cf, V>(d0, dst0 + d * dst_elem_size, n);  
          store_complex<dst_cf, V>(d1, dst1 + d * dst_elem_size, n);  
          store_complex<dst_cf, V>(d2, dst2 + d * dst_elem_size, n);  
          store_complex<dst_cf, V>(d3, dst3 + d * dst_elem_size, n);  

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

        store_complex<dst_cf, V>(d0, dst0 + d * dst_elem_size, n);  
        store_complex<dst_cf, V>(d1, dst1 + d * dst_elem_size, n);  
        store_complex<dst_cf, V>(d2, dst2 + d * dst_elem_size, n);  
        store_complex<dst_cf, V>(d3, dst3 + d * dst_elem_size, n);  

        s++;
        d++;
        strided_s++;
      }
    }
}

template<typename V, ComplexFormat dst_cf>
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
    two_passes_strided_impl<0, chunk_size, true, false, ComplexFormat::vec, V>(
      n, nchunks, dft_size, offset, src, twiddle, working);
    two_passes_strided_impl<2, chunk_size, false, true, dst_cf, V>(
      n, nchunks, dft_size, offset, (C*) working, twiddle, dst);
  }
}

template<typename V, ComplexFormat dst_cf>
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

  const Int dst_elem_size = complex_element_size<dst_cf>();

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
        store_complex<dst_cf, V>(a0 + mul, dst + 0, n);
        store_complex<dst_cf, V>(a0 - mul, dst + l4 * dst_elem_size, n);
      }

      {
        auto mul = tw3.mul_neg_i() * a6;
        store_complex<dst_cf, V>(a2 + mul, dst + l2 * dst_elem_size, n);
        store_complex<dst_cf, V>(a2 - mul, dst + l6 * dst_elem_size, n);
      }
    }

    {
      C tw4 = twiddle[4];
      {
        auto mul = tw4 * a5;
        store_complex<dst_cf, V>(a1 + mul, dst + l1 * dst_elem_size, n);
        store_complex<dst_cf, V>(a1 - mul, dst + l5 * dst_elem_size, n);
      }

      {
        auto mul = tw4.mul_neg_i() * a7;
        store_complex<dst_cf, V>(a3 + mul, dst + l3 * dst_elem_size, n);
        store_complex<dst_cf, V>(a3 - mul, dst + l7 * dst_elem_size, n);
      }
    }

    src += 1;
    dst += dst_elem_size;
    twiddle += 5;
    if(src == end) break;
  }
}

template<typename V, ComplexFormat dst_cf>
void last_three_passes_vec(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  last_three_passes_impl<V, dst_cf>(
    arg.n / V::vec_size,
    (Complex<Vec>*) arg.src,
    (Complex<Vec>*) arg.twiddle,
    (Vec*) arg.dst);
}

template<typename V, ComplexFormat dst_cf, Int n>
void last_three_passes_vec_ct_size(const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  last_three_passes_impl<V, dst_cf>(
    n / V::vec_size,
    (Complex<Vec>*) arg.src,
    (Complex<Vec>*) arg.twiddle,
    (Vec*) arg.dst);
}

template<typename V, ComplexFormat cfmt>
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
          step.fun_ptr = &last_three_passes_vec_ct_size<V, cfmt, V::vec_size * 8>;
        else
          step.fun_ptr = &last_three_passes_vec<V, cfmt>;

        step.npasses = 3;
      }
#if 1
      else if(state.n >= large_fft_size && dft_size * 16 == state.n)
      {
        step.fun_ptr = &four_passes<V, cfmt>;
        step.npasses = 4;
      }
      else if(state.n >= large_fft_size && dft_size * 16 < state.n)
      {
        step.fun_ptr = &four_passes<V, ComplexFormat::vec>;
        step.npasses = 4;
      }
#endif
      else if(dft_size * 4 == state.n)
      {
        step.fun_ptr = &two_passes<V, cfmt>;
        step.npasses = 2;
      }
      else if(dft_size * 4 < state.n)
      {
        step.fun_ptr = &two_passes<V, ComplexFormat::vec>;
        step.npasses = 2;
      }
      else
      {
        if(state.n == 2 * V::vec_size)
          step.fun_ptr = &last_pass_vec_ct_size<V, cfmt, 2 * V::vec_size>;
        else if(state.n == 4 * V::vec_size)
          step.fun_ptr = &last_pass_vec_ct_size<V, cfmt, 4 * V::vec_size>;
        else if(state.n == 8 * V::vec_size)
          step.fun_ptr = &last_pass_vec_ct_size<V, cfmt, 8 * V::vec_size>;
        else
          step.fun_ptr = &last_pass_vec<V, cfmt>;

        step.npasses = 1;
      }
    }
    else if(dft_size == 1 && state.n >= 8 * V::vec_size)
    {
      if(state.n == 8 * V::vec_size)
        step.fun_ptr = &first_three_passes_ct_size<V, cfmt, 8 * V::vec_size>;
      else
        step.fun_ptr = &first_three_passes<V, cfmt>;

      step.npasses = 3;
    }
    else if(dft_size == 1 && state.n >= 4 * V::vec_size)
    {
      if(state.n == 4 * V::vec_size)
        step.fun_ptr = &first_two_passes_ct_size<V, cfmt, 4 * V::vec_size>;
      else if(state.n == 8 * V::vec_size)
        step.fun_ptr = &first_two_passes_ct_size<V, cfmt, 8 * V::vec_size>;
      else
        step.fun_ptr = &first_two_passes<V, cfmt>;

      step.npasses = 2;
    }
    else
    {
      if(V::vec_size > 1 && dft_size == 1)
        step.fun_ptr = &ct_dft_size_pass<V, 1, cfmt>;
      else if(V::vec_size > 2 && dft_size == 2)
        step.fun_ptr = &ct_dft_size_pass<V, 2, ComplexFormat::vec>;
      else if(V::vec_size > 4 && dft_size == 4)
        step.fun_ptr = &ct_dft_size_pass<V, 4, ComplexFormat::vec>;
      else if(V::vec_size > 8 && dft_size == 8)
        step.fun_ptr = &ct_dft_size_pass<V, 8, ComplexFormat::vec>;

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
  typename V::T* src,
  typename V::T* dst)
{
  VEC_TYPEDEFS(V);

  Arg<T> arg;
  arg.n = state.n;
  arg.dft_size = 1;
  arg.src = src;
  arg.twiddle = state.twiddle;
  arg.tiny_twiddle = state.tiny_twiddle;
  
  auto is_odd = bool(state.num_copies & 1);
  arg.dst = is_odd ? dst : state.working;
  auto next_dst = is_odd ? state.working : dst;

  for(Int step = 0; step < state.nsteps; step++)
  {
    auto next_dft_size = arg.dft_size << state.steps[step].npasses;
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
T* alloc_complex_array(Int n)
{
  return (T*) valloc(2 * n * sizeof(T));
}

int main(int argc, char** argv)
{
  const auto cf = ComplexFormat::scalar;
  typedef AvxFloat V;
  //typedef SseFloat V;
  //typedef Scalar<float> V;
  VEC_TYPEDEFS(V);

  Int log2n = atoi(argv[1]);
  Int n = 1 << log2n;

  T* src = alloc_complex_array<T>(n);
  T* working = alloc_complex_array<T>(n);
  T* dst = alloc_complex_array<T>(n);

  State<T> state;
  state.n = n;
  state.working = alloc_complex_array<T>(n);
  state.twiddle = alloc_complex_array<T>(n);
  state.tiny_twiddle = alloc_complex_array<T>(
    max_vec_size * tiny_log2(max_vec_size));

  init_steps<V, cf>(state);
  init_twiddle<V, cf>(state);

  std::fill_n(dst, n, T(0));
  std::fill_n(dst + n, n, T(0));

  std::mt19937 mt;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for(Int i = 0; i < n; i++)
  {
    //src.im[i] = 0;
    //T x = std::min(i, n - i) / T(n);
    //src.re[i] = T(1) / (1 + 100 * x * x);
#if 1
    src[2 * i] = dist(mt);
    src[2 * i + 1] = dist(mt);
#else
    //src[i] = dist(mt);
    //src[i + n] = dist(mt);
#endif
  }

  auto tmp_re = alloc_complex_array<T>(n / 2);
  auto tmp_im = alloc_complex_array<T>(n / 2);

#if 1
  deinterleave(src, n, tmp_re, tmp_im);
#else
  copy(src, n, tmp_re);
  copy(src + n, n, tmp_re);
#endif
  dump(tmp_re, n, "src_re.float32");
  dump(tmp_im, n, "src_im.float32");

  double t = get_time();
  for(int i = 0; i < 100LL*1000*1000*1000 / (5 * n * log2n); i++)
    fft<V>(state, src, dst);

  printf("time %f\n", get_time() - t);

#if 1
  deinterleave(dst, n, tmp_re, tmp_im);
#else
  copy(dst, n, tmp_re);
  copy(dst + n, n, tmp_re);
#endif
  dump(tmp_re, n, "dst_re.float32");
  dump(tmp_im, n, "dst_im.float32");

  return 0;
}
