#include <immintrin.h>

typedef long Int;

#define FORCEINLINE __attribute__((always_inline)) inline

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
  
  static Vec FORCEINLINE transpose(
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
  static Vec FORCEINLINE transpose(
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

  static Vec FORCEINLINE transpose(
    Vec a0, Vec a1, Vec a2, Vec a3,
    Vec a4, Vec a5, Vec a6, Vec a7,
    Vec& r0, Vec& r1, Vec& r2, Vec& r3,
    Vec& r4, Vec& r5, Vec& r6, Vec& r7)
  {
    transpose4x4_two(a0, a2, a4, a6);
    transpose4x4_two(a1, a3, a5, a7);

    transpose_128(a0, a1, r0, r1);
    transpose_128(a2, a3, r2, r3);
    transpose_128(a4, a5, r4, r5);
    transpose_128(a6, a7, r6, r7);
  }

  static Vec FORCEINLINE to_vec(T a){ return _mm256_set1_ps(a); }

private: 
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
  typedef typename V::Vec Vec;

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

  ComplexPtrs operator+(Int offset) { return {re + offset, im + offset}; }
  ComplexPtrs operator-(Int offset) { return {re - offset, im - offset}; }
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

  FORCEINLINE void store(T* re_ptr, T* im_ptr, Int offset)
  {
    re_ptr[offset] = re;
    im_ptr[offset] = im;
  }
};

template<typename V>
void init_twiddle(Int n, ComplexPtrs<typename V::T> dst)
{
  VEC_TYPEDEFS(V);
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

  Int iter = 0;
  for(Int dft_size = 1; dft_size < n;)
  {
    iter++;
    if(dft_size == 1)
    {
      dft_size *= 4;
    }
    else if(dft_size >= V::vec_size)
    {
      if(dft_size * 4 > n)
        dft_size *= 2;
      else
      {
        Int vn = n / V::vec_size;
        Int vdft_size = dft_size / V::vec_size;

        Vec* row_re = (Vec*) dst.re + vn - 4 * vdft_size;
        Vec* row_im = (Vec*) dst.im + vn - 4 * vdft_size;
        for(Int i = vdft_size - 1; i >= 0; i--)
        {
          Complex<Vec> w1 = {row_re[i], row_im[i]};
          Complex<Vec> w2 = w1 * w1;
          Complex<Vec> w3 = w2 * w1;
          row_re[3 * i] = w1.re;
          row_re[3 * i + 1] = w2.re;
          row_re[3 * i + 2] = w3.re;
          row_im[3 * i] = w1.im;
          row_im[3 * i + 1] = w2.im;
          row_im[3 * i + 2] = w3.im;
        }

        dft_size *= 4;
      }
    }
    else
      dft_size *= 2;
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
FORCEINLINE void ct_dft_size_pass(
  Int n,
  ComplexPtrs<typename V::T> src,
  ComplexPtrs<typename V::T> twiddle,
  ComplexPtrs<typename V::T> dst)
{
  VEC_TYPEDEFS(V);
  ComplexPtrs<T> tw = twiddle + n - 2 * dft_size;
  Int vn = n / V::vec_size;
  auto vsrc0_re = (Vec*) src.re;
  auto vsrc0_im = (Vec*) src.im;
  auto vsrc1_re = (Vec*) src.re + vn / 2;
  auto vsrc1_im = (Vec*) src.im + vn / 2;
  auto vdst_re = (Vec*) dst.re;
  auto vdst_im = (Vec*) dst.im;
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
FORCEINLINE void first_two_passes(
  Int n, ComplexPtrs<typename V::T> src, ComplexPtrs<typename V::T> dst)
{
  VEC_TYPEDEFS(V);
  Int vn = n / V::vec_size;
  Vec* vsrc0_re = (Vec*) src.re;
  Vec* vsrc1_re = (Vec*) src.re + vn / 4;
  Vec* vsrc2_re = (Vec*) src.re + 2 * vn / 4;
  Vec* vsrc3_re = (Vec*) src.re + 3 * vn / 4;
  
  Vec* vsrc0_im = (Vec*) src.im;
  Vec* vsrc1_im = (Vec*) src.im + vn / 4;
  Vec* vsrc2_im = (Vec*) src.im + 2 * vn / 4;
  Vec* vsrc3_im = (Vec*) src.im + 3 * vn / 4;

  Vec* vdst_re = (Vec*) dst.re;
  Vec* vdst_im = (Vec*) dst.im;

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
FORCEINLINE void first_three_passes(
  Int n,
  ComplexPtrs<typename V::T> src,
  ComplexPtrs<typename V::T> twiddle,
  ComplexPtrs<typename V::T> dst)
{
  VEC_TYPEDEFS(V);
  Int vn = n / V::vec_size;

  Int l = vn / 8;
  
  Vec* vsrc_re = (Vec*) src.re;
  Vec* vsrc_im = (Vec*) src.im;

  Vec* vdst_re = (Vec*) dst.re;
  Vec* vdst_im = (Vec*) dst.im;

  ComplexPtrs<V> tw = twiddle + n - 2 * 8;
  for(Int i = 0; i < l; i++)
  {
    Complex<Vec> c0, c1, c2, c3;
    {
      Complex<Vec> a0 = {vsrc_re[i], vsrc_im[i]};
      Complex<Vec> a1 = {vsrc_re[i + 2 * l], vsrc_im[i + 2 * l]};
      Complex<Vec> a2 = {vsrc_re[i + 4 * l], vsrc_im[i + 4 * l]};
      Complex<Vec> a3 = {vsrc_re[i + 6 * l], vsrc_im[i + 6 * l]};
      Complex<Vec> b0 = a0 + a2;
      Complex<Vec> b1 = a0 - a2;
      Complex<Vec> b2 = a1 + a3; 
      Complex<Vec> b3 = a1 - a3; 
      c0 = b0 + b2; 
      c2 = b0 - b2;
      c1 = b1 + b3.mul_neg_i();
      c3 = b1 - b3.mul_neg_i();
    }

    Complex<Vec> c4, c5, c6, c7;
    {
      Complex<Vec> a0 = {vsrc_re[i + l], vsrc_im[i + l]};
      Complex<Vec> a1 = {vsrc_re[i + 3 * l], vsrc_im[i + 3 * l]};
      Complex<Vec> a2 = {vsrc_re[i + 5 * l], vsrc_im[i + 5 * l]};
      Complex<Vec> a3 = {vsrc_re[i + 7 * l], vsrc_im[i + 7 * l]};
      Complex<Vec> b0 = a0 + a2;
      Complex<Vec> b1 = a0 - a2;
      Complex<Vec> b2 = a1 + a3; 
      Complex<Vec> b3 = a1 - a3; 
      c4 = b0 + b2; 
      c5 = b0 - b2;
      c6 = b1 + b3.mul_neg_i();
      c7 = b1 - b3.mul_neg_i();
    }

    Complex<Vec> mul0 = c4 * (Complex<Vec>){tw.re[0], tw.im[0]};
    Complex<Vec> d0 = c0 + mul0;
    Complex<Vec> d4 = c0 - mul0;

    Complex<Vec> mul1 = c5 * (Complex<Vec>){tw.re[1], tw.im[1]};
    Complex<Vec> d1 = c1 + mul1;
    Complex<Vec> d5 = c1 - mul1;

    Complex<Vec> mul2 = c6 * (Complex<Vec>){tw.re[2], tw.im[2]};
    Complex<Vec> d2 = c2 + mul2;
    Complex<Vec> d6 = c2 - mul1;

    Complex<Vec> mul3 = c7 * (Complex<Vec>){tw.re[3], tw.im[3]};
    Complex<Vec> d3 = c3 + mul3;
    Complex<Vec> d7 = c3 - mul3;

    Int j = 8 * i;
    V::transpose(
      d0.re, d1.re, d2.re, d3.re, d4.re, d5.re, d6.re, d7.re,
      vdst_re[j], vdst_re[j + 1], vdst_re[j + 2], vdst_re[j + 3],
      vdst_re[j + 4], vdst_re[j + 5], vdst_re[j + 6], vdst_re[j + 7]);

    V::transpose(
      d0.im, d1.im, d2.im, d3.im, d4.im, d5.im, d6.im, d7.im,
      vdst_im[j], vdst_im[j + 1], vdst_im[j + 2], vdst_im[j + 3],
      vdst_im[j + 4], vdst_im[j + 5], vdst_im[j + 6], vdst_im[j + 7]);
  }
}

template<typename T>
FORCEINLINE void pass(
  Int n, Int dft_size,
  ComplexPtrs<T> src,
  ComplexPtrs<T> twiddle,
  ComplexPtrs<T> dst)
{
  for(Int i = 0; i < n / 2; i += dft_size)
  {
    T* re0_ptr = src.re + i;
    T* im0_ptr = src.im + i;
    T* re1_ptr = src.re + n / 2 + i;
    T* im1_ptr = src.im + n / 2 + i;
    T* dst0_re_ptr = dst.re + 2 * i;
    T* dst0_im_ptr = dst.im + 2 * i;
    T* dst1_re_ptr = dst.re + 2 * i + dft_size;
    T* dst1_im_ptr = dst.im + 2 * i + dft_size;

    for(Int j = 0; j < dft_size; j++)
    {
      T tw_re = twiddle.re[j];
      T tw_im = twiddle.im[j];
      T re0 = re0_ptr[j];
      T im0 = im0_ptr[j];
      T re1 = re1_ptr[j];
      T im1 = im1_ptr[j];
      T mul_re = tw_re * re1 - tw_im * im1;
      T mul_im = tw_re * im1 + tw_im * re1;
      dst0_re_ptr[j] = re0 + mul_re;
      dst1_re_ptr[j] = re0 - mul_re;
      dst0_im_ptr[j] = im0 + mul_im;
      dst1_im_ptr[j] = im0 - mul_im;
    }
  }
}

template<typename T>
FORCEINLINE void two_passes(
  Int n, Int dft_size,
  ComplexPtrs<T> src,
  ComplexPtrs<T> twiddle,
  ComplexPtrs<T> dst)
{
  T* re_ptr = src.re;
  T* im_ptr = src.im;
  T* dst_re_ptr = dst.re;
  T* dst_im_ptr = dst.im;
  T* tw_re = twiddle.re;
  T* tw_im = twiddle.im;

  Int l = n / 4;
  Int l2 = l * 2;
  Int l3 = l * 3;
  Int dft_size2 = dft_size * 2;
  Int dft_size3 = dft_size * 3;

  for(T* end = re_ptr + l; re_ptr < end;)
  {
    for(T* end1 = re_ptr + dft_size;;)
    {
      Complex<T> a0 = {re_ptr[0], im_ptr[0]};
      
      Complex<T> w1 = {tw_re[0], tw_im[0]};
      Complex<T> a1 = {re_ptr[l], im_ptr[l]};
      Complex<T> mul1 = w1 * a1;

      Complex<T> w2 = {tw_re[1], tw_im[1]};
      Complex<T> a2 = {re_ptr[l2], im_ptr[l2]};
      Complex<T> mul2 = w2 * a2;

      Complex<T> w3 = {tw_re[2], tw_im[2]};
      Complex<T> a3 = {re_ptr[l3], im_ptr[l3]};
      Complex<T> mul3 = w3 * a3;

      re_ptr++;
      im_ptr++;
      tw_re += 3;
      tw_im += 3;

      Complex<T> sum02 = a0 + mul2;
      Complex<T> dif02 = a0 - mul2;
      Complex<T> sum13 = mul1 + mul3;
      Complex<T> dif13 = mul1 - mul3;

      (sum02 + sum13).store(dst_re_ptr, dst_im_ptr, 0); 
      (sum02 - sum13).store(dst_re_ptr, dst_im_ptr, dft_size2);
      (dif02 + dif13.mul_neg_i()).store(dst_re_ptr, dst_im_ptr, dft_size);
      (dif02 - dif13.mul_neg_i()).store(dst_re_ptr, dst_im_ptr, dft_size3); 

      dst_re_ptr++;
      dst_im_ptr++;
      //We check the condition here, so that we don't need to check
      //it for the first iteration
      if(!(re_ptr < end1)) break;
    }

    dst_re_ptr += dft_size3;
    dst_im_ptr += dft_size3;
    tw_re -= dft_size3;
    tw_im -= dft_size3;
  }
}

template<typename V>
Int top_level_loop_iterations(Int n)
{
  Int iter = 0;
  for(Int dft_size = 1; dft_size < n;)
  {
    iter++;
    if(dft_size == 1)
    {
      dft_size *= 4;
    }
    else if(dft_size >= V::vec_size)
    {
      if(dft_size * 4 > n)
        dft_size *= 2;
      else
        dft_size *= 4;
    }
    else
      dft_size *= 2;
  }

  return iter;
}

Int most_significant_bit_index(Int n)
{
  Int r;
  for(r = -1; n; r++, n >>= 1) {}
  return r; 
}

template<typename V>
void fft(
  Int n,
  ComplexPtrs<typename V::T> src,
  ComplexPtrs<typename V::T> twiddle,
  ComplexPtrs<typename V::T> working,
  ComplexPtrs<typename V::T> dst)
{
  VEC_TYPEDEFS(V);

  auto is_odd = bool(top_level_loop_iterations<V>(n) & 1);
  auto current_src = src;
  auto current_dst = is_odd ? dst : working;
  auto next_dst = is_odd ? working : dst;

  for(Int dft_size = 1; dft_size < n;)
  {
    auto tw = twiddle + (n - 2 * dft_size);
    if(dft_size == 1)
    {
      first_two_passes<V>(n, current_src, current_dst);

      dft_size *= 4;
    } 
    else if(dft_size >= V::vec_size)
    {
      if(dft_size * 4 > n)
      {
        pass(
          n / V::vec_size,
          dft_size / V::vec_size,
          (ComplexPtrs<Vec>&) current_src,
          (ComplexPtrs<Vec>&) tw,
          (ComplexPtrs<Vec>&) current_dst);

        dft_size *= 2;
      }
      else
      {
        auto other_tw = twiddle + (n - 4 * dft_size); 
        two_passes(
          n / V::vec_size,
          dft_size / V::vec_size,
          (ComplexPtrs<Vec>&) current_src,
          (ComplexPtrs<Vec>&) other_tw,
          (ComplexPtrs<Vec>&) current_dst);

        dft_size *= 4;
      }
    }
    else
    {
      if(V::vec_size > 1 && dft_size == 1)
        ct_dft_size_pass<V, 1>(n, current_src, tw, current_dst);
      else if(V::vec_size > 2 && dft_size == 2)
        ct_dft_size_pass<V, 2>(n, current_src, tw, current_dst);
      else if(V::vec_size > 4 && dft_size == 4)
        ct_dft_size_pass<V, 4>(n, current_src, tw, current_dst);
      else if(V::vec_size > 8 && dft_size == 8)
        ct_dft_size_pass<V, 8>(n, current_src, tw, current_dst);

      dft_size *= 2;
    }

    swap(next_dst, current_dst);
    current_src = next_dst;
  }
}

#include <string>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include <fstream>
#include <unistd.h>

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

int main(int argc, char** argv)
{
  printf("%d\n", setpriority(PRIO_PROCESS, 0, -20));
  typedef AvxFloat V;
  //typedef SseFloat V;
  VEC_TYPEDEFS(V);

  Int log2n = atoi(argv[1]);
  Int n = 1 << log2n;
  ComplexPtrs<T> twiddle = {new T[n], new T[n]};
  init_twiddle<V>(n, twiddle);

  ComplexPtrs<T> src = {new T[n], new T[n]};
  ComplexPtrs<T> working = {new T[n], new T[n]};
  ComplexPtrs<T> dst = {new T[n], new T[n]};

  std::fill_n(dst.re, n, T(0));
  std::fill_n(dst.im, n, T(0));

  for(Int i = 0; i < n; i++)
  {
    src.im[i] = 0;
    T x = std::min(i, n - i) / T(n);
    src.re[i] = T(1) / (1 + 100 * x * x);
  }

  dump(twiddle.re, n, "twiddle_re.float32");
  dump(twiddle.im, n, "twiddle_im.float32");
  dump(src.re, n, "src_re.float32");
  dump(src.im, n, "src_im.float32");

  double t = get_time();
  for(int i = 0; i < 100LL*1000*1000*1000 / (5 * n * log2n); i++)
    fft<V>(n, src, twiddle, working, dst);

  printf("time %f\n", get_time() - t);

  dump(dst.re, n, "dst_re.float32");
  dump(dst.im, n, "dst_im.float32");

  return 0;
}
