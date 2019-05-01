#ifndef AFFT_AVX_HPP
#define AFFT_AVX_HPP

#include "common.hpp"

#include <immintrin.h>

struct AvxFloat
{
  typedef float T;
  typedef __m256 Vec;
  static constexpr Int vec_size = 8;
  static constexpr bool prefer_three_passes = false;

  template<Int elements_per_vec>
  static FORCEINLINE void interleave_multi(Vec a0, Vec a1, Vec& r0, Vec& r1)
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

  template<Int stride>
  static FORCEINLINE void transposed_store(
    Vec a0, Vec a1, Vec a2, Vec a3,
    Vec a4, Vec a5, Vec a6, Vec a7,
    T* dst)
  {
    transpose4x4_two(a0, a1, a2, a3);
    transpose4x4_two(a4, a5, a6, a7);

    Vec r0, r1, r2, r3, r4, r5, r6, r7;

    transpose_128(a0, a4, r0, r4);
    store(r0, dst + 0 * stride);
    store(r4, dst + 4 * stride);

    transpose_128(a1, a5, r1, r5);
    store(r1, dst + 1 * stride);
    store(r5, dst + 5 * stride);

    transpose_128(a2, a6, r2, r6);
    store(r2, dst + 2 * stride);
    store(r6, dst + 6 * stride);

    transpose_128(a3, a7, r3, r7);
    store(r3, dst + 3 * stride);
    store(r7, dst + 7 * stride);
  }

  static FORCEINLINE Vec madd(Vec a, Vec b, Vec c)
  {
    return _mm256_fmadd_ps(a, b, c);
  }

  static FORCEINLINE Vec msub(Vec a, Vec b, Vec c)
  {
    return _mm256_fmsub_ps(a, b, c);
  }

  static FORCEINLINE Vec vec(T a){ return _mm256_set1_ps(a); }

//private:
  static FORCEINLINE void transpose_128(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    r0 = _mm256_permute2f128_ps(a0, a1, _MM_SHUFFLE(0, 2, 0, 0)),
    r1 = _mm256_permute2f128_ps(a0, a1, _MM_SHUFFLE(0, 3, 0, 1));
  }

  static FORCEINLINE void transpose4x4_two(Vec& a0, Vec& a1, Vec& a2, Vec& a3)
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

  static FORCEINLINE Vec reverse(Vec v)
  {
    v = _mm256_shuffle_ps(v, v, _MM_SHUFFLE(0, 1, 2, 3));
    return _mm256_permute2f128_ps(v, v, _MM_SHUFFLE(0, 0, 0, 1));
  }

  static FORCEINLINE __m128 load_128(const T* p) { return _mm_load_ps(p); }
  static FORCEINLINE Vec load(const T* p) { return _mm256_load_ps(p); }
  static FORCEINLINE Vec unaligned_load(const T* p)
  {
    return _mm256_loadu_ps(p);
  }

  static FORCEINLINE void load_deinterleaved(const T* src, Vec& dst0, Vec& dst1)
  {
    Vec a0 = _mm256_insertf128_ps(
      _mm256_castps128_ps256(load_128(src)),
      load_128(src + 8), 1);

    Vec a1 = _mm256_insertf128_ps(
      _mm256_castps128_ps256(load_128(src + 4)),
      load_128(src + 12), 1);

    dst0 = _mm256_shuffle_ps(a0, a1, _MM_SHUFFLE(2, 0, 2, 0));
    dst1 = _mm256_shuffle_ps(a0, a1, _MM_SHUFFLE(3, 1, 3, 1));
  }

  static FORCEINLINE void store(Vec val, T* p) { _mm256_store_ps(p, val); }
  static FORCEINLINE void stream_store(Vec val, T* p)
  {
    _mm256_stream_ps(p, val);
  }

  static FORCEINLINE void unaligned_store(Vec val, T* p)
  {
    _mm256_storeu_ps(p, val);
  }

  static void sfence(){ _mm_sfence(); }
};

struct AvxDouble
{
  typedef double T;
  typedef __m256d Vec;
  static constexpr Int vec_size = 4;
  static constexpr bool prefer_three_passes = false;

  template<Int elements_per_vec>
  static FORCEINLINE void interleave_multi(
    Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    if(elements_per_vec == 4)
      interleave(a0, a1, r0, r1);
    else if (elements_per_vec == 2)
      transpose_128(a0, a1, r0, r1);
  }

  static FORCEINLINE void interleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    r0 = _mm256_unpacklo_pd(a0, a1);
    r1 = _mm256_unpackhi_pd(a0, a1);
    transpose_128(r0, r1, r0, r1);
  }

  static FORCEINLINE void deinterleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    transpose_128(a0, a1, a0, a1);
    r0 = _mm256_unpacklo_pd(a0, a1);
    r1 = _mm256_unpackhi_pd(a0, a1);
  }

  template<Int stride>
  static FORCEINLINE void transposed_store(
    Vec a0, Vec a1, Vec a2, Vec a3, T* dst)
  {
    Vec b0, b1, b2, b3;
    b0 = _mm256_unpacklo_pd(a0, a1);
    b1 = _mm256_unpackhi_pd(a0, a1);
    b2 = _mm256_unpacklo_pd(a2, a3);
    b3 = _mm256_unpackhi_pd(a2, a3);

    Vec r0, r1, r2, r3;

    transpose_128(b0, b2, r0, r2);
    store(r0, dst + 0 * stride);
    store(r2, dst + 2 * stride);

    transpose_128(b1, b3, r1, r3);
    store(r1, dst + 1 * stride);
    store(r3, dst + 3 * stride);
  }

  static FORCEINLINE Vec madd(Vec a, Vec b, Vec c)
  {
    return _mm256_fmadd_pd(a, b, c);
  }

  static FORCEINLINE Vec msub(Vec a, Vec b, Vec c)
  {
    return _mm256_fmsub_pd(a, b, c);
  }

  static FORCEINLINE Vec vec(T a){ return _mm256_set1_pd(a); }

//private:
  static FORCEINLINE void transpose_128(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    r0 = _mm256_permute2f128_pd(a0, a1, _MM_SHUFFLE(0, 2, 0, 0)),
    r1 = _mm256_permute2f128_pd(a0, a1, _MM_SHUFFLE(0, 3, 0, 1));
  }

  static FORCEINLINE Vec reverse(Vec v)
  {
    v = _mm256_shuffle_pd(v, v, 0x5);
    return _mm256_permute2f128_pd(v, v, _MM_SHUFFLE(0, 0, 0, 1));
  }

  static FORCEINLINE Vec load(const T* p) { return _mm256_load_pd(p); }
  static FORCEINLINE Vec unaligned_load(const T* p)
  {
    return _mm256_loadu_pd(p);
  }

  static FORCEINLINE void load_deinterleaved(const T* src, Vec& r0, Vec& r1)
  {
    deinterleave(load(src), load(src + vec_size), r0, r1);
  }

  static FORCEINLINE void store(Vec val, T* p) { _mm256_store_pd(p, val); }
  static FORCEINLINE void stream_store(Vec val, T* p)
  {
      _mm256_stream_pd(p, val);
  }

  static FORCEINLINE void unaligned_store(Vec val, T* p)
  {
    _mm256_storeu_pd(p, val);
  }

  static void sfence(){ _mm_sfence(); }
};

#ifdef _MSC_VER
  FORCEINLINE __m256 operator+(__m256 a, __m256 b)
  {
    return _mm256_add_ps(a, b);
  }

  FORCEINLINE __m256 operator-(__m256 a, __m256 b)
  {
    return _mm256_sub_ps(a, b);
  }

  FORCEINLINE __m256 operator-(__m256 a)
  {
    return _mm256_sub_ps(_mm256_setzero_ps(), a);
  }

  FORCEINLINE __m256 operator*(__m256 a, __m256 b)
  {
    return _mm256_mul_ps(a, b);
  }

  FORCEINLINE __m256d operator+(__m256d a, __m256d b)
  {
    return _mm256_add_pd(a, b);
  }

  FORCEINLINE __m256d operator-(__m256d a, __m256d b)
  {
    return _mm256_sub_pd(a, b);
  }

  FORCEINLINE __m256d operator-(__m256d a)
  {
    return _mm256_sub_pd(_mm256_setzero_pd(), a);
  }

  FORCEINLINE __m256d operator*(__m256d a, __m256d b)
  {
    return _mm256_mul_pd(a, b);
  }
#endif

#endif
