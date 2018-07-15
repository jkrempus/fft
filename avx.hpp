#ifndef AFFT_AVX_HPP
#define AFFT_AVX_HPP

#include "common.hpp"

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
  static Vec load(const T* p)
  {
    return (flags & stream_flag) ?
      _mm256_castsi256_ps(_mm256_stream_load_si256((__m256i*) p)) :
      _mm256_load_ps(p);
  }

  static Vec unaligned_load(const T* p) { return _mm256_loadu_ps(p); }
  template<Uint flags = 0>
  static void store(Vec val, T* p)
  {
    if((flags & stream_flag))
      _mm256_stream_ps(p, val);
    else
      _mm256_store_ps(p, val);
  }

  static void unaligned_store(Vec val, T* p) { _mm256_storeu_ps(p, val); }

  static void sfence(){ _mm_sfence(); }
};
#endif

#endif
