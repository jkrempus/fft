#ifndef AFFT_SSE_HPP
#define AFFT_SSE_HPP

#include "common.hpp"

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
  
  static Vec FORCEINLINE vec(T a){ return _mm_set1_ps(a); }
  
  static Vec reverse(Vec v)
  {
    return _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 1, 2, 3));
  }

  template<Uint flags = 0>
  static Vec load(const T* p)
  {
    return (flags & stream_flag) ? 
      _mm_castsi128_ps(_mm_stream_load_si128((__m128i*) p)) :
      _mm_load_ps(p);
  }

  static Vec unaligned_load(const T* p) { return _mm_loadu_ps(p); }
  template<Uint flags = 0>
  static void store(Vec val, T* p)
  {
    if((flags & stream_flag))
      _mm_stream_ps(p, val);
    else
      _mm_store_ps(p, val);
  }

  static void unaligned_store(Vec val, T* p) { _mm_storeu_ps(p, val); }

  static void sfence(){ _mm_sfence(); }
};

struct SseDouble
{
  typedef double T;
  typedef __m128d Vec;
  const static Int vec_size = 2;
  
  template<Int elements_per_vec>
  static FORCEINLINE void interleave_multi(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    if(elements_per_vec == 2)
    {
      r0 = _mm_unpacklo_pd(a0, a1);
      r1 = _mm_unpackhi_pd(a0, a1);
    }
  }

  static FORCEINLINE void interleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    r0 = _mm_unpacklo_pd(a0, a1);
    r1 = _mm_unpackhi_pd(a0, a1);
  }

  static FORCEINLINE void deinterleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    r0 = _mm_unpacklo_pd(a0, a1);
    r1 = _mm_unpackhi_pd(a0, a1);
  }
  
  static void FORCEINLINE transpose(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    r0 = _mm_unpacklo_pd(a0, a1);
    r1 = _mm_unpackhi_pd(a0, a1);
  }

  static Vec FORCEINLINE vec(T a){ return _mm_set1_pd(a); }

  static Vec reverse(Vec v)
  {
    return _mm_shuffle_pd(v, v, 0x1);
  }

  template<Uint flags = 0>
  static Vec load(const T* p)
  {
    return (flags & stream_flag) ? 
      _mm_castsi128_pd(_mm_stream_load_si128((__m128i*) p)) :
      _mm_load_pd(p);
  }

  static Vec unaligned_load(const T* p) { return _mm_loadu_pd(p); }
  template<Uint flags = 0>
  static void store(Vec val, T* p)
  {
    if((flags & stream_flag))
      _mm_stream_pd(p, val);
    else
      _mm_store_pd(p, val);
  }

  static void unaligned_store(Vec val, T* p) { _mm_storeu_pd(p, val); }

  static void sfence(){ _mm_sfence(); }
};
#endif

#endif
