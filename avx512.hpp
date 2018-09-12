#ifndef AFFT_AVX_HPP
#define AFFT_AVX_HPP

#include "common.hpp"

#include <immintrin.h>

struct AvxFloat
{
  typedef float T;
  typedef __v16sf Vec;
  const static Int vec_size = 16;

  template<Int elements_per_vec>
  static FORCEINLINE void interleave_multi(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    //TODO
  }

  static FORCEINLINE void interleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    //TODO
  }

  static FORCEINLINE void deinterleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    //TODO
  }

  static FORCEINLINE void unpack(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    r0 = _mm512_unpacklo_ps(a0, a1);
    r1 = _mm512_unpackhi_ps(a0, a1);
  }

  static FORCEINLINE void transpose_8x4(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    r0 = _mm512_permutex2var_ps(
      a0,
      _mm512_setr_epi32(
        0, 4, 8, 12, 16, 20, 24, 28,
        1, 5, 9, 13, 17, 21, 25, 29),
      a1);

    r1 = _mm512_permutex2var_ps(
      a0,
      _mm512_setr_epi32(
        2, 6, 10, 14, 18, 22, 26, 30,
        3, 7, 11, 15, 19, 23, 27, 31),
      a1);
  }

  template<Int stride>
  static void NOINLINE transposed_store(
    Vec (&src)[16], T* dst)
  {
    Vec a[16];
    for(Int i = 0; i < 8; i++)
      unpack(src[i], src[i + 8], a[i], a[i + 8]);

    Vec b[16];
    for(Int i = 0; i < 8; i++)
    {
      Int lo_mask = 3;
      Int j = (i & lo_mask) | ((i & ~lo_mask) << 1);
      transpose_8x4(a[j], a[j + 4], b[j], b[j + 4]);
    }

    Vec c[16];

    for(Int i = 0; i < 8; i++)
    {
      Int lo_mask = 1;
      Int j = (i & lo_mask) | ((i & ~lo_mask) << 1);
      unpack(b[j], b[j + 2], c[j], c[j + 2]);
    }

    Vec d[16];
    for(Int i = 0; i < 8; i++)
      unpack(c[2 * i], c[2 * i + 1], d[2 * i], d[2 * i + 1]);

    for(int i = 0; i < 16; i++)
    {
      Int src_i = (i >> 2) | ((i & Int(3)) << 2);
      store(d[src_i], dst + stride * i);
    }
  }

  static Vec FORCEINLINE vec(T a){ return _mm512_set1_ps(a); }

  static Vec reverse(Vec v)
  {
    //TODO
    return vec(0.0f);
  }

  template<Uint flags = 0>
  static Vec load(const T* p)
  {
    return (flags & stream_flag) ?
      _mm512_castsi512_ps(_mm512_stream_load_si512((__m512i*) p)) :
      _mm512_load_ps(p);
  }

  static Vec unaligned_load(const T* p) { return _mm512_loadu_ps(p); }
  template<Uint flags = 0>
  static void store(Vec val, T* p)
  {
    if((flags & stream_flag))
      _mm512_stream_ps(p, val);
    else
      _mm512_store_ps(p, val);
  }

  static void unaligned_store(Vec val, T* p) { _mm512_storeu_ps(p, val); }

  static void sfence(){ _mm_sfence(); }


private:

  
};

#endif
