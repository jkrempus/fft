#ifndef AFFT_AVX_HPP
#define AFFT_AVX_HPP

#include "common.hpp"

#include <immintrin.h>

struct Avx512Float
{
  typedef float T;
  typedef __v16sf Vec;
  static constexpr Int vec_size = 16;
  static constexpr bool prefer_three_passes = true;

  static FORCEINLINE Vec madd(Vec a, Vec b, Vec c)
  {
    return _mm512_fmadd_ps(a, b, c);
  }

  static FORCEINLINE Vec msub(Vec a, Vec b, Vec c)
  {
    return _mm512_fmsub_ps(a, b, c);
  }

  template<Int elements_per_vec>
  static FORCEINLINE void interleave_multi(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    static_assert(elements_per_vec > 1 && elements_per_vec <= 16);

    if constexpr(elements_per_vec == 16)
    {
      r0 = _mm512_permutex2var_ps(a0, _mm512_setr_epi32(
        0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23), a1);

      r1 = _mm512_permutex2var_ps(a0, _mm512_setr_epi32(
        8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31), a1);
    }
    else if constexpr(elements_per_vec == 8)
    {
      r0 = _mm512_permutex2var_ps(a0, _mm512_setr_epi32(
        0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23), a1);

      r1 = _mm512_permutex2var_ps(a0, _mm512_setr_epi32(
        8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31), a1);
    }
    else if constexpr(elements_per_vec == 4)
    {
      r0 = _mm512_permutex2var_ps(a0, _mm512_setr_epi32(
        0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23), a1);

      r1 = _mm512_permutex2var_ps(a0, _mm512_setr_epi32(
        8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31), a1);
    }
    else if constexpr(elements_per_vec == 2)
    {
      r0 = _mm512_permutex2var_ps(a0, _mm512_setr_epi32(
        0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23), a1);

      r1 = _mm512_permutex2var_ps(a0, _mm512_setr_epi32(
        8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31), a1);
    }
  }

  static FORCEINLINE void interleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
    return interleave_multi<16>(a0, a1, r0, r1);
  }

  static FORCEINLINE void deinterleave(Vec a0, Vec a1, Vec& r0, Vec& r1)
  {
      r0 = _mm512_permutex2var_ps(a0, _mm512_setr_epi32(
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30), a1);

      r1 = _mm512_permutex2var_ps(a0, _mm512_setr_epi32(
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31), a1);
  }

  template<Int stride>
  static FORCEINLINE void transposed_store(
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

  static FORCEINLINE Vec vec(T a){ return _mm512_set1_ps(a); }

  static FORCEINLINE Vec reverse(Vec v)
  {
    return _mm512_permutexvar_ps(_mm512_setr_epi32(
      15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0), v);
  }

  template<Uint flags = 0>
  static FORCEINLINE Vec load(const T* p)
  {
    return (flags & stream_flag) ?
      _mm512_castsi512_ps(_mm512_stream_load_si512((__m512i*) p)) :
      _mm512_load_ps(p);
  }

  static FORCEINLINE Vec unaligned_load(const T* p)
  {
    return _mm512_loadu_ps(p);
  }

  template<Uint flags = 0>
  static FORCEINLINE void store(Vec val, T* p)
  {
    if((flags & stream_flag))
      _mm512_stream_ps(p, val);
    else
      _mm512_store_ps(p, val);
  }

  static FORCEINLINE void unaligned_store(Vec val, T* p)
  {
    _mm512_storeu_ps(p, val);
  }

  static void sfence(){ _mm_sfence(); }

private:

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
};

#endif
