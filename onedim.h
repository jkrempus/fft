#ifndef FFT_ONEDIM_H
#define FFT_ONEDIM_H

#include "common.h"

namespace onedim
{
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
struct Fft
{
  Int n;
  Int im_off;
  T* working0;
  T* working1;
  T* twiddle;
  T* tiny_twiddle;
  Step<T> steps[8 * sizeof(Int)];
  Int nsteps;
  Int ncopies;
  typedef void (*tiny_transform_fun_type)(T* src, T* dst, Int im_off);
  tiny_transform_fun_type tiny_transform_fun;
};

template<typename T> struct Ifft;

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
void last_two_passes_impl(Int n, const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  Int vn = n / V::vec_size;
  auto tw = arg.twiddle;

  auto src = arg.src;
  
  auto dst0 = arg.dst; 
  auto dst1 = dst0 + vn / 4 * stride<V, DstCf>(); 
  auto dst2 = dst1 + vn / 4 * stride<V, DstCf>(); 
  auto dst3 = dst2 + vn / 4 * stride<V, DstCf>(); 

  for(BitReversed br(vn / 4); br.i < vn / 4; br.advance())
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

    Int d = br.br * stride<V, DstCf>();
    DstCf::store(d0, dst0 + d, arg.im_off);
    DstCf::store(d1, dst1 + d, arg.im_off);
    DstCf::store(d2, dst2 + d, arg.im_off);
    DstCf::store(d3, dst3 + d, arg.im_off);
  }
}

template<typename V, typename DstCf>
void last_two_passes(const Arg<typename V::T>& arg)
{
  last_two_passes_impl<V, DstCf>(arg.n, arg);
}

template<typename V, typename DstCf, Int n>
void last_two_passes(const Arg<typename V::T>& arg)
{
  last_two_passes_impl<V, DstCf>(n, arg);
}

template<typename V, typename DstCf>
FORCEINLINE void last_pass_impl(Int n, const Arg<typename V::T>& arg)
{
  VEC_TYPEDEFS(V);
  Int vn = n / V::vec_size;
  auto tw = arg.twiddle;

  auto src = arg.src;
  auto dst0 = arg.dst; 
  auto dst1 = dst0 + vn / 2 * stride<V, DstCf>(); 

  for(BitReversed br(vn / 2); br.i < vn / 2; br.advance())
  {
    C a0 = load<V, cf::Vec>(src, 0);
    C mul = load<V, cf::Vec>(src + stride<V, cf::Vec>(), 0) * load<V, cf::Vec>(tw, 0);
    tw += stride<V, cf::Vec>();
    src += 2 * stride<V, cf::Vec>();

    Int d = br.br * stride<V, DstCf>();
    DstCf::store(a0 + mul, dst0 + d, arg.im_off);
    DstCf::store(a0 - mul, dst1 + d, arg.im_off);
  }
}

template<typename V, typename DstCf>
void last_pass(const Arg<typename V::T>& arg)
{
  last_pass_impl<V, DstCf>(arg.n, arg);
}

template<typename V, typename DstCf, Int n>
void last_pass(const Arg<typename V::T>& arg)
{
  last_pass_impl<V, DstCf>(n, arg);
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
  //const Int br_table[] = {0, 2, 1, 3};
  constexpr Int br_table[] = {0, 4, 2, 6, 1, 5, 3, 7};
  //constexpr Int br_table[] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
  constexpr int m = sizeof(br_table) / sizeof(br_table[0]);

  if(vn < m * m)
  {
    for(BitReversed br(vn); br.i < vn; br.advance())
      DstCf::template store<stream_flag>(
        load<V, cf::Vec>(arg.src + br.i * stride<V, cf::Vec>(), 0),
        arg.dst + br.br * stride<V, DstCf>(),
        im_off);
  }
  else
  {
    Int stride_ = vn / m;

    for(BitReversed br(vn / (m * m)); br.i < vn / (m * m); br.advance())
    {
      T* src = arg.src + br.i * m * stride<V, cf::Vec>();
      T* dst = arg.dst + br.br * m * stride<V, DstCf>();

      for(Int i0 = 0; i0 < m; i0++)
      {
        T* s = src + br_table[i0] * stride<V, cf::Vec>();
        T* d = dst + i0 * stride_ * stride<V, DstCf>();
        for(Int i1 = 0; i1 < m; i1++)
        {
          auto this_s = s + br_table[i1] * stride_ * stride<V, cf::Vec>();
          DstCf::template store<stream_flag>(
            load<V, cf::Vec>(this_s, 0),
            d + i1 * stride<V, DstCf>(), im_off);
        }
      }
    }
  }

  _mm_sfence();
}

template<typename V, typename DstCf>
FORCEINLINE void last_three_passes_impl(
  Int n,
  Int im_off,
  typename V::T* src,
  typename V::T* twiddle,
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

  for(BitReversed br(l1); br.i < l1; br.advance())
  {
    auto d = dst + br.br * stride<V, DstCf>();
    auto s = src + 8 * stride<V, cf::Vec>() * br.i;
    auto tw = twiddle + 5 * stride<V, cf::Vec>() * br.i;

    C a0, a1, a2, a3, a4, a5, a6, a7;
    {
      C tw0 = load<V, cf::Vec>(tw, 0);
      C tw1 = load<V, cf::Vec>(tw + stride<V, cf::Vec>(), 0);
      C tw2 = load<V, cf::Vec>(tw + 2 * stride<V, cf::Vec>(), 0);

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
      C tw3 = load<V, cf::Vec>(tw + 3 * stride<V, cf::Vec>(), 0);
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
      C tw4 = load<V, cf::Vec>(tw + 4 * stride<V, cf::Vec>(), 0);
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
  }
}

template<typename V, typename DstCf>
void last_three_passes_vec(const Arg<typename V::T>& arg)
{
  last_three_passes_impl<V, DstCf>(
    arg.n, arg.im_off, arg.src, arg.twiddle, arg.dst);
}

template<typename V, typename DstCf, Int n>
void last_three_passes_vec_ct_size(const Arg<typename V::T>& arg)
{
  last_three_passes_impl<V, DstCf>(
    n, arg.im_off, arg.src, arg.twiddle, arg.dst);
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

template<int len, typename T>
struct ReImTable
{
  T re[len];
  T im[len];
};

template<int n, int vsz, typename T>
constexpr ReImTable<(n > vsz ? n : vsz), T>
create_ct_sized_fft_twiddle_table()
{
  constexpr int len = n > vsz ? n : vsz;
  ReImTable<len, T> r = {0};
  for(int i = 0; i < n; i++)
  {
    T re = T(1);
    T im = T(0);
    int table_i = 1;
    for(int bit = n / 2; bit > 0; bit >>= 1, table_i++)
      if((i & bit) != 0)
      {
        T table_re = SinCosTable<T>::cos[table_i];
        T table_im = SinCosTable<T>::sin[table_i];

        T new_re = table_re * re - table_im * im;
        T new_im = table_re * im + table_im * re;

        re = new_re;
        im = new_im;
      }

    r.re[i] = re;
    r.im[i] = -im;
  }

  for(int i = 0; i < vsz; i++)
  {
    r.re[i] = r.re[i & (n - 1)];
    r.im[i] = r.im[i & (n - 1)];
  }

  return r;
}

template<int n, int vsz, typename T>
struct CtSizedFftTwiddleTable
{
  static constexpr ReImTable<(n > vsz ? n : vsz), float> value =
    create_ct_sized_fft_twiddle_table<n, vsz, float>();
};

template<int n, int vsz, typename T>
constexpr ReImTable<(n > vsz ? n : vsz), float>
CtSizedFftTwiddleTable<n, vsz, T>::value;

template<bool cond, typename T> struct enable_if { };
template<typename T> struct enable_if<true, T> { using type = T; };

template<bool cond, typename T>
using enif = typename enable_if<cond, T>::type;

constexpr bool is_power_of_4(Int n)
{
  while(n >= 4) n /= 4;
  return n == 1;
}

template<typename V, Int vn, Int dft_sz>
FORCEINLINE enif<(dft_sz < V::vec_size), void> tiny_transform_pass(
  typename V::Vec (&src_re)[vn],
  typename V::Vec (&src_im)[vn],
  typename V::Vec (&dst_re)[vn],
  typename V::Vec (&dst_im)[vn])
{
  VEC_TYPEDEFS(V);
  constexpr Int vsz = V::vec_size;
  auto& table = CtSizedFftTwiddleTable<vn * vsz, vsz, T>::value;

  for(Int i = 0; i < vn / 2; i++)
  {
    C a = { src_re[i], src_im[i] };
    C b = { src_re[i + vn / 2], src_im[i + vn / 2] };
    C t = { table.re[0], table.im[0] };
    if(dft_sz > 1) b = b * t;
    C dst_a = a + b;
    C dst_b = a - b;

    V::template interleave_multi<vsz / dft_sz>(
      dst_a.re, dst_b.re, dst_re[2 * i], dst_re[2 * i + 1]);

    V::template interleave_multi<vsz / dft_sz>(
      dst_a.im, dst_b.im, dst_im[2 * i], dst_im[2 * i + 1]);
  }
}

template<typename V, Int vn, Int dft_sz>
FORCEINLINE enif<(dft_sz >= V::vec_size), void> tiny_transform_pass(
  typename V::Vec (&src_re)[vn],
  typename V::Vec (&src_im)[vn],
  typename V::Vec (&dst_re)[vn],
  typename V::Vec (&dst_im)[vn])
{
  VEC_TYPEDEFS(V);
  constexpr Int vsz = V::vec_size;
  constexpr Int vdft_sz = dft_sz / vsz;
  auto& table = CtSizedFftTwiddleTable<vn * vsz, vsz, T>::value;

  for(Int i = 0; i < vn / 2; i += vdft_sz)
  {
    for(Int j = 0; j < vdft_sz; j++)
    {
      C src_a = { src_re[i + j], src_im[i + j] };
      C src_b = { src_re[i + j + vn / 2], src_im[i + j + vn / 2] };
      C t = { table.re[i], table.im[i] };
      C m = src_b * t;
      C dst_a = src_a + m; 
      C dst_b = src_a - m;

      dst_re[2 * i + j] = dst_a.re;
      dst_im[2 * i + j] = dst_a.im;

      dst_re[2 * i + j + vsz] = dst_b.re;
      dst_im[2 * i + j + vsz] = dst_b.im;
    }
  }
}

template<typename V, typename SrcCf, typename DstCf, Int n>
void tiny_transform(typename V::T* src, typename V::T* dst, Int im_off)
{
  VEC_TYPEDEFS(V);
  constexpr Int vsz = V::vec_size;

  //Round up just to make it compile
  constexpr Int vn = (n + V::vec_size - 1) / V::vec_size;

  Vec a_re[vn];
  Vec a_im[vn];
  Vec b_re[vn];
  Vec b_im[vn];

  for(Int i = 0; i < vn; i++)
  {
    auto c = load<V, SrcCf>(src + i * stride<V, SrcCf>(), im_off);
    a_re[i] = c.re;
    a_im[i] = c.im;
  }

  if(n >  1) tiny_transform_pass<V, vn,  1>(a_re, a_im, b_re, b_im);
  if(n >  2) tiny_transform_pass<V, vn,  2>(b_re, b_im, a_re, a_im);
  if(n >  4) tiny_transform_pass<V, vn,  4>(a_re, a_im, b_re, b_im);
  if(n >  8) tiny_transform_pass<V, vn,  8>(b_re, b_im, a_re, a_im);
  if(n > 16) tiny_transform_pass<V, vn, 16>(a_re, a_im, b_re, b_im);

  for(Int i = 0; i < vn; i++)
  {
    C c;
    constexpr bool result_in_a = is_power_of_4(n);
    if(result_in_a) c = { a_re[i], a_im[i] };
    else c = { b_re[i], b_im[i] };

    DstCf::store(c, dst + i * stride<V, DstCf>(), im_off);
  }
}

template<typename V, typename SrcCf, typename DstCf>
void init_steps(Fft<typename V::T>& state)
{
  VEC_TYPEDEFS(V);
  Int step_index = 0;
  state.ncopies = 0;

  if(state.n <= 8 || state.n < V::vec_size * V::vec_size)
  {
    state.nsteps = 0;
    state.tiny_transform_fun = 
      state.n ==  1 ?  &tiny_transform<V, SrcCf, DstCf,  1> :
      state.n ==  2 ?  &tiny_transform<V, SrcCf, DstCf,  2> :
      state.n ==  4 ?  &tiny_transform<V, SrcCf, DstCf,  4> :
      state.n ==  8 ?  &tiny_transform<V, SrcCf, DstCf,  8> :
      state.n == 16 ?  &tiny_transform<V, SrcCf, DstCf, 16> :
      state.n == 32 ?  &tiny_transform<V, SrcCf, DstCf, 32> : nullptr;

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
      if((state.n < large_fft_size) && dft_size * 8 == state.n)
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

  init_steps<V, SrcCf, DstCf>(*state);

  init_twiddle<V>([state](Int s, Int){ return state->steps[s].npasses; },
    n, state->working0, state->twiddle, state->tiny_twiddle);

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
    DstCf::template store<0, S>({r0 + r1, r0 - r1}, dst, dst_off);
  }
  else
  {
    SC r0 = load<S, SrcCf>(src, src_off);
    DstCf::template store<0, S>({r0.re + r0.im, 0}, dst, dst_off);
    DstCf::template store<0, S>(
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
  auto complex_state = ((Fft<T>*) state->state);
  state->real_pass(
    complex_state->n * 2,
    src,
    align_size<T>(complex_state->n + 1),
    state->twiddle,
    complex_state->working1,
    complex_state->n);

  ifft(state->state, complex_state->working1, dst);
}
}
#endif
