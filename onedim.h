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
  T* dst;
};

template<typename T>
struct Fft
{
  typedef void(*transform_fun_type)(
    const Fft<T>* state,
    const T* src_re, const T* src_im,
    T* dst_re, T* dst_im);

  Int n;
  transform_fun_type transform_fun;
  T* working;
  T** twiddle;
};

template<typename T> struct Ifft;

template<typename V, typename SrcCf>
void first_two_passes(
  Int n, const ET<V>* src_re, const ET<V>* src_im, ET<V>* dst)
{
  VEC_TYPEDEFS(V);
  Int l = n * SrcCf::idx_ratio / 4;
  Int im_off = src_im - src_re;
  const T* src0 = src_re;
  const T* src1 = src_re + l;
  const T* src2 = src_re + 2 * l;
  const T* src3 = src_re + 3 * l;

  for(const T* end = src1; src0 < end;)
  {
    C a0 = load<V, SrcCf>(src0, im_off);
    C a1 = load<V, SrcCf>(src1, im_off);
    C a2 = load<V, SrcCf>(src2, im_off);
    C a3 = load<V, SrcCf>(src3, im_off);
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
void first_three_passes(
  Int n, const ET<V>* src_re, const ET<V>* src_im, ET<V>* dst)
{
  VEC_TYPEDEFS(V);
  Int im_off = src_im - src_re;
  Int l = n / 8 * SrcCf::idx_ratio;
  Vec invsqrt2 = V::vec(SinCosTable<T>::cos[2]);

  for(T* end = dst + n * cf::Vec::idx_ratio; dst < end;)
  {
    C c0, c1, c2, c3;
    {
      C a0 = load<V, SrcCf>(src_re + 0 * l, im_off);
      C a1 = load<V, SrcCf>(src_re + 2 * l, im_off);
      C a2 = load<V, SrcCf>(src_re + 4 * l, im_off);
      C a3 = load<V, SrcCf>(src_re + 6 * l, im_off);
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
      C a0 = load<V, SrcCf>(src_re + 1 * l, im_off);
      C a1 = load<V, SrcCf>(src_re + 3 * l, im_off);
      C a2 = load<V, SrcCf>(src_re + 5 * l, im_off);
      C a3 = load<V, SrcCf>(src_re + 7 * l, im_off);
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

    src_re += stride<V, SrcCf>();

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
  }
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

  auto tw = arg.twiddle;
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
void last_two_passes(
  Int n, const ET<V>* src, const ET<V>* tw, ET<V>* dst_re, ET<V>* dst_im)
{
  VEC_TYPEDEFS(V);
  Int vn = n / V::vec_size;
  Int im_off = dst_im - dst_re;

  auto dst0 = dst_re; 
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
    DstCf::store(d0, dst0 + d, im_off);
    DstCf::store(d1, dst1 + d, im_off);
    DstCf::store(d2, dst2 + d, im_off);
    DstCf::store(d3, dst3 + d, im_off);
  }
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
void last_three_passes(
  Int n, const ET<V>* src, const ET<V>* tw, ET<V>* dst_re, ET<V>* dst_im)
{
  VEC_TYPEDEFS(V);

  Int im_off = dst_im - dst_re;

  Int l1 = n / 8 / V::vec_size;
  Int l2 = 2 * l1;
  Int l3 = 3 * l1;
  Int l4 = 4 * l1;
  Int l5 = 5 * l1;
  Int l6 = 6 * l1;
  Int l7 = 7 * l1;

  for(BitReversed br(l1); br.i < l1; br.advance())
  {
    auto d = dst_re + br.br * stride<V, DstCf>();
    auto s = src + 8 * stride<V, cf::Vec>() * br.i;
    auto this_tw = tw + 5 * stride<V, cf::Vec>() * br.i;

    C a0, a1, a2, a3, a4, a5, a6, a7;
    {
      C tw0 = load<V, cf::Vec>(this_tw, 0);
      C tw1 = load<V, cf::Vec>(this_tw + stride<V, cf::Vec>(), 0);
      C tw2 = load<V, cf::Vec>(this_tw + 2 * stride<V, cf::Vec>(), 0);

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
      C tw3 = load<V, cf::Vec>(this_tw + 3 * stride<V, cf::Vec>(), 0);
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
      C tw4 = load<V, cf::Vec>(this_tw + 4 * stride<V, cf::Vec>(), 0);
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

constexpr bool is_power_of_4(Int n)
{
  while(n >= 4) n /= 4;
  return n == 1;
}

template<typename V, Int vn, Int dft_sz, typename A>
FORCEINLINE void tiny_transform_pass(A& src_re, A& src_im, A& dst_re, A& dst_im)
{
  VEC_TYPEDEFS(V);
  constexpr Int vsz = V::vec_size;
  auto& table = CtSizedFftTwiddleTable<dft_sz, vsz, T>::value;

  if constexpr(dft_sz < V::vec_size)
  {
    for(Int i = 0; i < vn / 2; i++)
    {
      C a = { src_re[i], src_im[i] };
      C b = { src_re[i + vn / 2], src_im[i + vn / 2] };
      C t = { V::unaligned_load(table.re), V::unaligned_load(table.im) };
      if(dft_sz > 1) b = b * t;
      C dst_a = a + b;
      C dst_b = a - b;

      V::template interleave_multi<vsz / dft_sz>(
        dst_a.re, dst_b.re, dst_re[2 * i], dst_re[2 * i + 1]);

      V::template interleave_multi<vsz / dft_sz>(
        dst_a.im, dst_b.im, dst_im[2 * i], dst_im[2 * i + 1]);
    }
  }
  else
  {
    constexpr Int vdft_sz = dft_sz / vsz;

    for(Int i = 0; i < vn / 2; i += vdft_sz)
    {
      for(Int j = 0; j < vdft_sz; j++)
      {
        C src_a = { src_re[i + j], src_im[i + j] };
        C src_b = { src_re[i + j + vn / 2], src_im[i + j + vn / 2] };
        C t = {
          V::unaligned_load(table.re + j * vsz),
          V::unaligned_load(table.im + j * vsz) };

        C m = src_b * t;
        C dst_a = src_a + m;
        C dst_b = src_a - m;

        dst_re[2 * i + j] = dst_a.re;
        dst_im[2 * i + j] = dst_a.im;

        dst_re[2 * i + j + vdft_sz] = dst_b.re;
        dst_im[2 * i + j + vdft_sz] = dst_b.im;
      }
    }
  }
}

//One weird trick to prevent GCC from needlessly
//writing array elements to the stack.
template<typename Vec, int n> struct Locals
{
  Vec a[n];
  Vec& operator[](int i) { return a[i]; }
};

template<typename Vec> struct Locals<Vec, 1>
{
  Vec a0;
  Vec& operator[](int i) { return a0; }
};

template<typename Vec> struct Locals<Vec, 2>
{
  Vec a0, a1;
  Vec& operator[](int i) { return i == 0 ? a0 : a1; }
};

template<typename Vec> struct Locals<Vec, 4>
{
  Vec a0, a1, a2, a3;
  Vec& operator[](int i)
  {
    return 
      i == 0 ? a0 :
      i == 1 ? a1 :
      i == 2 ? a2 : a3;
  }
};

template<typename V, typename SrcCf, typename DstCf, Int n>
void tiny_transform(
  const Fft<typename V::T>* state,
  const typename V::T* src_re,
  const typename V::T* src_im,
  typename V::T* dst_re, 
  typename V::T* dst_im)
{
  VEC_TYPEDEFS(V);
  constexpr Int vsz = V::vec_size;

  //Round up just to make it compile
  constexpr Int vn = (n + V::vec_size - 1) / V::vec_size;

  Int src_im_off = src_im - src_re;
  Int dst_im_off = dst_im - dst_re;

  Locals<Vec, vn> a_re;
  Locals<Vec, vn> a_im;
  Locals<Vec, vn> b_re;
  Locals<Vec, vn> b_im;

  for(Int i = 0; i < vn; i++)
  {
    auto c = load<V, SrcCf>(src_re + i * stride<V, SrcCf>(), src_im_off);
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

    DstCf::store(c, dst_re + i * stride<V, DstCf>(), dst_im_off);
  }
}

template<bool do_create, typename V, typename SrcCf, typename DstCf>
Int tiny_fft_create_impl(Int n, void* ptr)
{
  VEC_TYPEDEFS(V);
  Fft<T> local_state;
  auto state = do_create ? (Fft<T>*) ptr : &local_state;

  ptr = aligned_increment(ptr, sizeof(Fft<T>));

  state->working = nullptr;
  state->twiddle = nullptr;
  state->n = n;
  state->transform_fun = 
    n ==  1 ?  &tiny_transform<V, SrcCf, DstCf,  1> :
    n ==  2 ?  &tiny_transform<V, SrcCf, DstCf,  2> :
    n ==  4 ?  &tiny_transform<V, SrcCf, DstCf,  4> :
    n ==  8 ?  &tiny_transform<V, SrcCf, DstCf,  8> :
    n == 16 ?  &tiny_transform<V, SrcCf, DstCf, 16> :
    n == 32 ?  &tiny_transform<V, SrcCf, DstCf, 32> : nullptr;

  return (Int) ptr;
}

template<typename V>
Int get_npasses(Int n, Int dft_size)
{
  VEC_TYPEDEFS(V);
  if(dft_size == 1)
  {
    if constexpr(V::vec_size == 8)
      return 3;
    else
      return 2;
  }
  else if((dft_size << 3) == n)
    return 3;
  else
    return 2;
}

template<typename V>
Int total_num_steps(Int n)
{
  Int r = 0;
  for(Int ds = 1; ds != n; ds <<= get_npasses<V>(n, ds)) r++;
  return r;
}

template<typename V, typename SrcCf, typename DstCf>
void small_transform(
  const Fft<typename V::T>* state,
  const typename V::T* src_re,
  const typename V::T* src_im,
  typename V::T* dst_re, 
  typename V::T* dst_im)
{
  VEC_TYPEDEFS(V);

  Int n = state->n;
  T* w = state->working;

  Arg<T> arg;
  arg.n = state->n;
  arg.im_off = src_im - src_re;
  arg.dft_size = 1;
  arg.start_offset = 0;
  arg.end_offset = state->n;
  arg.src = (T*) src_re;
  arg.dst = state->working;

  Int first_npasses = get_npasses<V>(arg.n, 1);
  arg.twiddle = state->twiddle[0];
  if(first_npasses == 2) first_two_passes<V, SrcCf>(n, src_re, src_im, w);
  else first_three_passes<V, SrcCf>(n, src_re, src_im, w);

  arg.dft_size <<= first_npasses;
  arg.src = state->working;
  arg.dst = state->working;

  for(Int i = 1;; i++)
  {
    Int npasses = get_npasses<V>(arg.n, arg.dft_size);
    Int next_dft_size = arg.dft_size << npasses; 
    arg.twiddle = state->twiddle[i];
    if(next_dft_size == arg.n)
    {
      arg.dst = dst_re;
      arg.im_off = dst_im - dst_re;
      if(npasses == 2)
        last_two_passes<V, DstCf>(n, w, state->twiddle[i], dst_re, dst_im);
      else
        last_three_passes<V, DstCf>(n, w, state->twiddle[i], dst_re, dst_im);

      break;
    }
    else
    {
      two_passes<V>(arg);
      arg.dft_size = next_dft_size;
    }
  }
}

template<typename V>
NOINLINE void recursive_passes(
  const Fft<typename V::T>* state,
  Int step, typename V::T* p, Int start, Int end)
{
  VEC_TYPEDEFS(V);
  Int dft_size = 1;
  for(Int i = 0; i < step; i++)
    dft_size <<= get_npasses<V>(state->n, dft_size);

  Int npasses = get_npasses<V>(state->n, dft_size);

  Arg<T> arg;
  arg.n = state->n;
  arg.im_off = 0;
  arg.dft_size = dft_size;
  arg.start_offset = start;
  arg.end_offset = end;
  arg.src = p;
  arg.dst = nullptr;
  arg.twiddle = state->twiddle[step];

  if(npasses == 3) last_three_passes_in_place<V>(arg);
  else two_passes<V>(arg);

  if((dft_size << npasses) < state->n)
  {
    if(end - start > optimal_size)
    {
      Int next_sz = (end - start) >> npasses;
      for(Int s = start; s < end; s += next_sz)
        recursive_passes<V>(state, step + 1, p, s, s + next_sz);
    }
    else
      recursive_passes<V>(state, step + 1, p, start, end);
  }
}


template<typename V, typename SrcCf, typename DstCf>
void large_transform(
  const Fft<typename V::T>* state,
  const typename V::T* src_re,
  const typename V::T* src_im,
  typename V::T* dst_re, 
  typename V::T* dst_im)
{
  VEC_TYPEDEFS(V);

  Int first_npasses = get_npasses<V>(state->n, 1);

  Int n = state->n;
  T* w = state->working;

  Arg<T> arg;
  arg.n = state->n;
  arg.im_off = src_im - src_re;
  arg.dft_size = 1;
  arg.start_offset = 0;
  arg.end_offset = state->n;
  arg.src = (T*) src_re;
  arg.dst = state->working;

  arg.twiddle = state->twiddle[0];
  if(first_npasses == 2) first_two_passes<V, SrcCf>(arg.n, src_re, src_im, w);
  else first_three_passes<V, SrcCf>(arg.n, src_re, src_im, w);

  recursive_passes<V>(state, 1, state->working, 0, state->n);

  arg.src = state->working;
  arg.dst = dst_re;
  arg.im_off = dst_im - dst_re;
  bit_reverse_pass<V, DstCf>(arg);
}


template<bool do_create, typename V, typename SrcCf, typename DstCf>
Int fft_create_impl(Int n, void* ptr)
{
  VEC_TYPEDEFS(V);
  if(n <= V::vec_size && V::vec_size != 1)
    return fft_create_impl<do_create, Scalar<T>, SrcCf, DstCf>(n, ptr);
  else if(n < max(V::vec_size * V::vec_size, Int(16)))
    return tiny_fft_create_impl<do_create, V, SrcCf, DstCf>(n, ptr);
  else
  {
    VEC_TYPEDEFS(V);
    Fft<T> local_state;
    
    auto state = do_create ? (Fft<T>*) ptr : &local_state;
    ptr = aligned_increment(ptr, sizeof(Fft<T>));

    state->n = n;
    state->transform_fun = n < large_fft_size ?
      &small_transform<V, SrcCf, DstCf> :  
      &large_transform<V, SrcCf, DstCf>;
   
    state->working = (T*) ptr;
    ptr = aligned_increment(ptr, sizeof(T) * 2 * n);

    state->twiddle = (T**) ptr;
    ptr = aligned_increment(ptr, total_num_steps<V>(n) * sizeof(T*));

    if(do_create)
      compute_twiddle_range<V>(n, state->working, state->working + n);

    for(Int i = 0, dft_size = 1; dft_size != n; i++)
    {
      Int npasses = get_npasses<V>(n, dft_size);

      T* tw = (T*) ptr;
      if(do_create) state->twiddle[i] = tw;
      ptr = aligned_increment(
        ptr, twiddle_for_step_memsize<V>(dft_size, npasses));

      if(do_create)
        twiddle_for_step_create<V>(state->working, n, dft_size, npasses, tw);

      dft_size <<= npasses;
    }

    return Int(ptr);
  }
}

template<typename V, typename SrcCf, typename DstCf>
Int fft_memsize(Int n)
{
  return fft_create_impl<false, V, SrcCf, DstCf>(n, nullptr);
}

template<typename V, typename SrcCf, typename DstCf>
Fft<typename V::T>* fft_create(Int n, void* ptr)
{
  fft_create_impl<true, V, SrcCf, DstCf>(n, ptr);
  return (Fft<typename V::T>*) ptr;
}

template<typename V, typename SrcCf, typename DstCf>
Int ifft_memsize(Int n)
{
  return fft_memsize<V, cf::Swapped<SrcCf>, cf::Swapped<DstCf>>(n);
}

template<typename V, typename SrcCf, typename DstCf>
Ifft<typename V::T>* ifft_create(Int n, void* ptr)
{
  return (Ifft<typename V::T>*) fft_create<
    V, cf::Swapped<SrcCf>, cf::Swapped<DstCf>>(n, ptr);
}

#if 0
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
  arg.twiddle = state->steps[step].twiddle;
  
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
FORCEINLINE void fft_impl(
  const Fft<T>* state,
  T* src, Int src_im_off,
  T* dst, Int dst_im_off)
{
  if(state->tiny_transform_fun)
  {
    state->tiny_transform_fun(src, src_im_off, dst, dst_im_off);
    return;
  }

  Arg<T> arg;
  arg.n = state->n;
  arg.im_off = src_im_off;
  arg.dft_size = 1;
  arg.start_offset = 0;
  arg.end_offset = state->n;
  arg.src = src;

  auto w0 = state->working;
 
  arg.src = src;
  arg.dst = w0;
  state->steps[0].fun_ptr(arg);
  arg.dft_size <<= state->steps[0].npasses;

  arg.src = w0;
  arg.dst = w0;
  for(Int step = 1; step < state->nsteps - 1; )
  {
    if(state->steps[step].is_recursive)
    {
      recursive_passes(state, step, arg.src, 0, state->n);
      while(step < state->nsteps && state->steps[step].is_recursive) step++;
    }
    else
    {
      arg.twiddle = state->steps[step].twiddle;
      state->steps[step].fun_ptr(arg);
      arg.dft_size <<= state->steps[step].npasses;
      step++;
    }
  }

  arg.dst = dst;
  arg.im_off = dst_im_off;
  arg.twiddle = state->steps[state->nsteps - 1].twiddle;
  state->steps[state->nsteps - 1].fun_ptr(arg);
}
#endif

template<typename T>
FORCEINLINE void fft_impl(
  const Fft<T>* state,
  T* src, Int src_im_off,
  T* dst, Int dst_im_off)
{
  state->transform_fun(state, src, src + src_im_off, dst, dst + dst_im_off);
}

template<typename T>
void fft(const Fft<T>* state, T* src, T* dst)
{
  fft_impl(state, src, state->n, dst, state->n);
}

template<typename T>
void ifft(const Ifft<T>* state, T* src, T* dst)
{
  Int n = ((Fft<T>*) state)->n;
  fft_impl((Fft<T>*) state, src, n, dst, n);
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

  //src and dst may be the same, so we need to store some values
  //here before they get overwritten

  SC src_start = load<S, SrcCf>(src, src_off);
  SC src_end = load<S, SrcCf>(src + n / 2 * src_ratio, src_off);
  SC middle = load<S, SrcCf>(src + n / 4 * src_ratio, src_off);

  for(
    Int i0 = 1, i1 = n / 2 - V::vec_size, iw = 0; 
    i0 <= i1; 
    i0 += V::vec_size, i1 -= V::vec_size, iw += V::vec_size)
  {
    C w = load<V, cf::Split>(twiddle + iw, n / 2);
    C s0 = SrcCf::template unaligned_load<V>(src + i0 * src_ratio, src_off);
    C s1 = reverse_complex<V>(load<V, SrcCf>(src + i1 * src_ratio, src_off));

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
    DstCf::template store<0, S>(
      {src_start.re + src_end.re, src_start.re - src_end.re},
      dst, dst_off);
  }
  else
  {
    DstCf::template store<0, S>({src_start.re + src_start.im, 0}, dst, dst_off);
    DstCf::template store<0, S>(
      {src_start.re - src_start.im, 0}, dst + n / 2 * dst_ratio, dst_off);
  }
}

template<typename T>
struct Rfft
{
  Fft<T>* state;
  T* working;
  T* twiddle;
  void (*real_pass)(Int, T*, Int, T*, T*, Int);
};

template<bool do_create, typename V, typename DstCf>
Int rfft_create_impl(Int n, void* ptr)
{
  VEC_TYPEDEFS(V);
  if(V::vec_size != 1 && n <= 2 * V::vec_size)
    return rfft_create_impl<do_create, Scalar<T>, DstCf>(n, ptr);

  Rfft<T>* r = (Rfft<T>*) ptr;
  ptr = aligned_increment(ptr, sizeof(Rfft<T>));

  if(SameType<DstCf, cf::Split>::value)
  {
    if(do_create) r->working = nullptr;
  }
  else
  {
    if(do_create) r->working = (T*) ptr;
    ptr = aligned_increment(ptr, sizeof(T) * n);
  }

  if(do_create) r->twiddle = (T*) ptr;
  ptr = aligned_increment(ptr, sizeof(T) * n);
 
  if(do_create)
  {
    r->real_pass = &real_pass<V, cf::Split, DstCf, false>;

    Int m =  n / 2;
    compute_twiddle<V>(m, r->twiddle, r->twiddle + m);
    copy(r->twiddle + 1, m - 1, r->twiddle);
    copy(r->twiddle + m + 1, m - 1, r->twiddle + m);

    r->state = fft_create<V, cf::Scal, cf::Split>(n / 2, ptr);
  }

  ptr = aligned_increment(ptr, fft_memsize<V, cf::Scal, cf::Split>(n / 2));

  return Int(ptr);
}

template<typename V, typename DstCf>
Int rfft_memsize(Int n)
{
  return rfft_create_impl<false, V, DstCf>(n, nullptr);
}

template<typename V, typename DstCf>
Rfft<typename V::T>* rfft_create(Int n, void* ptr)
{
  rfft_create_impl<true, V, DstCf>(n, ptr);
  return (Rfft<typename V::T>*) ptr;
}

template<typename T>
void rfft(const Rfft<T>* state, T* src, T* dst)
{
  Int n = state->state->n;
  Int dst_im_off = align_size<T>(n + 1);

  T* w = state->working ? state->working : dst;
  Int w_im_off = state->working ? n : dst_im_off;

  fft_impl(state->state, src, n, w, w_im_off);

  state->real_pass(
    n * 2,
    w, w_im_off, 
    state->twiddle,
    dst, dst_im_off);
}

template<typename T>
struct Irfft
{
  Ifft<T>* state;
  T* twiddle;
  void (*real_pass)(Int, T*, Int, T*, T*, Int);
};

template<bool do_create, typename V, typename SrcCf>
Int irfft_create_impl(Int n, void* ptr)
{
  VEC_TYPEDEFS(V);
  if(V::vec_size != 1 && n <= 2 * V::vec_size)
    return irfft_create_impl<do_create, Scalar<T>, SrcCf>(n, ptr);

  auto r = (Irfft<T>*) ptr;
  ptr = aligned_increment(ptr, sizeof(Irfft<T>));

  if(do_create) r->twiddle = (T*) ptr;
  ptr = aligned_increment(ptr, sizeof(T) * n);

  if(do_create)
  {
    r->real_pass = &real_pass<V, SrcCf, cf::Split, true>;

    Int m =  n / 2;
    compute_twiddle<V>(m, r->twiddle, r->twiddle + m);
    copy(r->twiddle + 1, m - 1, r->twiddle);
    copy(r->twiddle + m + 1, m - 1, r->twiddle + m);

    r->state = ifft_create<V, cf::Split, cf::Scal>(n / 2, ptr);
  }

  ptr = aligned_increment(ptr, ifft_memsize<V, cf::Split, cf::Scal>(n / 2));

  return Int(ptr);
}

template<typename V, typename SrcCf>
Int irfft_memsize(Int n)
{
  return irfft_create_impl<false, V, SrcCf>(n, nullptr);
}

template<typename V, typename SrcCf>
Irfft<typename V::T>* irfft_create(Int n, void* ptr)
{
  irfft_create_impl<true, V, SrcCf>(n, ptr);
  return (Irfft<typename V::T>*) ptr;
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
    dst,
    complex_state->n);

  ifft(state->state, dst, dst);
}
}
#endif
