#ifndef FFT_ONEDIM_H
#define FFT_ONEDIM_H

#include "common.hpp"

namespace
{
namespace onedim
{
template<typename T>
struct Step
{
  Int npasses;
  T* twiddle;
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
  Step<T>* steps;
};

template<typename T> struct Ifft;

template<int len, typename T>
struct ReImTable
{
  T re[len];
  T im[len];
};

template<typename T>
constexpr Complex<Scalar<T>> root_of_unity(Int i, Int n)
{
  using C = Complex<Scalar<T>>;
  C r{1, 0};
  int table_i = 0;
  for(int bit = n / 2; bit > 0; bit >>= 1, table_i++)
    if((i & bit) != 0)
      r = r * C{
        SinCosTable<T>::cos[table_i],
        SinCosTable<T>::sin[table_i]};

  return r;
}

template<typename V>
FORCEINLINE void two_passes_inner_unity_twiddle(
    Complex<V> src0, Complex<V> src1, Complex<V> src2, Complex<V> src3,
    Complex<V>& dst0, Complex<V>& dst1, Complex<V>& dst2, Complex<V>& dst3)
{
  typedef Complex<V> C;
  C sum02 = src0 + src2;
  C dif02 = src0 - src2;
  C sum13 = src1 + src3;
  C dif13 = src1 - src3;

  dst0 = sum02 + sum13;
  dst2 = sum02 - sum13;
  dst1 = dif02 + dif13.mul_neg_i();
  dst3 = dif02 - dif13.mul_neg_i();
}

template<typename V>
FORCEINLINE void two_passes_inner(
    Complex<V> src0, Complex<V> src1, Complex<V> src2, Complex<V> src3,
    Complex<V>& dst0, Complex<V>& dst1, Complex<V>& dst2, Complex<V>& dst3,
    Complex<V> tw0, Complex<V> tw1, Complex<V> tw2)
{
  two_passes_inner_unity_twiddle<V>(
    src0, tw0 * src1, tw1 * src2, tw2 * src3,
    dst0, dst1, dst2, dst3);
}

template<typename V, typename DstCf, bool br_dst>
FORCEINLINE void three_passes_inner(
  const ET<V>* src, Int src_stride,
  Complex<V> tw0,
  Complex<V> tw1,
  Complex<V> tw2,
  Complex<V> tw3,
  Complex<V> tw4,
  ET<V>* dst_re, ET<V>* dst_im, Int dst_stride)
{
  VEC_TYPEDEFS(V);

  C a0, a1, a2, a3, a4, a5, a6, a7;
  {
    two_passes_inner(
      C::load(src + 0 * src_stride), C::load(src + 2 * src_stride),
      C::load(src + 4 * src_stride), C::load(src + 6 * src_stride),
      a0, a1, a2, a3,
      tw0, tw1, tw2);

    two_passes_inner(
      C::load(src + 1 * src_stride), C::load(src + 3 * src_stride),
      C::load(src + 5 * src_stride), C::load(src + 7 * src_stride),
      a4, a5, a6, a7,
      tw0, tw1, tw2);
  }

  {
    auto mul = tw3 * a4;
    store<DstCf>(a0 + mul, dst_re, dst_im, (br_dst ? 0 : 0) * dst_stride);
    store<DstCf>(a0 - mul, dst_re, dst_im, (br_dst ? 1 : 4) * dst_stride);
  }

  {
    auto mul = tw3.mul_neg_i() * a6;
    store<DstCf>(a2 + mul, dst_re, dst_im, (br_dst ? 2 : 2) * dst_stride);
    store<DstCf>(a2 - mul, dst_re, dst_im, (br_dst ? 3 : 6) * dst_stride);
  }

  {
    auto mul = tw4 * a5;
    store<DstCf>(a1 + mul, dst_re, dst_im, (br_dst ? 4 : 1) * dst_stride);
    store<DstCf>(a1 - mul, dst_re, dst_im, (br_dst ? 5 : 5) * dst_stride);
  }

  {
    auto mul = tw4.mul_neg_i() * a7;
    store<DstCf>(a3 + mul, dst_re, dst_im, (br_dst ? 6 : 3) * dst_stride);
    store<DstCf>(a3 - mul, dst_re, dst_im, (br_dst ? 7 : 7) * dst_stride);
  }
}

template<typename V, typename SrcCf>
void first_two_passes(
  Int n, const ET<V>* src_re, const ET<V>* src_im, ET<V>* dst)
{
  VEC_TYPEDEFS(V);
  Int l = n * SrcCf::idx_ratio / 4;
  Int dst_l = n * cf::Vec::idx_ratio / 4;

  for(const T* end = src_re + l; src_re < end;)
  {
    C c0, c1, c2, c3;
    two_passes_inner_unity_twiddle(
      load<V, SrcCf>(src_re, src_im, 0 * l),
      load<V, SrcCf>(src_re, src_im, 1 * l),
      load<V, SrcCf>(src_re, src_im, 2 * l),
      load<V, SrcCf>(src_re, src_im, 3 * l),
      c0, c1, c2, c3);

    src_re += stride<V, SrcCf>();
    src_im += stride<V, SrcCf>();

    if constexpr(V::vec_size == 1)
    {
      c0.store(dst + 0 * dst_l);
      c2.store(dst + 1 * dst_l);
      c1.store(dst + 2 * dst_l);
      c3.store(dst + 3 * dst_l);
      dst += stride<V, cf::Vec>();
    }
    else if constexpr(V::vec_size == 2)
    {
      C d0, d1;
      V::interleave(c0.re, c1.re, d0.re, d1.re);
      V::interleave(c0.im, c1.im, d0.im, d1.im);
      d0.store(dst + 0 * dst_l);
      d1.store(dst + 0 * dst_l + stride<V, cf::Vec>());

      C d2, d3;
      V::interleave(c2.re, c3.re, d2.re, d3.re);
      V::interleave(c2.im, c3.im, d2.im, d3.im);
      d2.store(dst + 2 * dst_l);
      d3.store(dst + 2 * dst_l + stride<V, cf::Vec>());

      dst += 2 * stride<V, cf::Vec>();
    }
    else
    {
      V::template transposed_store<stride<V, cf::Vec>()>(
        c0.re, c1.re, c2.re, c3.re, dst);

      V::template transposed_store<stride<V, cf::Vec>()>(
        c0.im, c1.im, c2.im, c3.im, dst + V::vec_size);

      dst += 4 * stride<V, cf::Vec>();
    }
  }
}

template<typename V, typename SrcCf>
void first_three_passes(
  Int n, const ET<V>* src_re, const ET<V>* src_im, ET<V>* dst)
{
  VEC_TYPEDEFS(V);
  Int l = n / 8 * SrcCf::idx_ratio;
  Vec invsqrt2 = V::vec(SinCosTable<T>::cos[2]);

  for(T* end = dst + n * cf::Vec::idx_ratio; dst < end;)
  {
    C a0, a1, a2, a3;
    two_passes_inner_unity_twiddle(
      load<V, SrcCf>(src_re, src_im, 0 * l),
      load<V, SrcCf>(src_re, src_im, 2 * l),
      load<V, SrcCf>(src_re, src_im, 4 * l),
      load<V, SrcCf>(src_re, src_im, 6 * l),
      a0, a1, a2, a3);

    C mul0, mul1, mul2, mul3;
    {
      C b0, b1, b2, b3;
      two_passes_inner_unity_twiddle(
        load<V, SrcCf>(src_re, src_im, 1 * l),
        load<V, SrcCf>(src_re, src_im, 3 * l),
        load<V, SrcCf>(src_re, src_im, 5 * l),
        load<V, SrcCf>(src_re, src_im, 7 * l),
        b0, b1, b2, b3);

      mul0 = b0;
      mul1 = {invsqrt2 * (b1.re + b1.im), invsqrt2 * (b1.im - b1.re)};
      mul2 = b2.mul_neg_i();
      mul3 = {invsqrt2 * (b3.im - b3.re), invsqrt2 * (-b3.im - b3.re)};
    }

    src_re += stride<V, SrcCf>();
    src_im += stride<V, SrcCf>();

    V::template transposed_store<stride<V, cf::Vec>()>(
      a0.re + mul0.re, a1.re + mul1.re, a2.re + mul2.re, a3.re + mul3.re,
      a0.re - mul0.re, a1.re - mul1.re, a2.re - mul2.re, a3.re - mul3.re,
      dst);

    V::template transposed_store<stride<V, cf::Vec>()>(
      a0.im + mul0.im, a1.im + mul1.im, a2.im + mul2.im, a3.im + mul3.im,
      a0.im - mul0.im, a1.im - mul1.im, a2.im - mul2.im, a3.im - mul3.im,
      dst + V::vec_size);

    dst += 8 * stride<V, cf::Vec>();
  }
}

template<typename V, int i>
FORCEINLINE void first_four_passes_helper(Complex<V> (&interm)[16])
{
  VEC_TYPEDEFS(V);
  constexpr auto tw0 = root_of_unity<T>(i, 16).adj();
  constexpr auto tw1 = tw0 * tw0;
  constexpr auto tw2 = tw1 * tw0;
  two_passes_inner(
    interm[i], interm[i + 4], interm[i + 8], interm[i + 12],
    interm[i], interm[i + 4], interm[i + 8], interm[i + 12],
    C{ V::vec(tw0.re), V::vec(tw0.im) },
    C{ V::vec(tw1.re), V::vec(tw1.im) },
    C{ V::vec(tw2.re), V::vec(tw2.im) });
}

template<typename V, typename SrcCf>
void first_four_passes(
  Int n, const ET<V>* src_re, const ET<V>* src_im, ET<V>* dst)
{
  VEC_TYPEDEFS(V);
  constexpr Int m = 16;
  Int l = n / m * SrcCf::idx_ratio;

  for(T* end = dst + n * cf::Vec::idx_ratio; dst < end;)
  {
    C interm[m];

    for(Int i = 0; i < 4; i++)
      two_passes_inner_unity_twiddle(
        load<V, SrcCf>(src_re, src_im, (0 + i) * l),
        load<V, SrcCf>(src_re, src_im, (4 + i) * l),
        load<V, SrcCf>(src_re, src_im, (8 + i) * l),
        load<V, SrcCf>(src_re, src_im, (12 + i) * l),
        interm[4 * i + 0], interm[4 * i + 1],
        interm[4 * i + 2], interm[4 * i + 3]);

    src_re += stride<V, SrcCf>();
    src_im += stride<V, SrcCf>();

    first_four_passes_helper<V, 0>(interm);
    first_four_passes_helper<V, 1>(interm);
    first_four_passes_helper<V, 2>(interm);
    first_four_passes_helper<V, 3>(interm);

    Vec real[m];
    for(Int i = 0; i < m; i++) real[i] = interm[i].re;
    V::template transposed_store<stride<V, cf::Vec>()>(real, dst);

    Vec imag[m];
    for(Int i = 0; i < m; i++) imag[i] = interm[i].im;
    V::template transposed_store<stride<V, cf::Vec>()>(imag, dst + V::vec_size);

    dst += m * stride<V, cf::Vec>();
  }
}

template<typename T>
const T* two_pass_twiddle_ptr(const T* tw, Int n, Int offset, Int dft_size)
{
  return tw + ((3 * offset * cf::Vec::idx_ratio * dft_size) >> log2(n));
} 

template<typename T>
const T* three_pass_twiddle_ptr(const T* tw, Int n, Int offset, Int dft_size)
{
  return tw + ((5 * offset * cf::Vec::idx_ratio * dft_size) >> log2(n));
} 

template<typename V>
void two_passes(
  Int n, Int dft_size, ET<V>* data_ptr, Int data_n, const ET<V>* tw)
{
  VEC_TYPEDEFS(V);

  auto off1 = (n >> log2(dft_size)) / 4 * stride<V, cf::Vec>();
  auto off2 = off1 + off1;
  auto off3 = off2 + off1;

  for(
    auto p = data_ptr, end = data_ptr + data_n * cf::Vec::idx_ratio;
    p < end;)
  {
    auto tw0 = C::load(tw);
    auto tw1 = C::load(tw + stride<V, cf::Vec>());
    auto tw2 = C::load(tw + 2 * stride<V, cf::Vec>());
    tw += 3 * stride<V, cf::Vec>();

    for(auto end1 = p + off1;;)
    {
      ASSERT(p >= data_ptr);
      ASSERT(p + off3 < data_ptr + data_n * cf::Vec::idx_ratio);

      C d0, d1, d2, d3;
      two_passes_inner(
        C::load(p), C::load(p + off1),
        C::load(p + off2), C::load(p + off3),
        d0, d1, d2, d3, tw0, tw1, tw2);

      d0.store(p);
      d2.store(p + off1);
      d1.store(p + off2);
      d3.store(p + off3);

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

  auto dst0_re = dst_re; 
  auto dst1_re = dst0_re + vn / 4 * stride<V, DstCf>(); 
  auto dst2_re = dst1_re + vn / 4 * stride<V, DstCf>(); 
  auto dst3_re = dst2_re + vn / 4 * stride<V, DstCf>(); 
  
  auto dst0_im = dst_im; 
  auto dst1_im = dst0_im + vn / 4 * stride<V, DstCf>(); 
  auto dst2_im = dst1_im + vn / 4 * stride<V, DstCf>(); 
  auto dst3_im = dst2_im + vn / 4 * stride<V, DstCf>(); 

  for(BitReversed br(vn / 4); br.i < vn / 4; br.advance())
  {
    auto tw0 = C::load(tw);
    auto tw1 = C::load(tw + stride<V, cf::Vec>());
    auto tw2 = C::load(tw + 2 * stride<V, cf::Vec>());
    tw += 3 * stride<V, cf::Vec>();

    C d0, d1, d2, d3;
    two_passes_inner(
      C::load(src),
      C::load(src + stride<V, cf::Vec>()),
      C::load(src + 2 * stride<V, cf::Vec>()),
      C::load(src + 3 * stride<V, cf::Vec>()),
      d0, d1, d2, d3, tw0, tw1, tw2);

    src += 4 * stride<V, cf::Vec>();

    Int d = br.br * stride<V, DstCf>();
    store<DstCf>(d0, dst0_re, dst0_im, d);
    store<DstCf>(d1, dst1_re, dst1_im, d);
    store<DstCf>(d2, dst2_re, dst2_im, d);
    store<DstCf>(d3, dst3_re, dst3_im, d);
  }
}

template<typename V, typename DstCf>
void bit_reverse_pass(Int n, const ET<V>* src, ET<V>* dst_re, ET<V>* dst_im)
{
  VEC_TYPEDEFS(V);

  Int vn = n / V::vec_size;
  //const Int br_table[] = {0, 2, 1, 3};
  constexpr Int br_table[] = {0, 4, 2, 6, 1, 5, 3, 7};
  //constexpr Int br_table[] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
  constexpr int m = sizeof(br_table) / sizeof(br_table[0]);

  if(vn < m * m)
  {
    for(BitReversed br(vn); br.i < vn; br.advance())
      store<DstCf, stream_flag>(
        C::load(src + br.i * stride<V, cf::Vec>()),
        dst_re, dst_im, 
        br.br * stride<V, DstCf>());
  }
  else
  {
    Int stride_ = vn / m;

    for(BitReversed br(vn / (m * m)); br.i < vn / (m * m); br.advance())
    {
      const T* s_outer = src + br.i * m * stride<V, cf::Vec>();
      T* d_outer_re = dst_re + br.br * m * stride<V, DstCf>();
      T* d_outer_im = dst_im + br.br * m * stride<V, DstCf>();

      for(Int i0 = 0; i0 < m; i0++)
      {
        const T* s = s_outer + br_table[i0] * stride<V, cf::Vec>();
        T* d_re = d_outer_re + i0 * stride_ * stride<V, DstCf>();
        T* d_im = d_outer_im + i0 * stride_ * stride<V, DstCf>();
        for(Int i1 = 0; i1 < m; i1++)
        {
          auto this_s = s + br_table[i1] * stride_ * stride<V, cf::Vec>();
          store<DstCf, stream_flag>(
            C::load(this_s),
            d_re, d_im,
            i1 * stride<V, DstCf>());
        }
      }
    }
  }

  V::sfence();
}

template<typename V>
void three_passes(
  Int n, Int dft_size, ET<V>* data_ptr, Int data_n, const ET<V>* tw)
{
  VEC_TYPEDEFS(V);

  auto off1 = (n >> log2(dft_size)) / 8 * stride<V, cf::Vec>();
  auto gap = 7 * off1;

  for(
    auto p = data_ptr, end = data_ptr + data_n * cf::Vec::idx_ratio;
    p < end;)
  {
    auto tw0 = C::load(tw);
    auto tw1 = C::load(tw + stride<V, cf::Vec>());
    auto tw2 = C::load(tw + 2 * stride<V, cf::Vec>());
    auto tw3 = C::load(tw + 3 * stride<V, cf::Vec>());
    auto tw4 = C::load(tw + 4 * stride<V, cf::Vec>());
    tw += 5 * stride<V, cf::Vec>();

    for(auto end1 = p + off1;;)
    {
      ASSERT(p >= data_ptr);
      ASSERT(p + gap < data_ptr + data_n * cf::Vec::idx_ratio);

      C d0, d1, d2, d3;
      three_passes_inner<V, cf::Vec, true>(
        p, off1, tw0, tw1, tw2, tw3, tw4, p, nullptr, off1);

      p += stride<V, cf::Vec>();
      if(!(p < end1)) break;
    }

    p += gap;
  }
}

template<typename V, typename DstCf>
void last_three_passes(
  Int n, const ET<V>* src, const ET<V>* tw, ET<V>* dst_re, ET<V>* dst_im)
{
  VEC_TYPEDEFS(V);

  Int m = n / 8 / V::vec_size;
  for(BitReversed br(m); br.i < m; br.advance())
  {
    auto this_tw = tw + 5 * stride<V, cf::Vec>() * br.i;
    C tw0 = C::load(this_tw);
    C tw1 = C::load(this_tw + stride<V, cf::Vec>());
    C tw2 = C::load(this_tw + 2 * stride<V, cf::Vec>());
    C tw3 = C::load(this_tw + 3 * stride<V, cf::Vec>());
    C tw4 = C::load(this_tw + 4 * stride<V, cf::Vec>());
    three_passes_inner<V, DstCf, false>(
      src + 8 * stride<V, cf::Vec>() * br.i, stride<V, cf::Vec>(),
      tw0, tw1, tw2, tw3, tw4,
      dst_re + br.br * stride<V, DstCf>(),
      dst_im + br.br * stride<V, DstCf>(),
      m * stride<V, DstCf>());
  }
}

template<int n, int vsz, typename T>
constexpr ReImTable<(n > vsz ? n : vsz), T>
create_ct_sized_fft_twiddle_table()
{
  constexpr int len = n > vsz ? n : vsz;
  ReImTable<len, T> r = {0};
  for(int i = 0; i < n; i++)
  {
    auto tmp = root_of_unity<T>(i, n * 2);
    r.re[i] = tmp.re;
    r.im[i] = -tmp.im;
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
  static constexpr ReImTable<(n > vsz ? n : vsz), T> value =
    create_ct_sized_fft_twiddle_table<n, vsz, T>();
};

template<int n, int vsz, typename T>
constexpr ReImTable<(n > vsz ? n : vsz), T>
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
  FORCEINLINE Vec& operator[](int i) { return a[i]; }
};

template<typename Vec> struct Locals<Vec, 1>
{
  Vec a0;
  FORCEINLINE Vec& operator[](int i) { return a0; }
};

template<typename Vec> struct Locals<Vec, 2>
{
  Vec a0, a1;
  FORCEINLINE Vec& operator[](int i) { return i == 0 ? a0 : a1; }
};

template<typename Vec> struct Locals<Vec, 4>
{
  Vec a0, a1, a2, a3;
  FORCEINLINE Vec& operator[](int i)
  {
    return 
      i == 0 ? a0 :
      i == 1 ? a1 :
      i == 2 ? a2 : a3;
  }
};

template<typename Vec> struct Locals<Vec, 8>
{
  Vec a0, a1, a2, a3, a4, a5, a6, a7;
  FORCEINLINE Vec& operator[](int i)
  {
    return 
      i == 0 ? a0 :
      i == 1 ? a1 :
      i == 2 ? a2 :
      i == 3 ? a3 :
      i == 4 ? a4 :
      i == 5 ? a5 :
      i == 6 ? a6 : a7;
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
  
  //Round up just to make it compile
  constexpr Int vn = (n + V::vec_size - 1) / V::vec_size;

  Locals<Vec, vn> a_re;
  Locals<Vec, vn> a_im;
  Locals<Vec, vn> b_re;
  Locals<Vec, vn> b_im;

  for(Int i = 0; i < vn; i++)
  {
    auto c = load<V, SrcCf>(src_re, src_im, i * stride<V, SrcCf>());
    a_re[i] = c.re;
    a_im[i] = c.im;
  }

  if(n >  1) tiny_transform_pass<V, vn,  1>(a_re, a_im, b_re, b_im);
  if(n >  2) tiny_transform_pass<V, vn,  2>(b_re, b_im, a_re, a_im);
  if(n >  4) tiny_transform_pass<V, vn,  4>(a_re, a_im, b_re, b_im);
  if(n >  8) tiny_transform_pass<V, vn,  8>(b_re, b_im, a_re, a_im);
  if(n > 16) tiny_transform_pass<V, vn, 16>(a_re, a_im, b_re, b_im);
  if(n > 32) tiny_transform_pass<V, vn, 32>(b_re, b_im, a_re, a_im);
  if(n > 64) tiny_transform_pass<V, vn, 64>(a_re, a_im, b_re, b_im);

  for(Int i = 0; i < vn; i++)
  {
    C c;
    if constexpr(is_power_of_4(n)) c = { a_re[i], a_im[i] };
    else c = { b_re[i], b_im[i] };

    store<DstCf>(c, dst_re, dst_im, i * stride<V, DstCf>());
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
  state->steps = nullptr;
  state->n = n;
  state->transform_fun = 
    n ==  1 ?  &tiny_transform<V, SrcCf, DstCf,  1> :
    n ==  2 ?  &tiny_transform<V, SrcCf, DstCf,  2> :
    n ==  4 ?  &tiny_transform<V, SrcCf, DstCf,  4> :
    n ==  8 ?  &tiny_transform<V, SrcCf, DstCf,  8> :
    n == 16 ?  &tiny_transform<V, SrcCf, DstCf, 16> :
    n == 32 ?  &tiny_transform<V, SrcCf, DstCf, 32> :
    n == 64 ?  &tiny_transform<V, SrcCf, DstCf, 64> :
    n == 128 ?  &tiny_transform<V, SrcCf, DstCf, 128> : nullptr;

  return (Int) ptr;
}

template<typename V> 
constexpr Int get_first_npasses()
{
  if constexpr(V::vec_size == 16)
    return 4;
  if constexpr(V::vec_size == 8)
    return 3;
  else
    return 2;
}

template<typename V>
Int get_npasses(Int n, Int dft_size)
{
  VEC_TYPEDEFS(V);
  if(dft_size == 1)
    return get_first_npasses<V>();
  else
  {
    if constexpr(V::prefer_three_passes)
    {
      if((log2(n) - log2(dft_size)) % 3 == 0)
        return 3;
      else
        return 2;
    }
    else
    {
      if((dft_size << 3) == n)
        return 3;
      else
        return 2;
    }
  }
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
  Int dft_size = 1;

  constexpr Int first_npasses = get_first_npasses<V>();
  if constexpr(first_npasses == 2)
    first_two_passes<V, SrcCf>(n, src_re, src_im, w);
  else if constexpr(first_npasses == 3)
    first_three_passes<V, SrcCf>(n, src_re, src_im, w);
  else
    first_four_passes<V, SrcCf>(n, src_re, src_im, w);

  dft_size <<= first_npasses;

  for(Int i = 1;; i++)
  {
    Int npasses = state->steps[i].npasses;

    Int next_dft_size = dft_size << npasses; 
    if(next_dft_size == n)
    {
      if(npasses == 2)
        last_two_passes<V, DstCf>(
          n, w, state->steps[i].twiddle, dst_re, dst_im);
      else
        last_three_passes<V, DstCf>(
          n, w, state->steps[i].twiddle, dst_re, dst_im);

      break;
    }
    else
    {
      if(V::prefer_three_passes && npasses == 3)
        three_passes<V>(n, dft_size, w, n, state->steps[i].twiddle);
      else
        two_passes<V>(n, dft_size, w, n, state->steps[i].twiddle);

      dft_size = next_dft_size;
    }
  }
}

template<typename V>
NOINLINE void last_recursive_passes(
  Int n, Int dft_size, ET<V>* p, Int start, Int end, Step<ET<V>>* steps)
{
  VEC_TYPEDEFS(V);
  
  for(; dft_size < n;)
  {
    Int npasses = steps[0].npasses;

    if(npasses == 3)
      three_passes<V>(
        n, dft_size, p + start * cf::Vec::idx_ratio, end - start,
        three_pass_twiddle_ptr(steps[0].twiddle, n, start, dft_size));
    else
      two_passes<V>(
        n, dft_size, p + start * cf::Vec::idx_ratio, end - start,
        two_pass_twiddle_ptr(steps[0].twiddle, n, start, dft_size));

    dft_size <<= npasses;
    steps++;
  }
}

template<typename V>
NOINLINE void recursive_passes(
  Int n, Int dft_size, ET<V>* p, Int start, Int end, Step<ET<V>>* steps)
{
  VEC_TYPEDEFS(V);
  Int npasses = steps[0].npasses;

  if(V::prefer_three_passes && npasses == 3)
    three_passes<V>(
      n, dft_size, p + start * cf::Vec::idx_ratio, end - start,
      three_pass_twiddle_ptr(steps[0].twiddle, n, start, dft_size));
  else
    two_passes<V>(
      n, dft_size, p + start * cf::Vec::idx_ratio, end - start,
      two_pass_twiddle_ptr(steps[0].twiddle, n, start, dft_size));

  if(end - start > optimal_size)
  {
    Int next_sz = (end - start) >> npasses;
    for(Int s = start; s < end; s += next_sz)
      recursive_passes<V>(n, dft_size << npasses, p, s, s + next_sz, steps + 1);
  }
  else
    last_recursive_passes<V>(n, dft_size << npasses, p, start, end, steps + 1);
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

  constexpr Int first_npasses = get_first_npasses<V>();

  Int n = state->n;
  T* w = state->working;

  if constexpr(first_npasses == 2)
    first_two_passes<V, SrcCf>(n, src_re, src_im, w);
  else if constexpr(first_npasses == 3)
    first_three_passes<V, SrcCf>(n, src_re, src_im, w);
  else
    first_four_passes<V, SrcCf>(n, src_re, src_im, w);

  recursive_passes<V>(n, Int(1) << first_npasses, w, 0, n, state->steps + 1);

  bit_reverse_pass<V, DstCf>(n, w, dst_re, dst_im);
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

    state->steps = (Step<ET<V>>*) ptr;
    ptr = aligned_increment(ptr, total_num_steps<V>(n) * sizeof(Step<ET<V>>));

    if(do_create)
      compute_twiddle_range<V>(n, state->working, state->working + n);

    for(Int i = 0, dft_size = 1; dft_size != n; i++)
    {
      Int npasses = get_npasses<V>(n, dft_size);

      T* tw = (T*) ptr;
      if(do_create)
      {
        state->steps[i].twiddle = tw;
        state->steps[i].npasses = npasses;
      }

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

template<typename T>
void fft(
  const Fft<T>* state, const T* src_re, const T* src_im, T* dst_re, T* dst_im)
{
  state->transform_fun(state, src_re, src_im, dst_re, dst_im);
}

template<typename T>
void ifft(
  const Ifft<T>* state, const T* src_re, const T* src_im, T* dst_re, T* dst_im)
{
  fft((const Fft<T>*) state, src_re, src_im, dst_re, dst_im);
}

template<
  typename V,
  typename SrcCf,
  typename DstCf,
  bool inverse>
void real_pass(
  Int n,
  const ET<V>* src_re,
  const ET<V>* src_im,
  const ET<V>* twiddle,
  typename V::T* dst_re,
  typename V::T* dst_im)
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

  SC src_start = load<S, SrcCf>(src_re, src_im);
  SC src_end = load<S, SrcCf>(src_re, src_im,  n / 2 * src_ratio);
  SC middle = load<S, SrcCf>(src_re, src_im, n / 4 * src_ratio);

  for(
    Int i0 = 1, i1 = n / 2 - V::vec_size, iw = 0; 
    i0 <= i1; 
    i0 += V::vec_size, i1 -= V::vec_size, iw += V::vec_size)
  {
    C w = load<V, cf::Split>(twiddle, twiddle + n / 2, iw);
    C s0 = unaligned_load<V, SrcCf>(src_re, src_im, i0 * src_ratio);
    C s1 = reverse_complex<V>(load<V, SrcCf>(src_re, src_im, i1 * src_ratio));

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

    unaligned_store<DstCf>(d0, dst_re, dst_im, i0 * dst_ratio);
    store<DstCf>(reverse_complex<V>(d1), dst_re, dst_im, i1 * dst_ratio);
  }

  // fixes the aliasing bug
  store<DstCf>(
    middle.adj() * (inverse ? 2.0f : 1.0f), dst_re, dst_im, n / 4 * dst_ratio);

  if(inverse)
  {
    store<DstCf, 0, S>(
      {src_start.re + src_end.re, src_start.re - src_end.re},
      dst_re, dst_im);
  }
  else
  {
    store<DstCf, 0, S>({src_start.re + src_start.im, 0}, dst_re, dst_im);
    store<DstCf, 0, S>(
      {src_start.re - src_start.im, 0}, dst_re, dst_im, n / 2 * dst_ratio);
  }
}

template<typename T>
struct Rfft
{
  Fft<T>* state;
  T* working;
  T* twiddle;
  void (*real_pass)(Int, const T*, const T*, const T*, T*, T*);
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
    copy<V>(r->twiddle + 1, m - 1, r->twiddle);
    copy<V>(r->twiddle + m + 1, m - 1, r->twiddle + m);

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
void rfft(const Rfft<T>* state, const T* src, T* dst_re, T* dst_im)
{
  Int n = state->state->n;
  T* w_re = state->working ? state->working : dst_re;
  T* w_im = state->working ? state->working + n : dst_im;

  fft(state->state, src, src + n, w_re, w_im);

  state->real_pass(
    n * 2,
    w_re, w_im,
    state->twiddle,
    dst_re, dst_im);
}

template<typename T>
struct Irfft
{
  Ifft<T>* state;
  T* twiddle;
  void (*real_pass)(Int, const T*, const T*, const T*, T*, T*);
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
    copy<V>(r->twiddle + 1, m - 1, r->twiddle);
    copy<V>(r->twiddle + m + 1, m - 1, r->twiddle + m);

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
void irfft(const Irfft<T>* state, const T* src_re, const T* src_im, T* dst)
{
  auto complex_state = ((Fft<T>*) state->state);
  state->real_pass(
    complex_state->n * 2,
    src_re, src_im,
    state->twiddle,
    dst,
    dst + complex_state->n);

  ifft(state->state, dst, dst + complex_state->n, dst, dst + complex_state->n);
}

}
}
#endif
