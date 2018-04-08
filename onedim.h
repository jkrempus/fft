#ifndef FFT_ONEDIM_H
#define FFT_ONEDIM_H

#include "common.h"

namespace onedim
{
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

  for(const T* end = src_re + l; src_re < end;)
  {
    C a0 = load<V, SrcCf>(src_re, src_im, 0 * l);
    C a1 = load<V, SrcCf>(src_re, src_im, 1 * l);
    C a2 = load<V, SrcCf>(src_re, src_im, 2 * l);
    C a3 = load<V, SrcCf>(src_re, src_im, 3 * l);
    src_re += stride<V, SrcCf>();
    src_im += stride<V, SrcCf>();

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

    d0.store(dst); 
    d1.store(dst + stride<V, cf::Vec>()); 
    d2.store(dst + 2 * stride<V, cf::Vec>()); 
    d3.store(dst + 3 * stride<V, cf::Vec>()); 
    dst += 4 * stride<V, cf::Vec>();
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
    C c0, c1, c2, c3;
    {
      C a0 = load<V, SrcCf>(src_re, src_im, 0 * l);
      C a1 = load<V, SrcCf>(src_re, src_im, 2 * l);
      C a2 = load<V, SrcCf>(src_re, src_im, 4 * l);
      C a3 = load<V, SrcCf>(src_re, src_im, 6 * l);
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
      C a0 = load<V, SrcCf>(src_re, src_im, 1 * l);
      C a1 = load<V, SrcCf>(src_re, src_im, 3 * l);
      C a2 = load<V, SrcCf>(src_re, src_im, 5 * l);
      C a3 = load<V, SrcCf>(src_re, src_im, 7 * l);
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
    src_im += stride<V, SrcCf>();

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
void two_passes(
  Int n, Int start_offset, Int end_offset, Int dft_size,
  ET<V>* ptr_arg, const ET<V>* tw)
{
  VEC_TYPEDEFS(V);

  auto off1 = (n >> log2(dft_size)) / 4 * stride<V, cf::Vec>();
  auto off2 = off1 + off1;
  auto off3 = off2 + off1;
  
  auto start = start_offset * cf::Vec::idx_ratio;
  auto end = end_offset * cf::Vec::idx_ratio;

  if(start != 0)
    tw += 3 * stride<V, cf::Vec>() * (start >> log2(off1 + off3));

  for(auto p = ptr_arg + start; p < ptr_arg + end;)
  {
    auto tw0 = C::load(tw);
    auto tw1 = C::load(tw + stride<V, cf::Vec>());
    auto tw2 = C::load(tw + 2 * stride<V, cf::Vec>());
    tw += 3 * stride<V, cf::Vec>();

    for(auto end1 = p + off1;;)
    {
      ASSERT(p >= ptr_arg);
      ASSERT(p + off3 < ptr_arg + n * cf::Vec::idx_ratio);

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
void bit_reverse_pass(
  Int n, const ET<V>* src, ET<V>* dst_re, ET<V>* dst_im)
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

  _mm_sfence();
}

template<typename V, typename DstCf>
void last_three_passes(
  Int n, const ET<V>* src, const ET<V>* tw, ET<V>* dst_re, ET<V>* dst_im)
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
    auto d_re = dst_re + br.br * stride<V, DstCf>();
    auto d_im = dst_im + br.br * stride<V, DstCf>();
    auto s = src + 8 * stride<V, cf::Vec>() * br.i;
    auto this_tw = tw + 5 * stride<V, cf::Vec>() * br.i;

    C a0, a1, a2, a3, a4, a5, a6, a7;
    {
      C tw0 = C::load(this_tw);
      C tw1 = C::load(this_tw + stride<V, cf::Vec>());
      C tw2 = C::load(this_tw + 2 * stride<V, cf::Vec>());

      {
        C mul0 =       C::load(s);
        C mul1 = tw0 * C::load(s + 2 * stride<V, cf::Vec>());
        C mul2 = tw1 * C::load(s + 4 * stride<V, cf::Vec>());
        C mul3 = tw2 * C::load(s + 6 * stride<V, cf::Vec>());

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
        C mul0 =       C::load(s + 1 * stride<V, cf::Vec>());
        C mul1 = tw0 * C::load(s + 3 * stride<V, cf::Vec>());
        C mul2 = tw1 * C::load(s + 5 * stride<V, cf::Vec>());
        C mul3 = tw2 * C::load(s + 7 * stride<V, cf::Vec>());

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
      C tw3 = C::load(this_tw + 3 * stride<V, cf::Vec>());
      {
        auto mul = tw3 * a4;
        store<DstCf>(a0 + mul, d_re, d_im, 0);
        store<DstCf>(a0 - mul, d_re, d_im, l4 * stride<V, DstCf>());
      }

      {
        auto mul = tw3.mul_neg_i() * a6;
        store<DstCf>(a2 + mul, d_re, d_im, l2 * stride<V, DstCf>());
        store<DstCf>(a2 - mul, d_re, d_im, l6 * stride<V, DstCf>());
      }
    }

    {
      C tw4 = C::load(this_tw + 4 * stride<V, cf::Vec>());
      {
        auto mul = tw4 * a5;
        store<DstCf>(a1 + mul, d_re, d_im, l1 * stride<V, DstCf>());
        store<DstCf>(a1 - mul, d_re, d_im, l5 * stride<V, DstCf>());
      }

      {
        auto mul = tw4.mul_neg_i() * a7;
        store<DstCf>(a3 + mul, d_re, d_im, l3 * stride<V, DstCf>());
        store<DstCf>(a3 - mul, d_re, d_im, l7 * stride<V, DstCf>());
      }
    }
  }
}

template<typename V>
void last_three_passes_in_place(
  Int n, Int start_offset, Int end_offset, ET<V>* ptr_arg, const ET<V>* tw_arg)
{
  VEC_TYPEDEFS(V);
  
  auto start = start_offset * cf::Vec::idx_ratio;
  auto end = end_offset * cf::Vec::idx_ratio;
  
  const T* tw = tw_arg + start * 5 / 8;

  for(auto p = ptr_arg + start; p < ptr_arg + end;)
  {
    C a0, a1, a2, a3, a4, a5, a6, a7;
    {
      C tw0 = C::load(tw);
      C tw1 = C::load(tw + stride<V, cf::Vec>());
      C tw2 = C::load(tw + 2 * stride<V, cf::Vec>());

      {
        C mul0 =       C::load(p);
        C mul1 = tw0 * C::load(p + 2 * stride<V, cf::Vec>());
        C mul2 = tw1 * C::load(p + 4 * stride<V, cf::Vec>());
        C mul3 = tw2 * C::load(p + 6 * stride<V, cf::Vec>());

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
        C mul0 =       C::load(p + 1 * stride<V, cf::Vec>());
        C mul1 = tw0 * C::load(p + 3 * stride<V, cf::Vec>());
        C mul2 = tw1 * C::load(p + 5 * stride<V, cf::Vec>());
        C mul3 = tw2 * C::load(p + 7 * stride<V, cf::Vec>());

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
      C tw3 = C::load(tw + 3 * stride<V, cf::Vec>());
      {
        auto mul = tw3 * a4;
        (a0 + mul).store(p + 0);
        (a0 - mul).store(p + 1 * stride<V, cf::Vec>());
      }

      {
        auto mul = tw3.mul_neg_i() * a6;
        (a2 + mul).store(p + 2 * stride<V, cf::Vec>());
        (a2 - mul).store(p + 3 * stride<V, cf::Vec>());
      }
    }

    {
      C tw4 = C::load(tw + 4 * stride<V, cf::Vec>());
      {
        auto mul = tw4 * a5;
        (a1 + mul).store(p + 4 * stride<V, cf::Vec>());
        (a1 - mul).store(p + 5 * stride<V, cf::Vec>());
      }

      {
        auto mul = tw4.mul_neg_i() * a7;
        (a3 + mul).store(p + 6 * stride<V, cf::Vec>());
        (a3 - mul).store(p + 7 * stride<V, cf::Vec>());
      }
    }

    p += 8 * stride<V, cf::Vec>();
    tw += 5 * stride<V, cf::Vec>();
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

  for(Int i = 0; i < vn; i++)
  {
    C c;
    constexpr bool result_in_a = is_power_of_4(n);
    if(result_in_a) c = { a_re[i], a_im[i] };
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
  Int dft_size = 1;

  Int first_npasses = get_npasses<V>(n, 1);
  if(first_npasses == 2) first_two_passes<V, SrcCf>(n, src_re, src_im, w);
  else first_three_passes<V, SrcCf>(n, src_re, src_im, w);

  dft_size <<= first_npasses;

  for(Int i = 1;; i++)
  {
    Int npasses = get_npasses<V>(n, dft_size);
    Int next_dft_size = dft_size << npasses; 
    if(next_dft_size == n)
    {
      if(npasses == 2)
        last_two_passes<V, DstCf>(n, w, state->twiddle[i], dst_re, dst_im);
      else
        last_three_passes<V, DstCf>(n, w, state->twiddle[i], dst_re, dst_im);

      break;
    }
    else
    {
      two_passes<V>(n, 0, n, dft_size, w, state->twiddle[i]);
      dft_size = next_dft_size;
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

  Int n = state->n;

  if(npasses == 3) 
    last_three_passes_in_place<V>(n, start, end, p, state->twiddle[step]);
  else
    two_passes<V>(n, start, end, dft_size, p, state->twiddle[step]);

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

  if(first_npasses == 2) first_two_passes<V, SrcCf>(n, src_re, src_im, w);
  else first_three_passes<V, SrcCf>(n, src_re, src_im, w);

  recursive_passes<V>(state, 1, w, 0, n);

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

template<typename T>
FORCEINLINE void fft_impl(
  const Fft<T>* state,
  const T* src_re, const T* src_im,
  T* dst_re, T* dst_im)
{
  state->transform_fun(state, src_re, src_im, dst_re, dst_im);
}

template<typename T>
void fft(const Fft<T>* state, T* src, T* dst)
{
  fft_impl(state, src, src + state->n, dst, dst + state->n);
}

template<typename T>
void ifft(const Ifft<T>* state, T* src, T* dst)
{
  Int n = ((Fft<T>*) state)->n;
  fft_impl((Fft<T>*) state, src, src + n, dst, dst + n);
}

template<
  typename V,
  typename SrcCf,
  typename DstCf,
  bool inverse>
void real_pass(
  Int n,
  typename V::T* src_re,
  typename V::T* src_im,
  typename V::T* twiddle,
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
    C w = load<V, cf::Split>(twiddle + iw, n / 2);
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
  void (*real_pass)(Int, T*, T*, T*, T*, T*);
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

  fft_impl(state->state, src, src + n, w, w + w_im_off);

  state->real_pass(
    n * 2,
    w, w + w_im_off, 
    state->twiddle,
    dst, dst + dst_im_off);
}

template<typename T>
struct Irfft
{
  Ifft<T>* state;
  T* twiddle;
  void (*real_pass)(Int, T*, T*, T*, T*, T*);
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
    src + align_size<T>(complex_state->n + 1),
    state->twiddle,
    dst,
    dst + complex_state->n);

  ifft(state->state, dst, dst);
}
}
#endif
