#ifndef FFT_MULTI_H
#define FFT_MULTI_H

namespace multi
{
template<typename SrcRows, typename DstRows>
void first_pass(Int n, const SrcRows& src, const DstRows& dst)
{
  typedef typename SrcRows::V V; VEC_TYPEDEFS(V);

  for(auto i = 0; i < n / 2; i++)
  {
    auto s0 = src.row(i);
    auto s1 = src.row(i + n / 2);
    auto d0 = dst.row(i);
    auto d1 = dst.row(i + n / 2);
    for(auto end = s0 + dst.m * SrcRows::Cf::idx_ratio; s0 < end;)
    {
      C a0 = load<V, typename SrcRows::Cf>(s0, src.im_off);
      C a1 = load<V, typename SrcRows::Cf>(s1, src.im_off);
      DstRows::Cf::store(a0 + a1, d0, dst.im_off);
      DstRows::Cf::store(a0 - a1, d1, dst.im_off);
      s0 += stride<V, typename SrcRows::Cf>();
      s1 += stride<V, typename SrcRows::Cf>();
      d0 += stride<V, typename DstRows::Cf>();
      d1 += stride<V, typename DstRows::Cf>();
    }
  }
}

template<typename SrcRows, typename DstRows>
void first_two_passes(
  Int n, const SrcRows& src , const DstRows& dst)
{
  typedef typename SrcRows::V V; VEC_TYPEDEFS(V);

  for(auto i = 0; i < n / 4; i++)
  {
    auto s0 = src.row(i);
    auto s1 = src.row(i + n / 4);
    auto s2 = src.row(i + 2 * n / 4);
    auto s3 = src.row(i + 3 * n / 4);
    auto d0 = dst.row(i);
    auto d1 = dst.row(i + n / 4);
    auto d2 = dst.row(i + 2 * n / 4);
    auto d3 = dst.row(i + 3 * n / 4);

    for(auto end = s0 + dst.m * SrcRows::Cf::idx_ratio; s0 < end;)
    {
      C a0 = load<V, typename SrcRows::Cf>(s0, src.im_off);
      C a1 = load<V, typename SrcRows::Cf>(s1, src.im_off);
      C a2 = load<V, typename SrcRows::Cf>(s2, src.im_off);
      C a3 = load<V, typename SrcRows::Cf>(s3, src.im_off);

      C b0 = a0 + a2;
      C b2 = a0 - a2;
      C b1 = a1 + a3;
      C b3 = a1 - a3;

      C c0 = b0 + b1;
      C c1 = b0 - b1;
      C c2 = b2 + b3.mul_neg_i();
      C c3 = b2 - b3.mul_neg_i();

      DstRows::Cf::store(c0, d0, dst.im_off);
      DstRows::Cf::store(c1, d1, dst.im_off);
      DstRows::Cf::store(c2, d2, dst.im_off);
      DstRows::Cf::store(c3, d3, dst.im_off);

      s0 += stride<V, typename SrcRows::Cf>();
      s1 += stride<V, typename SrcRows::Cf>();
      s2 += stride<V, typename SrcRows::Cf>();
      s3 += stride<V, typename SrcRows::Cf>();
      d0 += stride<V, typename DstRows::Cf>();
      d1 += stride<V, typename DstRows::Cf>();
      d2 += stride<V, typename DstRows::Cf>();
      d3 += stride<V, typename DstRows::Cf>();
    }
  }
}

template<typename V, typename Cf>
NOINLINE void two_passes_inner(
  typename V::T* a0,
  typename V::T* a1,
  typename V::T* a2,
  typename V::T* a3,
  Complex<V> t0,
  Complex<V> t1,
  Complex<V> t2,
  Int m,
  Int im_off)
{
  VEC_TYPEDEFS(V);
  for(Int i = 0; i < m * Cf::idx_ratio; i += stride<V, Cf>())
  {
    C b0 = load<V, Cf>(a0 + i, im_off);
    C b1 = load<V, Cf>(a1 + i, im_off);
    C b2 = load<V, Cf>(a2 + i, im_off);
    C b3 = load<V, Cf>(a3 + i, im_off);
    onedim::two_passes_inner(b0, b1, b2, b3, b0, b1, b2, b3, t0, t1, t2);
    Cf::store(b0, a0 + i, im_off);
    Cf::store(b2, a1 + i, im_off);
    Cf::store(b1, a2 + i, im_off);
    Cf::store(b3, a3 + i, im_off);
  }
}

template<typename Rows>
void two_passes(
  Int n,
  Int dft_size,
  Int start,
  Int end,
  typename Rows::V::T* twiddle,
  const Rows& data)
{
  typedef typename Rows::V V; VEC_TYPEDEFS(V);

  Int stride = end - start;
  ASSERT(stride * dft_size == n);
  auto tw = twiddle + 2 * (n - 4 * dft_size) + 6 * (start >> log2(stride));

  C tw0 = {V::vec(tw[0]), V::vec(tw[1])}; 
  C tw1 = {V::vec(tw[2]), V::vec(tw[3])}; 
  C tw2 = {V::vec(tw[4]), V::vec(tw[5])}; 

  for(Int j = start; j < start + stride / 4; j++)
    two_passes_inner<typename Rows::V, typename Rows::Cf>(
      data.row(j + 0 * stride / 4),
      data.row(j + 1 * stride / 4),
      data.row(j + 2 * stride / 4),
      data.row(j + 3 * stride / 4),
      tw0, tw1, tw2, data.m, data.im_off);
}

template<typename Rows>
void last_pass(
  Int n, Int start, Int end,
  typename Rows::V::T* twiddle, const Rows& rows)
{
  typedef typename Rows::V V; VEC_TYPEDEFS(V);
  typedef typename Rows::Cf Cf;
  ASSERT(end - start == 2);

  C tw = {V::vec(twiddle[start]), V::vec(twiddle[start + 1])}; 
  auto p0 = rows.row(start);
  auto p1 = rows.row(start + 1);
  for(Int i = 0; i < rows.m * Rows::Cf::idx_ratio; i += stride<V, Cf>())
  {
    C b0 = load<V, Cf>(p0 + i, rows.im_off);
    C mul = load<V, Cf>(p1 + i, rows.im_off) * tw;
    Rows::Cf::store(b0 + mul, p0 + i, rows.im_off);
    Rows::Cf::store(b0 - mul, p1 + i, rows.im_off);
  }
}

template<typename T>
struct Fft
{
  Int n;
  Int m;
  T* working;
  T* twiddle;
  void (*fun_ptr)(
    const Fft<T>* state,
    T* src,
    T* dst,
    Int im_off,
    bool interleaved_src_rows,
    bool interleaved_dst_rows);
};

template<typename V_, typename Cf_>
struct Rows
{
  typedef Cf_ Cf;
  typedef V_ V;
  typename V::T* ptr_;
  Int m;
  Int row_stride;
  Int im_off;
  typename V::T* row(Int i) const
  {
    return ptr_ + i * row_stride * Cf::idx_ratio;
  }
};

template<typename V_, typename Cf_>
struct BrRows
{
  typedef V_ V;
  typedef Cf_ Cf;
  Int log2n;
  typename V::T* ptr_;
  Int m;
  Int row_stride;
  Int im_off;

  BrRows(Int n, typename V::T* ptr_, Int m, Int row_stride, Int im_off)
  : log2n(log2(n)), ptr_(ptr_), m(m), row_stride(row_stride), im_off(im_off) {}

  typename V::T* row(Int i) const
  {
    return ptr_ + reverse_bits(i, log2n) * row_stride * Cf::idx_ratio;
  }
};

template<typename Rows>
void fft_recurse(
  Int n,
  Int start,
  Int end,
  Int dft_size,
  typename Rows::V::T* twiddle,
  const Rows& rows)
{
  if(4 * dft_size <= n)
  {
    two_passes(n, dft_size, start, end, twiddle, rows);

    Int l = (end - start) / 4;
    if(4 * dft_size < n)
      for(Int i = start; i < end; i += l)
        fft_recurse(n, i, i + l, dft_size * 4, twiddle, rows);
  }
  else
    last_pass(n, start, end, twiddle, rows);
}

// The result is bit reversed
template<typename SrcRows, typename DstRows>
void fft_impl(
  Int n,
  typename SrcRows::V::T* twiddle,
  const SrcRows& src,
  const DstRows& dst)
{
  if(n == 1)
    complex_copy<typename SrcRows::V, typename SrcRows::Cf, typename DstRows::Cf>(
      src.row(0), src.im_off, src.m, dst.row(0), dst.im_off);
  else
  {
    if(n == 2)
      first_pass(n, src, dst);
    else
      first_two_passes(n, src, dst);

    if(n > 4) 
      for(Int i = 0; i < n; i += n / 4)
        fft_recurse(n, i, i + n / 4, 4, twiddle, dst);
  }
}

template<typename V, typename SrcCf, typename DstCf, bool br_dst_rows>
void fft(
  const Fft<typename V::T>* s,
  typename V::T* src,
  typename V::T* dst,
  Int im_off,
  bool interleaved_src_rows,
  bool interleaved_dst_rows)
{
  auto src_rows = interleaved_src_rows
    ? Rows<V, SrcCf>({src, s->m, 2 * s->m, s->m})
    : Rows<V, SrcCf>({src, s->m, s->m, im_off});

  Int dst_off = interleaved_dst_rows ? s->m : im_off;
  Int dst_stride = interleaved_dst_rows ? 2 * s->m : s->m;

  if(br_dst_rows)
    fft_impl(
      s->n, s->twiddle, src_rows,
      Rows<V, DstCf>({dst, s->m, dst_stride, dst_off}));
  else
    fft_impl(
      s->n, s->twiddle, src_rows, 
      BrRows<V, DstCf>({s->n, dst, s->m, dst_stride, dst_off}));
}

template<typename V>
Int fft_memsize(Int n)
{
  VEC_TYPEDEFS(V);
  Int r = 0;
  r = aligned_increment(r, sizeof(Fft<T>));
  r = aligned_increment(r, 2 * n * sizeof(T));
  r = aligned_increment(r, 2 * n * sizeof(T));
  return r;
}

template<typename V, typename SrcCf, typename DstCf, bool br_dst_rows>
Fft<typename V::T>* fft_create(Int n, Int m, void* ptr)
{
  VEC_TYPEDEFS(V);
  auto r = (Fft<T>*) ptr;
  ptr = aligned_increment(ptr, sizeof(Fft<T>));
  r->n = n;
  r->m = m;
  r->working = (T*) ptr;
  ptr = aligned_increment(ptr, 2 * n * sizeof(T));
  r->twiddle = (T*) ptr;
  ptr = aligned_increment(ptr, 2 * n * sizeof(T));
  
  init_twiddle<Scalar<T>>(
    [n](Int s, Int dft_size){ return 4 * dft_size <= n ? 2 : 1; },
    n, r->working, r->twiddle, nullptr);

  r->fun_ptr = &fft<V, SrcCf, DstCf, br_dst_rows>;
  return r;
}

// src and dst must not be the same
template<typename V, typename DstCf, bool inverse>
void real_pass(
  Int n, Int m, typename V::T* twiddle, typename V::T* dst, Int dst_im_off)
{
  VEC_TYPEDEFS(V);

  Vec half = V::vec(0.5);
  Int nbits = log2(n / 2);

  for(Int i = 1; i <= n / 4; i++)
  {
    C w = { V::vec(twiddle[i]), V::vec(twiddle[i + n / 2]) };

    auto d0 = dst + i * m * DstCf::idx_ratio; 
    auto d1 = dst + (n / 2 - i) * m * DstCf::idx_ratio; 

    for(auto end = d0 + m * DstCf::idx_ratio; d0 < end;)
    {
      C sval0 = load<V, DstCf>(d0, dst_im_off);
      C sval1 = load<V, DstCf>(d1, dst_im_off);

      C a, b;

      if(inverse)
      {
        a = sval0 + sval1.adj();
        b = (sval1.adj() - sval0) * w.adj();
      }
      else
      {
        a = (sval0 + sval1.adj()) * half;
        b = ((sval0 - sval1.adj()) * w) * half;
      }

      C dval0 = a + b.mul_neg_i();
      C dval1 = a.adj() + b.adj().mul_neg_i();

      DstCf::store(dval0, d0, dst_im_off);
      DstCf::store(dval1, d1, dst_im_off);

      d0 += stride<V, DstCf>();
      d1 += stride<V, DstCf>();
    }
  }

  if(inverse)
  {
    auto s0 = dst; 
    auto s1 = dst + n / 2 * m * DstCf::idx_ratio; 
    auto d = dst; 

    for(auto end = s0 + m * DstCf::idx_ratio; s0 < end;)
    {
      Vec r0 = load<V, DstCf>(s0, dst_im_off).re;
      Vec r1 = load<V, DstCf>(s1, dst_im_off).re;
      DstCf::template store<V>({r0 + r1, r0 - r1}, d, dst_im_off);

      s0 += stride<V, DstCf>();
      s1 += stride<V, DstCf>();
      d += stride<V, DstCf>();
    }
  }
  else
  {
    auto d0 = dst; 
    auto d1 = dst + n / 2 * m * DstCf::idx_ratio; 
    auto s = dst; 

    for(auto end = s + m * DstCf::idx_ratio; s < end;)
    {
      C r0 = load<V, DstCf>(s, dst_im_off);
      DstCf::template store<V>({r0.re + r0.im, V::vec(0)}, d0, dst_im_off);
      DstCf::template store<V>({r0.re - r0.im, V::vec(0)}, d1, dst_im_off);
      
      s += stride<V, DstCf>();
      d0 += stride<V, DstCf>();
      d1 += stride<V, DstCf>();
    }
  }
}
}

#endif
