#ifndef FFT_MULTI_H
#define FFT_MULTI_H

namespace multi
{

template<typename T>
struct ComplexPointers
{
  T* re;
  T* im;
};

template<typename SrcRows, typename DstRows>
void first_pass(Int n, const SrcRows& src, const DstRows& dst)
{
  typedef typename SrcRows::V V;
  VEC_TYPEDEFS(V);
  using SrcCf = typename SrcRows::Cf;
  using DstCf = typename DstRows::Cf;

  for(auto i = 0; i < n / 2; i++)
  {
    auto src0 = src.row(i);
    auto src1 = src.row(i + n / 2);
    auto dst0 = dst.row(i);
    auto dst1 = dst.row(i + n / 2);
    for(Int s = 0, d = 0; s < dst.m * SrcCf::idx_ratio;
      s += stride<V, SrcCf>(), d += stride<V, DstCf>())
    {
      C a0 = load<V, SrcCf>(src0.re, src0.im, s);
      C a1 = load<V, SrcCf>(src1.re, src1.im, s);
      store<DstCf>(a0 + a1, dst0.re, dst0.im, d);
      store<DstCf>(a0 - a1, dst1.re, dst1.im, d);
    }
  }
}

template<typename SrcRows, typename DstRows>
void first_two_passes(
  Int n, const SrcRows& src , const DstRows& dst)
{
  typedef typename SrcRows::V V;
  VEC_TYPEDEFS(V);
  using SrcCf = typename SrcRows::Cf;
  using DstCf = typename DstRows::Cf;

  for(auto i = 0; i < n / 4; i++)
  {
    auto src0 = src.row(i);
    auto src1 = src.row(i + n / 4);
    auto src2 = src.row(i + 2 * n / 4);
    auto src3 = src.row(i + 3 * n / 4);
    auto dst0 = dst.row(i);
    auto dst1 = dst.row(i + n / 4);
    auto dst2 = dst.row(i + 2 * n / 4);
    auto dst3 = dst.row(i + 3 * n / 4);

    for(Int s = 0, d = 0; s < dst.m * SrcCf::idx_ratio;
      s += stride<V, SrcCf>(), d += stride<V, DstCf>())
    {
      C a0 = load<V, SrcCf>(src0.re, src0.im, s);
      C a1 = load<V, SrcCf>(src1.re, src1.im, s);
      C a2 = load<V, SrcCf>(src2.re, src2.im, s);
      C a3 = load<V, SrcCf>(src3.re, src3.im, s);

      C b0 = a0 + a2;
      C b2 = a0 - a2;
      C b1 = a1 + a3;
      C b3 = a1 - a3;

      C c0 = b0 + b1;
      C c1 = b0 - b1;
      C c2 = b2 + b3.mul_neg_i();
      C c3 = b2 - b3.mul_neg_i();

      store<DstCf>(c0, dst0.re, dst0.im, d);
      store<DstCf>(c1, dst1.re, dst1.im, d);
      store<DstCf>(c2, dst2.re, dst2.im, d);
      store<DstCf>(c3, dst3.re, dst3.im, d);
    }
  }
}

template<typename V, typename Cf>
NOINLINE void two_passes_inner(
  const ComplexPointers<ET<V>>& a0,
  const ComplexPointers<ET<V>>& a1,
  const ComplexPointers<ET<V>>& a2,
  const ComplexPointers<ET<V>>& a3,
  Complex<V> t0,
  Complex<V> t1,
  Complex<V> t2,
  Int m)
{
  VEC_TYPEDEFS(V);
  for(Int i = 0; i < m * Cf::idx_ratio; i += stride<V, Cf>())
  {
    C b0 = load<V, Cf>(a0.re, a0.im, i);
    C b1 = load<V, Cf>(a1.re, a1.im, i);
    C b2 = load<V, Cf>(a2.re, a2.im, i);
    C b3 = load<V, Cf>(a3.re, a3.im, i);
    onedim::two_passes_inner(b0, b1, b2, b3, b0, b1, b2, b3, t0, t1, t2);
    store<Cf>(b0, a0.re, a0.im, i);
    store<Cf>(b2, a1.re, a1.im, i);
    store<Cf>(b1, a2.re, a2.im, i);
    store<Cf>(b3, a3.re, a3.im, i);
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
  auto tw = twiddle + 6 * (start >> log2(stride));

  C tw0 = {V::vec(tw[0]), V::vec(tw[1])}; 
  C tw1 = {V::vec(tw[2]), V::vec(tw[3])}; 
  C tw2 = {V::vec(tw[4]), V::vec(tw[5])}; 

  for(Int j = start; j < start + stride / 4; j++)
    two_passes_inner<typename Rows::V, typename Rows::Cf>(
      data.row(j + 0 * stride / 4),
      data.row(j + 1 * stride / 4),
      data.row(j + 2 * stride / 4),
      data.row(j + 3 * stride / 4),
      tw0, tw1, tw2, data.m);
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
    C b0 = load<V, Cf>(p0.re, p0.im, i);
    C mul = load<V, Cf>(p1.re, p1.im, i) * tw;
    store<Cf>(b0 + mul, p0.re, p0.im, i);
    store<Cf>(b0 - mul, p1.re, p1.im, i);
  }
}

template<typename T>
struct Fft
{
  Int n;
  Int m;
  T* working;
  T* twiddle[8 * sizeof(Int) / 2];
  void (*fun_ptr)(
    const Fft<T>* state,
    const T* src_re, const T* src_im_arg,
    T* dst_re, T* dst_im_arg,
    bool interleaved_src_rows,
    bool interleaved_dst_rows);
};

template<typename V_, typename T, typename Cf_>
struct Rows
{
  typedef Cf_ Cf;
  typedef V_ V;
  T* re_ptr_;
  T* im_ptr_;
  Int m;
  Int row_stride;
  ComplexPointers<T> row(Int i) const
  {
    Int off = i * row_stride * Cf::idx_ratio;;
    return {re_ptr_ + off, im_ptr_ + off};
  }
};

template<typename V_, typename Cf_>
struct BrRows
{
  typedef V_ V;
  typedef Cf_ Cf;
  Int nbits;
  ET<V>* re_ptr_;
  ET<V>* im_ptr_;
  Int m;
  Int row_stride;
  ComplexPointers<ET<V>> row(Int i) const
  {
    Int off = reverse_bits(i, nbits) * row_stride * Cf::idx_ratio;
    return {re_ptr_ + off, im_ptr_ + off};
  }
};

template<typename Rows>
void fft_recurse(
  Int n,
  Int start,
  Int end,
  Int dft_size,
  typename Rows::V::T* const * twiddle,
  const Rows& rows)
{
  if(4 * dft_size <= n)
  {
    two_passes(n, dft_size, start, end, *twiddle, rows);

    Int l = (end - start) / 4;
    if(4 * dft_size < n)
      for(Int i = start; i < end; i += l)
        fft_recurse(n, i, i + l, dft_size * 4, twiddle + 1, rows);
  }
  else
    last_pass(n, start, end, *twiddle, rows);
}

// The result is bit reversed
template<typename SrcRows, typename DstRows>
void fft_impl(
  Int n,
  typename SrcRows::V::T* const * twiddle,
  const SrcRows& src,
  const DstRows& dst)
{
  if(n == 1)
  {
    auto [s_re, s_im] = src.row(0);
    auto [d_re, d_im] = dst.row(0);
    complex_copy<typename SrcRows::V, typename SrcRows::Cf, typename DstRows::Cf>(
      s_re, s_im, src.m, d_re, d_im);
  }
  else if(n == 2)
    first_pass(n, src, dst);
  else
  {
    first_two_passes(n, src, dst);

    if(n > 4) 
      for(Int i = 0; i < n; i += n / 4)
        fft_recurse(n, i, i + n / 4, 4, twiddle + 1, dst);
  }
}

//If interleaved_src_rows is true, then src_im_arg is ignored
//Same goes for interleaved_dst_rows and dst_im_arg.
template<typename V, typename SrcCf, typename DstCf, bool br_dst_rows>
void fft(
  const Fft<typename V::T>* s,
  const ET<V>* src_re, const ET<V>* src_im_arg,
  ET<V>* dst_re, ET<V>* dst_im_arg,
  bool interleaved_src_rows,
  bool interleaved_dst_rows)
{
  auto src_im = interleaved_src_rows ? src_re + s->m : src_im_arg;
  Int src_stride = interleaved_src_rows ? 2 * s->m : s->m;

  Rows<V, const ET<V>, SrcCf> src_rows{src_re, src_im, s->m, src_stride};

  auto dst_im = interleaved_dst_rows ? dst_re + s->m : dst_im_arg;
  Int dst_stride = interleaved_dst_rows ? 2 * s->m : s->m;

  if(br_dst_rows)
    fft_impl(
      s->n, s->twiddle, src_rows,
      Rows<V, ET<V>, DstCf>({dst_re, dst_im, s->m, dst_stride}));
  else
    fft_impl(
      s->n, s->twiddle, src_rows, 
      BrRows<V, DstCf>{log2(s->n), dst_re, dst_im, s->m, dst_stride});
}

template<
  bool do_create, typename V, typename SrcCf, typename DstCf, bool br_dst_rows>
Int fft_create_impl(Int n, Int m, void* ptr)
{
  VEC_TYPEDEFS(V);
 
  Fft<T> local_fft;
  auto r = do_create ? (Fft<T>*) ptr : &local_fft;
  ptr = aligned_increment(ptr, sizeof(Fft<T>));
  
  r->n = n;
  r->m = m;

  r->working = (T*) ptr;
  ptr = aligned_increment(ptr, 2 * n * sizeof(T));

  if(do_create) compute_twiddle_range<V>(n, r->working, r->working + n);

  for(Int dft_size = 1, i = 0; dft_size < n; i++)
  {
    Int npasses = 4 * dft_size <= n ? 2 : 1;

    using S = Scalar<T>;
    T* tw = (T*) ptr;
    r->twiddle[i] = tw;
    ptr = aligned_increment(
      ptr, twiddle_for_step_memsize<S>(dft_size, npasses));

    if(do_create)
      twiddle_for_step_create<S>(r->working, n, dft_size, npasses, tw);

    dft_size <<= npasses;
  }

  r->fun_ptr = &fft<V, SrcCf, DstCf, br_dst_rows>;

  ptr = aligned_increment(ptr, 2 * n * sizeof(T));

  return Int(ptr);
}

template<typename V, typename SrcCf, typename DstCf, bool br_dst_rows>
Int fft_memsize(Int n, Int m)
{
  return fft_create_impl<false, V, SrcCf, DstCf, br_dst_rows>(n, m, nullptr);
}

template<typename V, typename SrcCf, typename DstCf, bool br_dst_rows>
Fft<typename V::T>* fft_create(Int n, Int m, void* ptr)
{
  fft_create_impl<true, V, SrcCf, DstCf, br_dst_rows>(n, m, ptr);
  return (Fft<typename V::T>*) ptr;
}

template<typename V, typename DstCf, bool inverse>
void real_pass(Int n, Int m, ET<V>* twiddle, ET<V>* dst_re, ET<V>* dst_im)
{
  VEC_TYPEDEFS(V);

  Vec half = V::vec(0.5);
  Int nbits = log2(n / 2);

  for(Int i = 1; i <= n / 4; i++)
  {
    C w = { V::vec(twiddle[i]), V::vec(twiddle[i + n / 2]) };

    auto re0 = dst_re + i * m * DstCf::idx_ratio; 
    auto im0 = dst_im + i * m * DstCf::idx_ratio; 
    auto re1 = dst_re + (n / 2 - i) * m * DstCf::idx_ratio; 
    auto im1 = dst_im + (n / 2 - i) * m * DstCf::idx_ratio; 

    for(Int off = 0; off < m * DstCf::idx_ratio; off += stride<V, DstCf>())
    {
      C sval0 = load<V, DstCf>(re0, im0, off);
      C sval1 = load<V, DstCf>(re1, im1, off);

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

      store<DstCf>(dval0, re0, im0, off);
      store<DstCf>(dval1, re1, im1, off);
    }
  }

  if(inverse)
  {
    auto re0 = dst_re;
    auto im0 = dst_im;
    auto re1 = dst_re + n / 2 * m * DstCf::idx_ratio; 
    auto im1 = dst_im + n / 2 * m * DstCf::idx_ratio; 

    for(Int off = 0; off < m * DstCf::idx_ratio; off += stride<V, DstCf>())
    {
      Vec r0 = load<V, DstCf>(re0, im0, off).re;
      Vec r1 = load<V, DstCf>(re1, im1, off).re;
      store<DstCf>(C{r0 + r1, r0 - r1}, re0, im0, off);
    }
  }
  else
  {
    auto re0 = dst_re; 
    auto im0 = dst_im; 
    auto re1 = dst_re + n / 2 * m * DstCf::idx_ratio;
    auto im1 = dst_im + n / 2 * m * DstCf::idx_ratio;

    for(Int off = 0; off < m * DstCf::idx_ratio; off += stride<V, DstCf>())
    {
      C r0 = load<V, DstCf>(re0, im0, off);
      store<DstCf>(C{r0.re + r0.im, V::vec(0)}, re0, im0, off);
      store<DstCf>(C{r0.re - r0.im, V::vec(0)}, re1, im1, off);
    }
  }
}
}

#endif
