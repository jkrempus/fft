#ifndef FFT_CORE_H
#define FFT_CORE_H

#include "onedim.hpp"
#include "multi.hpp"

namespace
{
constexpr Int maxdim = 64;

template<typename T>
struct Fft
{
  Int ndim;
  Int num_elements;
  Int working_idx_ratio;
  Int dst_idx_ratio;
  T* working_re;
  T* working_im;
  onedim::Fft<T>* last_transform;
  multi::Fft<T>* transforms[maxdim];
};

template<bool do_create, typename V, typename SrcCf, typename DstCf>
Int fft_create_impl(Int ndim_in, const Int* dim_in, void* mem)
{
  Int dim[maxdim];
  Int ndim;
  remove_redundant_dimensions(dim_in, ndim_in, dim, ndim);
  Int num_elements = product(dim, ndim);

  VEC_TYPEDEFS(V);
  if(V::vec_size > 1 && dim[ndim - 1] < 2 * V::vec_size)
    return fft_create_impl<do_create, Scalar<T>, SrcCf, DstCf>(ndim, dim, mem);

  // Because the initial alignment of mem can differ between different calls,
  // the first aligned_increment call can increase mem more in some calls than
  // it does in others (but the difference cannot be larger than align_bytes -
  // 1).  We ensure that the size computed during the memsize call will always
  // be large enough by adding "padding" in that call.
  Uint padding = do_create ? 0 : align_bytes - 1;

  auto s = (Fft<typename V::T>*) mem;
  mem = aligned_increment(mem, sizeof(Fft<T>) + padding);

  using WorkingCf = cf::Vec;

  if(do_create)
  {
    s->ndim = ndim;
    s->working_idx_ratio = WorkingCf::idx_ratio;
    s->dst_idx_ratio = DstCf::idx_ratio;
    s->working_re = (T*) mem;
    s->working_im = SameType<WorkingCf, cf::Split>::value ?
      (T*) mem + num_elements : (T*) nullptr;
    s->num_elements = num_elements;
  }
 
  Int working_size = 2 * sizeof(T) * num_elements;
  mem = (void*) align_size(Uint(mem) + working_size);

  if(ndim == 1)
  {
    if(do_create)
      s->last_transform =
        onedim::fft_create<V, SrcCf, DstCf>(dim[ndim - 1], mem);

    mem = aligned_increment(
      mem, onedim::fft_memsize<V, SrcCf, DstCf>(dim[ndim - 1]));
  }
  else
  {
    for(Int i = 0; i < ndim - 1; i++)
    {
      Int m = product(dim + i + 1, dim + ndim);

      if(i == 0)
      {
        if(do_create)
          s->transforms[i] = multi::fft_create<V, SrcCf, WorkingCf, true>(
            dim[i], m, mem);

        mem = aligned_increment(
          mem, multi::fft_memsize<V, SrcCf, WorkingCf, true>(dim[i], m));
      }
      else
      {
        if(do_create)
          s->transforms[i] = multi::fft_create<V, WorkingCf, WorkingCf, true>(
            dim[i], m, mem);

        mem = aligned_increment(
          mem, multi::fft_memsize<V, WorkingCf, WorkingCf, true>(dim[i], m));
      }
    }

    if(do_create)
      s->last_transform =
        onedim::fft_create<V, WorkingCf, DstCf>(dim[ndim - 1], mem);

    mem = aligned_increment(
      mem, onedim::fft_memsize<V, WorkingCf, DstCf>(dim[ndim - 1]));
  }

  return Int(mem);
}

template<typename V, typename SrcCf, typename DstCf>
Int fft_memsize(Int ndim_in, const Int* dim_in)
{
  return fft_create_impl<false, V, SrcCf, DstCf>(ndim_in, dim_in, nullptr);
}

template<typename V, typename SrcCf, typename DstCf>
Fft<typename V::T>* fft_create(Int ndim_in, const Int* dim_in, void* mem)
{
  fft_create_impl<true, V, SrcCf, DstCf>(ndim_in, dim_in, mem);
  return (Fft<typename V::T>*) mem;
}

template<typename T>
void fft_impl(
  Int idim, Fft<T>* s,
  const T* src_re, const T* src_im,
  T* working_re, T* working_im,
  T* dst_re, T* dst_im,
  bool interleaved_src_rows)
{
  ASSERT(idim < s->ndim);
  if(idim == s->ndim - 1)
    onedim::fft(s->last_transform, src_re, src_im, dst_re, dst_im);
  else
  {
    s->transforms[idim]->fun_ptr(
      s->transforms[idim], src_re, src_im, working_re, working_im,
      interleaved_src_rows, false);

    Int m = s->transforms[idim]->m;
    Int n = s->transforms[idim]->n;
    for(BitReversed br(n); Int(br.i) < n; br.advance())
    {
      Int src_off = br.i * m * s->working_idx_ratio;;
      Int dst_off = br.br * m * s->dst_idx_ratio;
      fft_impl(
        idim + 1, s,
        working_re + src_off, working_im + src_off,
        working_re + src_off, working_im + src_off,
        dst_re + dst_off, dst_im + dst_off,
        interleaved_src_rows);
    }
  }
}

template<typename T>
void fft(
  Fft<T>* state, const T* src_re, const T* src_im, T* dst_re, T* dst_im)
{
  fft_impl<T>(
    0, state,
    src_re, src_im,
    state->working_re, state->working_im,
    dst_re, dst_im,
    false);
}

template<typename T>
struct Ifft
{
  Fft<T> state;
};

template<typename T>
void ifft(
  Ifft<T>* state, const T* src_re, const T* src_im, T* dst_re, T* dst_im)
{
  auto s = &state->state;
  fft_impl<T>(
    0, s,
    src_re, src_im,
    s->working_re, s->working_im,
    dst_re, dst_im,
    false);
}

template<typename V, typename SrcCf, typename DstCf>
Int ifft_memsize(Int ndim, const Int* dim)
{
  VEC_TYPEDEFS(V);
  return fft_memsize<V, cf::Swapped<SrcCf>, cf::Swapped<DstCf>>(ndim, dim);
}

template<typename V, typename SrcCf, typename DstCf>
Ifft<typename V::T>* ifft_create(Int ndim, const Int* dim, void* mem)
{
  VEC_TYPEDEFS(V);
  return (Ifft<T>*) fft_create<V, cf::Swapped<SrcCf>, cf::Swapped<DstCf>>(
    ndim, dim, mem);
}

template<typename T>
struct Rfft
{
  T* twiddle;
  T* working0;
  Int outer_n;
  Int inner_n;
  Int im_off;
  Int dst_idx_ratio;
  onedim::Rfft<T>* onedim_transform;
  Fft<T>* multidim_transform;
  multi::Fft<T>* first_transform;
  void (*real_pass)(Int n, Int m, T* twiddle, T* dst_re, T* dst_im);
};

template<typename T>
Int rfft_im_off(Int ndim, const Int* dim)
{
  return align_size<T>(product(dim + 1, dim + ndim) * (dim[0] / 2 + 1));
}

template<bool do_create, typename V, typename DstCf>
Int rfft_create_impl(Int ndim_in, const Int* dim_in, void* mem)
{
  Int dim[maxdim];
  Int ndim;
  remove_redundant_dimensions(dim_in, ndim_in, dim, ndim);

  VEC_TYPEDEFS(V)
  if(V::vec_size != 1 && dim[ndim - 1] < 2 * V::vec_size)
    return rfft_create_impl<do_create, Scalar<T>, DstCf>(ndim, dim, mem);

  // Because the initial alignment of mem can differ between different calls,
  // the first aligned_increment call can increase mem more in some calls than
  // it does in others (but the difference cannot be larger than align_bytes -
  // 1).  We ensure that the size computed during the memsize call will always
  // be large enough by adding "padding" in that call.
  Uint padding = do_create ? 0 : align_bytes - 1;

  auto r = (Rfft<T>*) mem;
  mem = aligned_increment(mem, sizeof(Rfft<T>) + padding);

  if(ndim == 1)
  {
    if(do_create)
    {
      r->onedim_transform = onedim::rfft_create<V, DstCf>(dim[0], mem);
      r->dst_idx_ratio = 0;
      r->working0 = nullptr;
      r->twiddle = nullptr;
      r->outer_n = 0;
      r->inner_n = 0;
      r->im_off = 0;
      r->multidim_transform = nullptr;
      r->first_transform = nullptr;
      r->real_pass = nullptr;
    }

    mem = aligned_increment(mem, onedim::rfft_memsize<V, DstCf>(dim[0]));
  }
  else
  {
    Int im_off = rfft_im_off<T>(ndim, dim);
    Int inner_n = product(dim + 1, dim + ndim);
    Int outer_n = dim[0];
    if(do_create)
    {
      r->dst_idx_ratio = DstCf::idx_ratio;
      r->outer_n = outer_n;
      r->inner_n = inner_n;
      r->onedim_transform = nullptr;
      r->im_off = im_off;
      r->twiddle = (T*) mem;
    }

    mem = aligned_increment(mem, sizeof(T) * dim[0]);

    if(do_create) r->working0 = (T*) mem;

    mem = aligned_increment(mem, 2 * sizeof(T) * im_off);

    if(do_create)
      r->first_transform = multi::fft_create<V, cf::Split, cf::Vec, false>(
        outer_n / 2, inner_n, mem);

    mem = aligned_increment(
      mem, multi::fft_memsize<V, cf::Split, cf::Vec, false>(
        outer_n / 2, inner_n));

    if(do_create)
    {
      r->multidim_transform =
        fft_create<V, cf::Vec, DstCf>(ndim - 1, dim + 1, mem);

      r->real_pass = &multi::real_pass<V, cf::Vec, false>;
  
      Int m =  r->outer_n / 2;
      compute_twiddle<V>(m, r->twiddle, r->twiddle + m);
    }

    mem = aligned_increment(
      mem, fft_memsize<V, cf::Vec, DstCf>(ndim - 1, dim + 1));
  }

  return Int(mem);
}

template<typename V, typename DstCf>
Int rfft_memsize(Int ndim_in, const Int* dim_in)
{
  return rfft_create_impl<false, V, DstCf>(ndim_in, dim_in, nullptr);
}

template<typename V, typename DstCf>
Rfft<typename V::T>* rfft_create(Int ndim_in, const Int* dim_in, void* mem)
{
  rfft_create_impl<true, V, DstCf>(ndim_in, dim_in, mem);
  return (Rfft<typename V::T>*) mem;
}

template<typename T>
void rfft(Rfft<T>* s, const T* src, T* dst_re, T* dst_im)
{
  if(s->onedim_transform)
    return onedim::rfft(s->onedim_transform, src, dst_re, dst_im);

  s->first_transform->fun_ptr(
    s->first_transform,
    src, nullptr,
    s->working0, s->working0 + s->outer_n / 2 * s->inner_n,
    true, false);

  s->real_pass(s->outer_n, s->inner_n, s->twiddle, s->working0, nullptr);

  const Int working_idx_ratio = s->multidim_transform->working_idx_ratio;
  const Int nbits = log2(s->outer_n / 2);
  for(Int i = 0; i < s->outer_n / 2 + 1 ; i++)
  {
    Int dst_off = i * s->inner_n * s->dst_idx_ratio;
    fft_impl(
      0, s->multidim_transform,
      s->working0 + i * s->inner_n * working_idx_ratio, (T*) nullptr,
      s->multidim_transform->working_re, s->multidim_transform->working_im,
      dst_re + dst_off, dst_im + dst_off,
      false);
  }
}

template<typename T>
struct Irfft
{
  T* twiddle;
  T* working0;
  T* working1;
  Int outer_n;
  Int inner_n;
  Int im_off;
  Int src_idx_ratio;
  onedim::Irfft<T>* onedim_transform;
  Fft<T>* multidim_transform;
  multi::Fft<T>* last_transform;
  void (*real_pass)(Int n, Int m, T* twiddle, T* dst_re, T* dst_im);
};

template<bool do_create, typename V, typename SrcCf>
Int irfft_create_impl(Int ndim_in, const Int* dim_in, void* mem)
{
  Int dim[maxdim];
  Int ndim;
  remove_redundant_dimensions(dim_in, ndim_in, dim, ndim);

  VEC_TYPEDEFS(V)
  if(V::vec_size != 1 && dim[ndim - 1] < 2 * V::vec_size)
    return irfft_create_impl<do_create, Scalar<T>, SrcCf>(ndim, dim, mem);

  // Because the initial alignment of mem can differ between different calls,
  // the first aligned_increment call can increase mem more in some calls than
  // it does in others (but the difference cannot be larger than align_bytes -
  // 1).  We ensure that the size computed during the memsize call will always
  // be large enough by adding "padding" in that call.
  Uint padding = do_create ? 0 : align_bytes - 1;

  auto r = (Irfft<T>*) mem;
  mem = aligned_increment(mem, sizeof(Irfft<T>) + padding);
  
  if(ndim == 1)
  {
    if(do_create)
    {
      r->onedim_transform = onedim::irfft_create<V, SrcCf>(dim[0], mem);
      r->src_idx_ratio = 0;
      r->working0 = nullptr;
      r->twiddle = nullptr;
      r->outer_n = 0;
      r->inner_n = 0;
      r->im_off = 0;
      r->multidim_transform = nullptr;
      r->last_transform = nullptr;
      r->real_pass = nullptr;
    }

    mem = aligned_increment(mem, onedim::irfft_memsize<V, SrcCf>(dim[0]));
  }
  else
  {
    Int im_off = rfft_im_off<T>(ndim, dim);
    Int inner_n = product(dim + 1, dim + ndim);
    Int outer_n = dim[0];
    if(do_create)
    {
      r->src_idx_ratio = SrcCf::idx_ratio;
      r->outer_n = outer_n;
      r->inner_n = inner_n;
      r->onedim_transform = nullptr;
      r->im_off = im_off;
      r->twiddle = (T*) mem;
    }

    mem = aligned_increment(mem, sizeof(T) * dim[0]);
    
    if(do_create) r->working0 = (T*) mem;
    mem = aligned_increment(mem, 2 * sizeof(T) * im_off);

    using SwSrc = cf::Swapped<SrcCf>;
    using SwVec = cf::Swapped<cf::Vec>;
    using SwSplit = cf::Swapped<cf::Split>;

    if(do_create)
      r->last_transform = multi::fft_create<
        V, SwVec, SwSplit, false>(outer_n / 2, inner_n, mem);

    mem = aligned_increment(
      mem, multi::fft_memsize<V, SwVec, SwSplit, false>(
        outer_n / 2, inner_n));

    if(do_create)
    {
      r->multidim_transform = fft_create<
        V, SwSrc, SwVec>(ndim - 1, dim + 1, mem);

      r->real_pass = &multi::real_pass<V, cf::Vec, true>;

      Int m =  r->outer_n / 2;
      compute_twiddle<V>(m, r->twiddle, r->twiddle + m);
    }

    mem = aligned_increment( 
      mem, fft_memsize<V, SwSrc, SwVec>(ndim - 1, dim + 1));
  }

  return Int(mem); 
}

template<typename V, typename SrcCf>
Int irfft_memsize(Int ndim_in, const Int* dim_in)
{
  return irfft_create_impl<false, V, SrcCf>(ndim_in, dim_in, nullptr);
}

template<typename V, typename SrcCf>
Irfft<typename V::T>* irfft_create(Int ndim_in, const Int* dim_in, void* mem)
{
  irfft_create_impl<true, V, SrcCf>(ndim_in, dim_in, mem);
  return (Irfft<typename V::T>*) mem;
}


template<typename T>
void irfft(Irfft<T>* s, const T* src_re, const T* src_im, T* dst)
{
  if(s->onedim_transform)
    return onedim::irfft(s->onedim_transform, src_re, src_im, dst);
  
  const Int working_idx_ratio = s->multidim_transform->working_idx_ratio;
  const Int nbits = log2(s->outer_n / 2);
  for(Int i = 0; i < s->outer_n / 2 + 1 ; i++)
  {
    Int src_off = i * s->inner_n * s->src_idx_ratio;
    fft_impl(
      0, s->multidim_transform,
      src_re + src_off, src_im + src_off,
      s->multidim_transform->working_re, s->multidim_transform->working_im,
      s->working0 + i * s->inner_n * working_idx_ratio, (T*) nullptr,
      false);
  }

  s->real_pass(s->outer_n, s->inner_n, s->twiddle, s->working0, nullptr);

  s->last_transform->fun_ptr(
    s->last_transform,
    s->working0, s->working0 + s->outer_n / 2 * s->inner_n,
    dst, nullptr,
    false, true);
}

}

#endif
