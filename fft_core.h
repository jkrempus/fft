#ifndef FFT_CORE_H
#define FFT_CORE_H

#include "common.h"
#include "onedim.h"
#include "multi.h"

const Int maxdim = 64;

template<typename T>
struct Fft
{
  Int ndim;
  Int num_elements;
  Int working_idx_ratio;
  Int dst_idx_ratio;
  T* working;
  onedim::Fft<T>* last_transform;
  multi::Fft<T>* transforms[maxdim];
};

template<bool do_create, typename V, typename SrcCf, typename DstCf>
Int fft_create_impl(Int ndim_in, const Int* dim_in, void* mem)
{
  Int dim[maxdim];
  Int ndim;
  remove_ones(dim_in, ndim_in, dim, ndim);
  Int num_elements = product(dim, ndim);

  VEC_TYPEDEFS(V);
  if(V::vec_size > 1 && dim[ndim - 1] < 2 * V::vec_size)
    return fft_create_impl<do_create, Scalar<T>, SrcCf, DstCf>(ndim, dim, mem);

  auto s = (Fft<typename V::T>*) mem;
  mem = aligned_increment(mem, sizeof(Fft<T>));
 
  if(do_create)
  {
    s->ndim = ndim;
    s->working_idx_ratio = cf::Vec::idx_ratio;
    s->dst_idx_ratio = DstCf::idx_ratio;
    s->working = (T*) mem;
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
          s->transforms[i] = multi::fft_create<V, SrcCf, cf::Vec, true>(
            dim[i], m, mem);

        mem = aligned_increment(mem, multi::fft_memsize<V>(dim[i]));
      }
      else
      {
        if(do_create)
          s->transforms[i] = multi::fft_create<V, cf::Vec, cf::Vec, true>(
            dim[i], m, mem);

        mem = aligned_increment(mem, multi::fft_memsize<V>(dim[i]));
      }
    }

    if(do_create)
      s->last_transform =
        onedim::fft_create<V, cf::Vec, DstCf>(dim[ndim - 1], mem);

    mem = aligned_increment(
      mem, onedim::fft_memsize<V, cf::Vec, DstCf>(dim[ndim - 1]));
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
  Int idim, Fft<T>* s, Int im_off, T* src, T* working, T* dst,
  bool interleaved_src_rows)
{
  ASSERT(idim < s->ndim);
  if(idim == s->ndim - 1)
    onedim::fft_impl(s->last_transform, src, im_off, dst, im_off);
  else
  {
    s->transforms[idim]->fun_ptr(
      s->transforms[idim], src, working, im_off, interleaved_src_rows, false);

    Int m = s->transforms[idim]->m;
    Int n = s->transforms[idim]->n;
    for(BitReversed br(n); br.i < n; br.advance())
    {
      auto next_src = working + br.i * m * s->working_idx_ratio;
      auto next_dst = dst + br.br * m * s->dst_idx_ratio;
      fft_impl(idim + 1, s, im_off, next_src, next_src, next_dst,
        interleaved_src_rows);
    }
  }
}

template<typename T>
void fft(Fft<T>* state, T* src, T* dst)
{
  fft_impl<T>(0, state, state->num_elements, src, state->working, dst, false);
}

template<typename T>
struct Ifft
{
  Fft<T> state;
};

template<typename T>
void ifft(Ifft<T>* state, T* src, T* dst)
{
  auto s = &state->state;
  fft_impl<T>(0, s, s->num_elements, src, s->working, dst, false);
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
  void (*real_pass)(Int n, Int m, T* twiddle, T* dst, Int dst_im_off);
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
  remove_ones(dim_in, ndim_in, dim, ndim);

  VEC_TYPEDEFS(V)
  if(V::vec_size != 1 && dim[ndim - 1] < 2 * V::vec_size)
    return rfft_create_impl<do_create, Scalar<T>, DstCf>(ndim, dim, mem);

  auto r = (Rfft<T>*) mem;
  mem = aligned_increment(mem, sizeof(Rfft<T>));

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
    if(do_create)
    {
      r->dst_idx_ratio = DstCf::idx_ratio;
      r->outer_n = dim[0];
      r->inner_n = product(dim + 1, dim + ndim);
      r->onedim_transform = nullptr;
      r->im_off = rfft_im_off<T>(ndim, dim);
      r->twiddle = (T*) mem;
    }

    mem = aligned_increment(mem, sizeof(T) * dim[0]);

    if(do_create) r->working0 = (T*) mem;

    mem = aligned_increment(mem, 2 * sizeof(T) * r->im_off);
    
    if(do_create)
      r->first_transform = multi::fft_create<V, cf::Split, cf::Vec, false>(
        r->outer_n / 2, r->inner_n, mem);

    mem = aligned_increment(mem, multi::fft_memsize<V>(dim[0] / 2));

    if(do_create)
    {
      r->multidim_transform =
        fft_create<V, cf::Vec, DstCf>(ndim - 1, dim + 1, mem);

      r->real_pass = &multi::real_pass<V, cf::Vec, false>;
  
      Int m =  r->outer_n / 2;
      compute_twiddle(m, m, r->twiddle, r->twiddle + m);
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
void rfft(Rfft<T>* s, T* src, T* dst)
{
  if(s->onedim_transform) return onedim::rfft(s->onedim_transform, src, dst);

  s->first_transform->fun_ptr(
    s->first_transform,
    src,
    s->working0, s->outer_n / 2 * s->inner_n,
    true,
    false);

  s->real_pass(s->outer_n, s->inner_n, s->twiddle, s->working0, 0);

  const Int working_idx_ratio = 2; // because we have cf::Vec in working
  const Int nbits = log2(s->outer_n / 2);
  for(Int i = 0; i < s->outer_n / 2 + 1 ; i++)
    fft_impl(
      0,
      s->multidim_transform,
      s->im_off,
      s->working0 + i * s->inner_n * working_idx_ratio,
      s->multidim_transform->working,
      dst + i * s->inner_n * s->dst_idx_ratio,
      false);
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
  void (*real_pass)(Int n, Int m, T* twiddle, T* dst, Int dst_im_off);
};

template<bool do_create, typename V, typename SrcCf>
Int irfft_create_impl(Int ndim_in, const Int* dim_in, void* mem)
{
  Int dim[maxdim];
  Int ndim;
  remove_ones(dim_in, ndim_in, dim, ndim);

  VEC_TYPEDEFS(V)
  if(V::vec_size != 1 && dim[ndim - 1] < 2 * V::vec_size)
    return irfft_create_impl<do_create, Scalar<T>, SrcCf>(ndim, dim, mem);

  auto r = (Irfft<T>*) mem;
  mem = aligned_increment(mem, sizeof(Irfft<T>));
  
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
    if(do_create)
    {
      r->src_idx_ratio = SrcCf::idx_ratio;
      r->outer_n = dim[0];
      r->inner_n = product(dim + 1, dim + ndim);
      r->onedim_transform = nullptr;
      r->im_off = rfft_im_off<T>(ndim, dim);
      r->twiddle = (T*) mem;
    }

    mem = aligned_increment(mem, sizeof(T) * dim[0]);
    
    if(do_create) r->working0 = (T*) mem;
    mem = aligned_increment(mem, 2 * sizeof(T) * r->im_off);

    if(do_create)
      r->last_transform = multi::fft_create<
        V, cf::Swapped<cf::Vec>, cf::Swapped<cf::Split>, false>(
          r->outer_n / 2, r->inner_n, mem);

    mem = aligned_increment(mem, multi::fft_memsize<V>(dim[0] / 2));

    if(do_create)
    {
      r->multidim_transform = fft_create<
        V, cf::Swapped<SrcCf>, cf::Swapped<cf::Vec>>(
          ndim - 1, dim + 1, mem);

      r->real_pass = &multi::real_pass<V, cf::Vec, true>;

      Int m =  r->outer_n / 2;
      compute_twiddle(m, m, r->twiddle, r->twiddle + m);
    }

    mem = aligned_increment( 
      mem,
      fft_memsize<V, cf::Swapped<SrcCf>, cf::Swapped<cf::Vec>>(
        ndim - 1, dim + 1));
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
void irfft(Irfft<T>* s, T* src, T* dst)
{
  if(s->onedim_transform) return onedim::irfft(s->onedim_transform, src, dst);
  
  const Int working_idx_ratio = 2; // because we have cf::Vec in working
  const Int nbits = log2(s->outer_n / 2);
  for(Int i = 0; i < s->outer_n / 2 + 1 ; i++)
    fft_impl(
      0,
      s->multidim_transform,
      s->im_off,
      src + i * s->inner_n * s->src_idx_ratio,
      s->multidim_transform->working,
      s->working0 + i * s->inner_n * working_idx_ratio,
      false);

  s->real_pass(s->outer_n, s->inner_n, s->twiddle, s->working0, 0);

  s->last_transform->fun_ptr(
    s->last_transform, s->working0, dst, s->outer_n / 2 * s->inner_n,
    false, true);
}

#endif
