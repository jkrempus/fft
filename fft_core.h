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

template<typename V>
Int fft_memsize(Int ndim_in, const Int* dim_in)
{
  Int dim[maxdim];
  Int ndim;
  remove_ones(dim_in, ndim_in, dim, ndim);

  VEC_TYPEDEFS(V);
  if(V::vec_size > 1 && dim[ndim - 1] < 2 * V::vec_size)
    return fft_memsize<Scalar<T>>(ndim, dim);

  Int r = align_size(sizeof(Fft<T>));

  Int working_size = 2 * sizeof(T) * product(dim, ndim);
  r = align_size(r + working_size);

  for(Int i = 0; i < ndim - 1; i++)
    r = align_size(r + multi::fft_memsize<V>(dim[i]));

  r = align_size(r + onedim::fft_memsize<V>(dim[ndim - 1]));

  return r;
}

template<typename V, typename SrcCf, typename DstCf>
Fft<typename V::T>* fft_create(Int ndim_in, const Int* dim_in, void* mem)
{
  Int dim[maxdim];
  Int ndim;
  remove_ones(dim_in, ndim_in, dim, ndim);

  VEC_TYPEDEFS(V);
  if(V::vec_size > 1 && dim[ndim - 1] < 2 * V::vec_size)
    return fft_create<Scalar<T>, SrcCf, DstCf>(ndim, dim, mem);

  auto s = (Fft<typename V::T>*) mem;
  s->ndim = ndim;
  s->working_idx_ratio = cf::Vec::idx_ratio;
  s->dst_idx_ratio = DstCf::idx_ratio;
  mem = (void*) align_size(Uint(mem) + sizeof(Fft<T>));
  s->working = (T*) mem;
 
  s->num_elements = product(dim, ndim);
  Int working_size = 2 * sizeof(T) * s->num_elements;
  mem = (void*) align_size(Uint(mem) + working_size);

  if(ndim == 1)
    s->last_transform = onedim::fft_create<V, SrcCf, DstCf>(dim[ndim - 1], mem);
  else
  {
    for(Int i = 0; i < ndim - 1; i++)
    {
      Int m = product(dim + i + 1, dim + ndim);

      if(i == 0)
        s->transforms[i] = multi::fft_create<V, SrcCf, cf::Vec, true>(
          dim[i], m, mem);
      else
        s->transforms[i] = multi::fft_create<V, cf::Vec, cf::Vec, true>(
          dim[i], m, mem);

      mem = (void*) align_size(Uint(mem) + multi::fft_memsize<V>(dim[i]));
    }

    s->last_transform = onedim::fft_create<V, cf::Vec, DstCf>(dim[ndim - 1], mem);
  }

  return s;
}

template<typename T>
void fft_impl(
  Int idim, Fft<T>* s, Int im_off, T* src, T* working, T* dst,
  bool interleaved_src_rows)
{
  ASSERT(idim < s->ndim);
  if(idim == s->ndim - 1) onedim::fft_impl(s->last_transform, im_off, src, dst);
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

template<typename V>
Int ifft_memsize(Int ndim, const Int* dim)
{
  return fft_memsize<V>(ndim, dim); 
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

template<typename V>
Int rfft_memsize(Int ndim_in, const Int* dim_in)
{
  Int dim[maxdim];
  Int ndim;
  remove_ones(dim_in, ndim_in, dim, ndim);

  VEC_TYPEDEFS(V)
  if(V::vec_size != 1 && dim[ndim - 1] < 2 * V::vec_size)
    return rfft_memsize<Scalar<T>>(ndim, dim);

  Int r = 0;
  r = align_size(r + sizeof(Rfft<T>));
  if(ndim == 1)
    r = align_size(r + onedim::rfft_memsize<V>(dim[0]));
  else
  {
    r = align_size(r + sizeof(T) * dim[0]);
    r = align_size(r + sizeof(T) * 2 * rfft_im_off<T>(ndim, dim));
    r = align_size(r + multi::fft_memsize<V>(dim[0] / 2));
    r = align_size(r + fft_memsize<V>(ndim - 1, dim + 1));
  }
  return r;
}

template<typename V, typename DstCf>
Rfft<typename V::T>* rfft_create(Int ndim_in, const Int* dim_in, void* mem)
{
  Int dim[maxdim];
  Int ndim;
  remove_ones(dim_in, ndim_in, dim, ndim);

  VEC_TYPEDEFS(V)
  if(V::vec_size != 1 && dim[ndim - 1] < 2 * V::vec_size)
    return rfft_create<Scalar<T>, DstCf>(ndim, dim, mem);

  auto r = (Rfft<T>*) mem;
  mem = (void*) align_size(Uint(mem) + sizeof(Rfft<T>)); 
  if(ndim == 1)
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
  else
  {
    r->dst_idx_ratio = DstCf::idx_ratio;
    r->outer_n = dim[0];
    r->inner_n = product(dim + 1, dim + ndim);
    r->onedim_transform = nullptr;
    r->im_off = rfft_im_off<T>(ndim, dim);
    r->twiddle = (T*) mem;
    mem = (void*) align_size(Uint(mem) + sizeof(T) * dim[0]);
    r->working0 = (T*) mem;
    mem = (void*) align_size(Uint(mem) + 2 * sizeof(T) * r->im_off);
    r->first_transform = multi::fft_create<V, cf::Split, cf::Vec, false>(
      r->outer_n / 2, r->inner_n, mem);

    mem = (void*) align_size(Uint(mem) + multi::fft_memsize<V>(dim[0] / 2));
    r->multidim_transform = fft_create<V, cf::Vec, DstCf>(ndim - 1, dim + 1, mem);
    r->real_pass = &multi::real_pass<V, cf::Vec, false>;
  
    Int m =  r->outer_n / 2;
    compute_twiddle(m, m, r->twiddle, r->twiddle + m);
  }
  
  return r;
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
  onedim::Rfft<T>* onedim_transform;
  Fft<T>* multidim_transform;
  multi::Fft<T>* last_transform;
  void (*real_pass)(Int n, Int m, T* twiddle, T* dst, Int dst_im_off);
};

template<typename V>
Int irfft_memsize(Int ndim, const Int* dim)
{
  return rfft_memsize<V>(ndim, dim);
}

template<typename V, typename SrcCf>
Irfft<typename V::T>* irfft_create(Int ndim_in, const Int* dim_in, void* mem)
{
  Int dim[maxdim];
  Int ndim;
  remove_ones(dim_in, ndim_in, dim, ndim);

  VEC_TYPEDEFS(V)
  if(V::vec_size != 1 && dim[ndim - 1] < 2 * V::vec_size)
    return irfft_create<Scalar<T>, SrcCf>(ndim, dim, mem);

  auto r = (Irfft<T>*) mem;
  mem = (void*) align_size(Uint(mem) + sizeof(Irfft<T>));
  
  if(ndim == 1)
  {
     r->onedim_transform = onedim::rfft_create<V, SrcCf>(dim[0], mem);
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
  else
  {
    r->src_idx_ratio = SrcCf::idx_ratio;
    r->outer_n = dim[0];
    r->inner_n = product(dim + 1, dim + ndim);
    r->onedim_transform = nullptr;
    r->im_off = rfft_im_off<T>(ndim, dim);
    r->twiddle = (T*) mem;
    mem = (void*) align_size(Uint(mem) + sizeof(T) * dim[0]);
    r->working0 = (T*) mem;
    mem = (void*) align_size(Uint(mem) + 2 * sizeof(T) * r->im_off);

    r->last_transform = multi::fft_create<
      V, cf::Swapped<cf::Vec>, cf::Swapped<cf::Split>, false>(
        r->outer_n / 2, r->inner_n, mem);

    mem = (void*) align_size(Uint(mem) + multi::fft_memsize<V>(dim[0] / 2));
    r->multidim_transform = fft_create<
      V, cf::Swapped<SrcCf>, cf::Swapped<cf::Vec>>(
        ndim - 1, dim + 1, mem);

    r->real_pass = &multi::real_pass<V, cf::Vec, true>;
  
    Int m =  r->outer_n / 2;
    compute_twiddle(m, m, r->twiddle, r->twiddle + m);
  }
  
  return r; 
}

template<typename T>
void irfft(Irfft<T>* s, T* src, T* dst)
{
  if(s->onedim_transform) return onedim::rfft(s->onedim_transform, src, dst);
  
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
