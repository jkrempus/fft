#include "fft.h"

#include "fft_internal.hpp"

typedef AvxFloat V;
typedef complex_format::Split Cf;

extern "C"
{

size_t afft_32_c_memsize(size_t ndim, const size_t* dim)
{
  return Uint(fft_memsize<V, Cf, Cf>(Int(ndim), (const Int*) dim));
}

afft_32_c_type* afft_32_c_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_32_c_type*) fft_create<V, Cf, Cf>(Int(ndim), (Int*) dim, mem);
}

void afft_32_c_transform(afft_32_c_type* state,
  const float* src_re, const float* src_im, float* dst_re, float* dst_im)
{
  fft((Fft<float>*) state, src_re, src_im, dst_re, dst_im);
}

size_t afft_32_ci_memsize(size_t ndim, const size_t* dim)
{
  return Uint(ifft_memsize<V, Cf, Cf>(Int(ndim), (const Int*) dim));
}

afft_32_ci_type* afft_32_ci_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_32_ci_type*) ifft_create<V, Cf, Cf>(Int(ndim), (Int*) dim, mem);
}

void afft_32_ci_transform(afft_32_ci_type* state,
  const float* src_re, const float* src_im, float* dst_re, float* dst_im)
{
  ifft((Ifft<float>*) state, src_re, src_im, dst_re, dst_im);
}

size_t afft_32_r_memsize(size_t ndim, const size_t* dim)
{
  return Uint(rfft_memsize<V, Cf>(Int(ndim), (const Int*) dim));
}

afft_32_r_type* afft_32_r_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_32_r_type*) rfft_create<V, Cf>(Int(ndim), (Int*) dim, mem);
}

void afft_32_r_transform(afft_32_r_type* state,
  const float* src, float* dst_re, float* dst_im)
{
  rfft((Rfft<float>*) state, src, dst_re, dst_im);
}

size_t afft_32_ri_memsize(size_t ndim, const size_t* dim)
{
  return Uint(irfft_memsize<V, Cf>(Int(ndim), (const Int*) dim));
}

afft_32_ri_type* afft_32_ri_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_32_ri_type*) irfft_create<V, Cf>(Int(ndim), (Int*) dim, mem);
}

void afft_32_ri_transform(afft_32_ri_type* state,
  const float* src_re, const float* src_im, float* dst)
{
  irfft((Irfft<float>*) state, src_re, src_im, dst);
}

size_t afft_32_align_size(size_t sz) { return align_size<float>(sz); }

const size_t afft_alignment = 64;

}
