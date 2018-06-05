#include "fft.h"

#include "fft_internal.h"

typedef AvxFloat V;
typedef complex_format::Split Cf;

extern "C"
{

size_t afft32_c_memsize(size_t ndim, const size_t* dim)
{
  return Uint(fft_memsize<V, Cf, Cf>(Int(ndim), (const Int*) dim));
}

Afft32C* afft32_c_create(size_t ndim, const size_t* dim, void* mem)
{
  return (Afft32C*) fft_create<V, Cf, Cf>(Int(ndim), (Int*) dim, mem);
}

void afft32_c_transform(Afft32C* state,
  const float* src_re, const float* src_im, float* dst_re, float* dst_im)
{
  fft((Fft<float>*) state, src_re, src_im, dst_re, dst_im);
}

size_t afft32_ci_memsize(size_t ndim, const size_t* dim)
{
  return Uint(ifft_memsize<V, Cf, Cf>(Int(ndim), (const Int*) dim));
}

Afft32CI* afft32_ci_create(size_t ndim, const size_t* dim, void* mem)
{
  return (Afft32CI*) ifft_create<V, Cf, Cf>(Int(ndim), (Int*) dim, mem);
}

void afft32_ci_transform(Afft32CI* state,
  const float* src_re, const float* src_im, float* dst_re, float* dst_im)
{
  ifft((Ifft<float>*) state, src_re, src_im, dst_re, dst_im);
}

size_t afft32_r_memsize(size_t ndim, const size_t* dim)
{
  return Uint(rfft_memsize<V, Cf>(Int(ndim), (const Int*) dim));
}

Afft32R* afft32_r_create(size_t ndim, const size_t* dim, void* mem)
{
  return (Afft32R*) rfft_create<V, Cf>(Int(ndim), (Int*) dim, mem);
}

void afft32_r_transform(Afft32R* state,
  const float* src, float* dst_re, float* dst_im)
{
  rfft((Rfft<float>*) state, src, dst_re, dst_im);
}

size_t afft32_ri_memsize(size_t ndim, const size_t* dim)
{
  return Uint(irfft_memsize<V, Cf>(Int(ndim), (const Int*) dim));
}

Afft32RI* afft32_ri_create(size_t ndim, const size_t* dim, void* mem)
{
  return (Afft32RI*) irfft_create<V, Cf>(Int(ndim), (Int*) dim, mem);
}

void afft32_ri_transform(Afft32RI* state,
  const float* src_re, const float* src_im, float* dst)
{
  irfft((Irfft<float>*) state, src_re, src_im, dst);
}

size_t afft32_align_size(size_t sz) { return align_size<float>(sz); }

}
