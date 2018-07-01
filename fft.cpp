#include "fft.h"

#include "fft_internal.hpp"
#include "sse.hpp"
#include "avx.hpp"

typedef AvxFloat VF;
//typedef SseFloat VF;
//typedef Scalar<float> VF;
typedef Scalar<double> VD;
typedef complex_format::Split Cf;

extern "C"
{

size_t afft_32_c_memsize(size_t ndim, const size_t* dim)
{
  return Uint(fft_memsize<VF, Cf, Cf>(Int(ndim), (const Int*) dim));
}

afft_32_c_type* afft_32_c_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_32_c_type*) fft_create<VF, Cf, Cf>(Int(ndim), (Int*) dim, mem);
}

void afft_32_c_transform(afft_32_c_type* state,
  const float* src_re, const float* src_im, float* dst_re, float* dst_im)
{
  fft((Fft<float>*) state, src_re, src_im, dst_re, dst_im);
}

size_t afft_32_ci_memsize(size_t ndim, const size_t* dim)
{
  return Uint(ifft_memsize<VF, Cf, Cf>(Int(ndim), (const Int*) dim));
}

afft_32_ci_type* afft_32_ci_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_32_ci_type*) ifft_create<VF, Cf, Cf>(Int(ndim), (Int*) dim, mem);
}

void afft_32_ci_transform(afft_32_ci_type* state,
  const float* src_re, const float* src_im, float* dst_re, float* dst_im)
{
  ifft((Ifft<float>*) state, src_re, src_im, dst_re, dst_im);
}

size_t afft_32_r_memsize(size_t ndim, const size_t* dim)
{
  return Uint(rfft_memsize<VF, Cf>(Int(ndim), (const Int*) dim));
}

afft_32_r_type* afft_32_r_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_32_r_type*) rfft_create<VF, Cf>(Int(ndim), (Int*) dim, mem);
}

void afft_32_r_transform(afft_32_r_type* state,
  const float* src, float* dst_re, float* dst_im)
{
  rfft((Rfft<float>*) state, src, dst_re, dst_im);
}

size_t afft_32_ri_memsize(size_t ndim, const size_t* dim)
{
  return Uint(irfft_memsize<VF, Cf>(Int(ndim), (const Int*) dim));
}

afft_32_ri_type* afft_32_ri_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_32_ri_type*) irfft_create<VF, Cf>(Int(ndim), (Int*) dim, mem);
}

void afft_32_ri_transform(afft_32_ri_type* state,
  const float* src_re, const float* src_im, float* dst)
{
  irfft((Irfft<float>*) state, src_re, src_im, dst);
}

size_t afft_32_align_size(size_t sz) { return align_size<float>(sz); }

size_t afft_64_c_memsize(size_t ndim, const size_t* dim)
{
  return Uint(fft_memsize<VD, Cf, Cf>(Int(ndim), (const Int*) dim));
}

afft_64_c_type* afft_64_c_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_64_c_type*) fft_create<VD, Cf, Cf>(Int(ndim), (Int*) dim, mem);
}

void afft_64_c_transform(afft_64_c_type* state,
  const double* src_re, const double* src_im, double* dst_re, double* dst_im)
{
  fft((Fft<double>*) state, src_re, src_im, dst_re, dst_im);
}

size_t afft_64_ci_memsize(size_t ndim, const size_t* dim)
{
  return Uint(ifft_memsize<VD, Cf, Cf>(Int(ndim), (const Int*) dim));
}

afft_64_ci_type* afft_64_ci_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_64_ci_type*) ifft_create<VD, Cf, Cf>(Int(ndim), (Int*) dim, mem);
}

void afft_64_ci_transform(afft_64_ci_type* state,
  const double* src_re, const double* src_im, double* dst_re, double* dst_im)
{
  ifft((Ifft<double>*) state, src_re, src_im, dst_re, dst_im);
}

size_t afft_64_r_memsize(size_t ndim, const size_t* dim)
{
  return Uint(rfft_memsize<VD, Cf>(Int(ndim), (const Int*) dim));
}

afft_64_r_type* afft_64_r_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_64_r_type*) rfft_create<VD, Cf>(Int(ndim), (Int*) dim, mem);
}

void afft_64_r_transform(afft_64_r_type* state,
  const double* src, double* dst_re, double* dst_im)
{
  rfft((Rfft<double>*) state, src, dst_re, dst_im);
}

size_t afft_64_ri_memsize(size_t ndim, const size_t* dim)
{
  return Uint(irfft_memsize<VD, Cf>(Int(ndim), (const Int*) dim));
}

afft_64_ri_type* afft_64_ri_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_64_ri_type*) irfft_create<VD, Cf>(Int(ndim), (Int*) dim, mem);
}

void afft_64_ri_transform(afft_64_ri_type* state,
  const double* src_re, const double* src_im, double* dst)
{
  irfft((Irfft<double>*) state, src_re, src_im, dst);
}

size_t afft_64_align_size(size_t sz) { return align_size<double>(sz); }

const size_t afft_alignment = 64;

}
