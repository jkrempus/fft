#ifndef AFFT_NO_STDLIB
#include <stddef.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

class Afft32C;
size_t afft32_c_memsize(size_t ndim, const size_t* dim);
Afft32C* afft32_c_create(size_t ndim, const size_t* dim, void* mem);
void afft32_c_transform(Afft32C* state,
  const float* src_re, const float* src_im, float* dst_re, float* dst_im);

class Afft32CI;
size_t afft32_ci_memsize(size_t ndim, const size_t* dim);
Afft32CI* afft32_ci_create(size_t ndim, const size_t* dim, void* mem);
void afft32_ci_transform(Afft32CI* state,
  const float* src_re, const float* src_im, float* dst_re, float* dst_im);

class Afft32R;
size_t afft32_r_memsize(size_t ndim, const size_t* dim);
Afft32R* afft32_r_create(size_t ndim, const size_t* dim, void* mem);
void afft32_r_transform(Afft32R* state,
  const float* src, float* dst_re, float* dst_im);

class Afft32RI;
size_t afft32_ri_memsize(size_t ndim, const size_t* dim);
Afft32RI* afft32_ri_create(size_t ndim, const size_t* dim, void* mem);
void afft32_ri_transform(Afft32RI* state,
  const float* src_re, const float* src_im, float* dst);

size_t afft32_align_size(size_t sz);

#ifdef __cplusplus
}
#endif
