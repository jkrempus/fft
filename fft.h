#ifndef AFFT_NO_STDLIB
#include <stddef.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct afft_32_c_struct afft_32_c_type;
size_t afft_32_c_memsize(size_t ndim, const size_t* dim);
afft_32_c_type* afft_32_c_create(size_t ndim, const size_t* dim, void* mem);
void afft_32_c_transform(afft_32_c_type* state,
  const float* src_re, const float* src_im, float* dst_re, float* dst_im);

typedef struct afft_32_ci_struct afft_32_ci_type;
size_t afft_32_ci_memsize(size_t ndim, const size_t* dim);
afft_32_ci_type* afft_32_ci_create(size_t ndim, const size_t* dim, void* mem);
void afft_32_ci_transform(afft_32_ci_type* state,
  const float* src_re, const float* src_im, float* dst_re, float* dst_im);

typedef struct afft_32_r_struct afft_32_r_type;
size_t afft_32_r_memsize(size_t ndim, const size_t* dim);
afft_32_r_type* afft_32_r_create(size_t ndim, const size_t* dim, void* mem);
void afft_32_r_transform(afft_32_r_type* state,
  const float* src, float* dst_re, float* dst_im);

typedef struct afft_32_ri_struct afft_32_ri_type;
size_t afft_32_ri_memsize(size_t ndim, const size_t* dim);
afft_32_ri_type* afft_32_ri_create(size_t ndim, const size_t* dim, void* mem);
void afft_32_ri_transform(afft_32_ri_type* state,
  const float* src_re, const float* src_im, float* dst);

size_t afft_32_align_size(size_t sz);

#ifdef __cplusplus
}
#endif
