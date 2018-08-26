#ifndef AFFT_NO_STDLIB
#include <stddef.h>
#endif

enum
{
  afft_auto = 0,
  afft_scalar = 1,
  afft_sse2 = 2,
  afft_avx2 = 3,
  afft_neon = 4,
};

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct afft_32_c_struct afft_32_c_type;
size_t afft_32_c_memsize(size_t ndim, const size_t* dim, int impl);
afft_32_c_type* afft_32_c_create(
  size_t ndim, const size_t* dim, void* mem, int impl);
void afft_32_c_transform(afft_32_c_type* state,
  const float* src_re, const float* src_im, float* dst_re, float* dst_im);

typedef struct afft_32_ci_struct afft_32_ci_type;
size_t afft_32_ci_memsize(size_t ndim, const size_t* dim, int impl);
afft_32_ci_type* afft_32_ci_create(
  size_t ndim, const size_t* dim, void* mem, int impl);
void afft_32_ci_transform(afft_32_ci_type* state,
  const float* src_re, const float* src_im, float* dst_re, float* dst_im);

typedef struct afft_32_r_struct afft_32_r_type;
size_t afft_32_r_memsize(size_t ndim, const size_t* dim, int impl);
afft_32_r_type* afft_32_r_create(
  size_t ndim, const size_t* dim, void* mem, int impl);
void afft_32_r_transform(afft_32_r_type* state,
  const float* src, float* dst_re, float* dst_im);

typedef struct afft_32_ri_struct afft_32_ri_type;
size_t afft_32_ri_memsize(size_t ndim, const size_t* dim, int impl);
afft_32_ri_type* afft_32_ri_create(
  size_t ndim, const size_t* dim, void* mem, int impl);
void afft_32_ri_transform(afft_32_ri_type* state,
  const float* src_re, const float* src_im, float* dst);

size_t afft_32_align_size(size_t sz);

typedef struct afft_64_c_struct afft_64_c_type;
size_t afft_64_c_memsize(size_t ndim, const size_t* dim, int impl);
afft_64_c_type* afft_64_c_create(
  size_t ndim, const size_t* dim, void* mem, int impl);
void afft_64_c_transform(afft_64_c_type* state,
  const double* src_re, const double* src_im, double* dst_re, double* dst_im);

typedef struct afft_64_ci_struct afft_64_ci_type;
size_t afft_64_ci_memsize(size_t ndim, const size_t* dim, int impl);
afft_64_ci_type* afft_64_ci_create(
  size_t ndim, const size_t* dim, void* mem, int impl);
void afft_64_ci_transform(afft_64_ci_type* state,
  const double* src_re, const double* src_im, double* dst_re, double* dst_im);

typedef struct afft_64_r_struct afft_64_r_type;
size_t afft_64_r_memsize(size_t ndim, const size_t* dim, int impl);
afft_64_r_type* afft_64_r_create(
  size_t ndim, const size_t* dim, void* mem, int impl);
void afft_64_r_transform(afft_64_r_type* state,
  const double* src, double* dst_re, double* dst_im);

typedef struct afft_64_ri_struct afft_64_ri_type;
size_t afft_64_ri_memsize(size_t ndim, const size_t* dim, int impl);
afft_64_ri_type* afft_64_ri_create(
  size_t ndim, const size_t* dim, void* mem, int impl);
void afft_64_ri_transform(afft_64_ri_type* state,
  const double* src_re, const double* src_im, double* dst);

size_t afft_64_align_size(size_t sz);

int afft_supported(int impl);

extern const size_t afft_alignment;

#ifdef __cplusplus
}
#endif
