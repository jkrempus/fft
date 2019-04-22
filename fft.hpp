#include "fft.h"

namespace afft
{

namespace detail
{
template<typename T, typename AfftType, int format>
struct members
{
  char* allocated = NULL;
  AfftType* state = NULL;

  members(const members&) = delete;

  members() = default;

  members(
    size_t (*memsize)(size_t ndim, const size_t* dim, int fmt, int impl),
    AfftType* (*create)(size_t ndim, const size_t* dim, int fmt, void* mem, int impl),
    size_t ndim, const size_t* dim, void* mem, int impl)
  {
    if(mem == 0)
    {
      size_t align_mask = afft_alignment - 1;
      size_t size = memsize(ndim, dim, format, impl);
      if(size == 0) return;

      allocated = new char[size + align_mask];
      mem = (void*)((size_t(allocated) + align_mask) & ~align_mask);
    }

    state = create(ndim, dim, format, mem, impl);
  }

  members(members&& other)
  {
    allocated = other.allocated;
    state = other.state;
    other.allocated = NULL;
    other.state = NULL;
  }

  ~members() { delete [] allocated; }
};
}

template<typename T, int format = afft_split>
struct complex_transform { };

template<int format>
class complex_transform<float, format>
{
public:
  complex_transform() = default;

  complex_transform(
    size_t ndim, const size_t* dim, void* mem = 0, int impl = afft_auto)
  : m(afft_32_c_memsize, afft_32_c_create, ndim, dim, mem, impl) { }

  void operator()(
    const float* src_re, const float* src_im,
    float* dst_re, float* dst_im)
  {
    afft_32_c_transform(m.state, src_re, src_im, dst_re, dst_im);
  }

  operator bool() const { return bool(m.state); }

private:
  detail::members<float, afft_32_c_type, format> m;
};

template<int format>
class complex_transform<double, format>
{
public:
  complex_transform() = default;

  complex_transform(
    size_t ndim, const size_t* dim, void* mem = 0, int impl = afft_auto)
  : m(afft_64_c_memsize, afft_64_c_create, ndim, dim, mem, impl) { }

  void operator()(
    const double* src_re, const double* src_im,
    double* dst_re, double* dst_im)
  {
    afft_64_c_transform(m.state, src_re, src_im, dst_re, dst_im);
  }

  operator bool() const { return bool(m.state); }

private:
  detail::members<double, afft_64_c_type, format> m;
};

template<typename T, int format = afft_split>
struct inverse_complex_transform { };

template<int format>
class inverse_complex_transform<float, format>
{
public:
  inverse_complex_transform() = default;

  inverse_complex_transform(
    size_t ndim, const size_t* dim, void* mem = 0, int impl = afft_auto)
  : m(afft_32_ci_memsize, afft_32_ci_create, ndim, dim, mem, impl) { }

  void operator()(
    const float* src_re, const float* src_im,
    float* dst_re, float* dst_im)
  {
    afft_32_ci_transform(m.state, src_re, src_im, dst_re, dst_im);
  }

  operator bool() const { return bool(m.state); }

private:
  detail::members<float, afft_32_ci_type, format> m;
};

template<int format>
class inverse_complex_transform<double, format>
{
public:
  inverse_complex_transform() = default;

  inverse_complex_transform(
    size_t ndim, const size_t* dim, void* mem = 0, int impl = afft_auto)
  : m(afft_64_ci_memsize, afft_64_ci_create, ndim, dim, mem, impl) { }

  void operator()(
    const double* src_re, const double* src_im,
    double* dst_re, double* dst_im)
  {
    afft_64_ci_transform(m.state, src_re, src_im, dst_re, dst_im);
  }

  operator bool() const { return bool(m.state); }

private:
  detail::members<double, afft_64_ci_type, format> m;
};

template<typename T, int format = afft_split>
struct real_transform { };

template<int format>
class real_transform<float, format>
{
public:
  real_transform() = default;

  real_transform(
    size_t ndim, const size_t* dim, void* mem = 0, int impl = afft_auto)
  : m(afft_32_r_memsize, afft_32_r_create, ndim, dim, mem, impl) { }

  void operator()(const float* src, float* dst_re, float* dst_im)
  {
    afft_32_r_transform(m.state, src, dst_re, dst_im);
  }

  operator bool() const { return bool(m.state); }

private:
  detail::members<float, afft_32_r_type, format> m;
};

template<int format>
class real_transform<double, format>
{
public:
  real_transform() = default;

  real_transform(
    size_t ndim, const size_t* dim, void* mem = 0, int impl = afft_auto)
  : m(afft_64_r_memsize, afft_64_r_create, ndim, dim, mem, impl) { }

  void operator()(const double* src, double* dst_re, double* dst_im)
  {
    afft_64_r_transform(m.state, src, dst_re, dst_im);
  }

  operator bool() const { return bool(m.state); }

private:
  detail::members<double, afft_64_r_type, format> m;
};

template<typename T, int format = afft_split>
struct inverse_real_transform { };

template<int format>
class inverse_real_transform<float, format>
{
public:
  inverse_real_transform() = default;

  inverse_real_transform(
    size_t ndim, const size_t* dim, void* mem = 0, int impl = afft_auto)
  : m(afft_32_ri_memsize, afft_32_ri_create, ndim, dim, mem, impl) { }

  void operator()(const float* src_re, const float* src_im, float* dst)
  {
    afft_32_ri_transform(m.state, src_re, src_im, dst);
  }

  operator bool() const { return bool(m.state); }

private:
  detail::members<float, afft_32_ri_type, format> m;
};

template<int format>
class inverse_real_transform<double, format>
{
public:
  inverse_real_transform() = default;

  inverse_real_transform(
    size_t ndim, const size_t* dim, void* mem = 0, int impl = afft_auto)
  : m(afft_64_ri_memsize, afft_64_ri_create, ndim, dim, mem, impl) { }

  void operator()(const double* src_re, const double* src_im, double* dst)
  {
    afft_64_ri_transform(m.state, src_re, src_im, dst);
  }

  operator bool() const { return bool(m.state); }

private:
  detail::members<double, afft_64_ri_type, format> m;
};

}
