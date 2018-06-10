#include "fft.h"

namespace afft
{

namespace detail
{
template<typename T, typename AfftType>
struct members
{
  char* allocated = NULL;
  AfftType* state = NULL;

  members(const members&) = delete;

  members(
    size_t (*memsize)(size_t ndim, const size_t* dim),
    AfftType* (*create)(size_t ndim, const size_t* dim, void* mem),
    size_t ndim, const size_t* dim, void* mem)
  {
    if(mem == 0)
    {
      size_t align_mask = afft_alignment - 1;
      allocated = new char[memsize(ndim, dim) + align_mask];
      mem = (void*)((size_t(allocated) + align_mask) & ~align_mask);
    }

    state = create(ndim, dim, mem);
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

template<typename T>
struct complex_transform { };

template<>
class complex_transform<float>
{
public:
  complex_transform(size_t ndim, const size_t* dim, void* mem = 0)
  : m(afft_32_c_memsize, afft_32_c_create, ndim, dim, mem) { }

  void operator()(
    const float* src_re, const float* src_im,
    float* dst_re, float* dst_im)
  {
    afft_32_c_transform(m.state, src_re, src_im, dst_re, dst_im);
  }

private:
  detail::members<float, afft_32_c_type> m;
};

template<typename T>
struct inverse_complex_transform { };

template<>
class inverse_complex_transform<float>
{
public:
  inverse_complex_transform(size_t ndim, const size_t* dim, void* mem = 0)
  : m(afft_32_ci_memsize, afft_32_ci_create, ndim, dim, mem) { }

  void operator()(
    const float* src_re, const float* src_im,
    float* dst_re, float* dst_im)
  {
    afft_32_ci_transform(m.state, src_re, src_im, dst_re, dst_im);
  }

private:
  detail::members<float, afft_32_ci_type> m;
};

template<typename T>
struct real_transform { };

template<>
class real_transform<float>
{
public:
  real_transform(size_t ndim, const size_t* dim, void* mem = 0)
  : m(afft_32_r_memsize, afft_32_r_create, ndim, dim, mem) { }

  void operator()(const float* src, float* dst_re, float* dst_im)
  {
    afft_32_r_transform(m.state, src, dst_re, dst_im);
  }

private:
  detail::members<float, afft_32_r_type> m;
};

template<typename T>
struct inverse_real_transform { };

template<>
class inverse_real_transform<float>
{
public:
  inverse_real_transform(size_t ndim, const size_t* dim, void* mem = 0)
  : m(afft_32_ri_memsize, afft_32_ri_create, ndim, dim, mem) { }

  void operator()(const float* src_re, const float* src_im, float* dst)
  {
    afft_32_ri_transform(m.state, src_re, src_im, dst);
  }

private:
  detail::members<float, afft_32_ri_type> m;
};

}
