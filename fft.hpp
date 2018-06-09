#include "fft.h"

namespace
{
namespace afft
{

template<typename T>
struct complex_transform { };

template<>
class complex_transform<float>
{
public:
  complex_transform(size_t ndim, const size_t* dim, void* mem = 0)
  {
    if(mem == 0)
    {
      size_t align_mask = alignment - 1;
      allocated = new char[afft_32_c_memsize(ndim, dim) + align_mask];
      mem = (void*)((size_t(allocated) + align_mask) & ~align_mask);
    }

    state = afft32_c_create(ndim, dim, mem);
  }

  complex_transform(const complex_transform&) = delete;

  complex_transform(complex_transform&& other)
  {
    delete [] allocated;
    allocated = other->allocated;
    state = other->state;
    other->allocated = NULL;
    other->state = NULL;
  }

  ~complex_transform() { delete [] allocated; }

  void operator()(
    const float* src_re, const float* src_im,
    float* dst_re, float* dst_im)
  {
    afft_32_c_transform(state, src_re, src_im, dst_re, dst_im);
  }

private:
  char* allocated = NULL;
  afft_32_c_type* state = NULL;
};

}
}
