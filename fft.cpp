#include "fft.h"

#include "fft_internal.hpp"

namespace afft
{
  namespace detail
  {
    struct OpC;
    struct OpCI;
    struct OpR;
    struct OpRI;

    template<Int impl>
    bool impl_supported();

    template<Int impl, typename Op, typename T>
    Int memsize_impl(Int ndim, const Int* dim);

    template<Int impl, typename Op, typename T>
    void* create_impl(Int ndim, const Int* dim, void* mem);

#define DECLARE_IMPL_FOR_TYPE(IMPL, T) \
    template<> Int memsize_impl<IMPL, OpC, T>(Int, const Int*); \
    template<> Int memsize_impl<IMPL, OpCI, T>(Int, const Int*); \
    template<> Int memsize_impl<IMPL, OpR, T>(Int, const Int*); \
    template<> Int memsize_impl<IMPL, OpRI, T>(Int, const Int*); \
    template<> void* create_impl<IMPL, OpC, T>(Int, const Int*, void*); \
    template<> void* create_impl<IMPL, OpCI, T>(Int, const Int*, void*); \
    template<> void* create_impl<IMPL, OpR, T>(Int, const Int*, void*); \
    template<> void* create_impl<IMPL, OpRI, T>(Int, const Int*, void*);

#define DECLARE_IMPL(IMPL) \
    template<> bool impl_supported<IMPL>(); \
    DECLARE_IMPL_FOR_TYPE(IMPL, float) \
    DECLARE_IMPL_FOR_TYPE(IMPL, float)

    DECLARE_IMPL(afft_scalar)
    DECLARE_IMPL(afft_sse)
    DECLARE_IMPL(afft_avx)
    DECLARE_IMPL(afft_neon)
  }
}

namespace
{
  struct MemsizeTag{};
  struct CreateTag{};

  template<Int impl, typename Op, typename T>
  Int call_impl(const MemsizeTag&, Int ndim, const Int* dim)
  {
    return afft::detail::memsize_impl<impl, Op, T>(ndim, dim);
  }

  template<Int impl, typename Op, typename T>
  void* call_impl(const CreateTag&, Int ndim, const Int* dim, void* ptr)
  {
    return afft::detail::create_impl<impl, Op, T>(ndim, dim, ptr);
  }

  Int get_null_value(const MemsizeTag&){ return 0; }
  void* get_null_value(const CreateTag&){ return nullptr; }

  template<typename FnTag, typename Op, typename T, typename... Args>
  auto call_selected_impl(Int impl, Args... args)
  {
    if(impl == afft_auto)
    {
      //if(afft::detail::impl_supported<afft_neon>())
      //  return call_impl<afft_neon, Op, T>(FnTag(), args...);

      if(afft::detail::impl_supported<afft_avx>())
        return call_impl<afft_avx, Op, T>(FnTag(), args...);

      if(afft::detail::impl_supported<afft_sse>())
        return call_impl<afft_sse, Op, T>(FnTag(), args...);

      if(afft::detail::impl_supported<afft_scalar>())
        return call_impl<afft_scalar, Op, T>(FnTag(), args...);
    }
    else
    {
      switch(impl)
      {
        //case afft_neon: return call_impl<afft_neon, Op, T>(FnTag(), args...);
        case afft_avx: return call_impl<afft_avx, Op, T>(FnTag(), args...);
        case afft_sse: return call_impl<afft_sse, Op, T>(FnTag(), args...);
        case afft_scalar: return call_impl<afft_scalar, Op, T>(FnTag(), args...);
      }
    }
    
    return get_null_value(FnTag());
  }
}

extern "C"
{

size_t afft_32_c_memsize(size_t ndim, const size_t* dim)
{
  return Uint(call_selected_impl<MemsizeTag, afft::detail::OpC, float>(
    afft_auto, Int(ndim), (const Int*) dim));
}

afft_32_c_type* afft_32_c_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_32_c_type*)
    call_selected_impl<CreateTag, afft::detail::OpC, float>(
      afft_auto, Int(ndim), (const Int*) dim, mem);
}

void afft_32_c_transform(afft_32_c_type* state,
  const float* src_re, const float* src_im, float* dst_re, float* dst_im)
{
  fft((Fft<float>*) state, src_re, src_im, dst_re, dst_im);
}

size_t afft_32_ci_memsize(size_t ndim, const size_t* dim)
{
  return Uint(call_selected_impl<MemsizeTag, afft::detail::OpCI, float>(
    afft_auto, Int(ndim), (const Int*) dim));
}

afft_32_ci_type* afft_32_ci_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_32_ci_type*)
    call_selected_impl<CreateTag, afft::detail::OpCI, float>(
      afft_auto, Int(ndim), (const Int*) dim, mem);
}

void afft_32_ci_transform(afft_32_ci_type* state,
  const float* src_re, const float* src_im, float* dst_re, float* dst_im)
{
  ifft((Ifft<float>*) state, src_re, src_im, dst_re, dst_im);
}

size_t afft_32_r_memsize(size_t ndim, const size_t* dim)
{
  return Uint(call_selected_impl<MemsizeTag, afft::detail::OpR, float>(
    afft_auto, Int(ndim), (const Int*) dim));
}

afft_32_r_type* afft_32_r_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_32_r_type*)
    call_selected_impl<CreateTag, afft::detail::OpR, float>(
      afft_auto, Int(ndim), (const Int*) dim, mem);
}

void afft_32_r_transform(afft_32_r_type* state,
  const float* src, float* dst_re, float* dst_im)
{
  rfft((Rfft<float>*) state, src, dst_re, dst_im);
}

size_t afft_32_ri_memsize(size_t ndim, const size_t* dim)
{
  return Uint(call_selected_impl<MemsizeTag, afft::detail::OpRI, float>(
    afft_auto, Int(ndim), (const Int*) dim));
}

afft_32_ri_type* afft_32_ri_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_32_ri_type*)
    call_selected_impl<CreateTag, afft::detail::OpRI, float>(
      afft_auto, Int(ndim), (const Int*) dim, mem);
}

void afft_32_ri_transform(afft_32_ri_type* state,
  const float* src_re, const float* src_im, float* dst)
{
  irfft((Irfft<float>*) state, src_re, src_im, dst);
}

size_t afft_32_align_size(size_t sz) { return align_size<float>(sz); }

size_t afft_64_c_memsize(size_t ndim, const size_t* dim)
{
  return Uint(call_selected_impl<MemsizeTag, afft::detail::OpC, double>(
    afft_auto, Int(ndim), (const Int*) dim));
}

afft_64_c_type* afft_64_c_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_64_c_type*)
    call_selected_impl<CreateTag, afft::detail::OpC, double>(
      afft_auto, Int(ndim), (const Int*) dim, mem);
}

void afft_64_c_transform(afft_64_c_type* state,
  const double* src_re, const double* src_im, double* dst_re, double* dst_im)
{
  fft((Fft<double>*) state, src_re, src_im, dst_re, dst_im);
}

size_t afft_64_ci_memsize(size_t ndim, const size_t* dim)
{
  return Uint(call_selected_impl<MemsizeTag, afft::detail::OpCI, double>(
    afft_auto, Int(ndim), (const Int*) dim));
}

afft_64_ci_type* afft_64_ci_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_64_ci_type*)
    call_selected_impl<CreateTag, afft::detail::OpCI, double>(
      afft_auto, Int(ndim), (const Int*) dim, mem);
}

void afft_64_ci_transform(afft_64_ci_type* state,
  const double* src_re, const double* src_im, double* dst_re, double* dst_im)
{
  ifft((Ifft<double>*) state, src_re, src_im, dst_re, dst_im);
}

size_t afft_64_r_memsize(size_t ndim, const size_t* dim)
{
  return Uint(call_selected_impl<MemsizeTag, afft::detail::OpR, double>(
    afft_auto, Int(ndim), (const Int*) dim));
}

afft_64_r_type* afft_64_r_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_64_r_type*)
    call_selected_impl<CreateTag, afft::detail::OpR, double>(
      afft_auto, Int(ndim), (const Int*) dim, mem);
}

void afft_64_r_transform(afft_64_r_type* state,
  const double* src, double* dst_re, double* dst_im)
{
  rfft((Rfft<double>*) state, src, dst_re, dst_im);
}

size_t afft_64_ri_memsize(size_t ndim, const size_t* dim)
{
  return Uint(call_selected_impl<MemsizeTag, afft::detail::OpRI, double>(
    afft_auto, Int(ndim), (const Int*) dim));
}

afft_64_ri_type* afft_64_ri_create(size_t ndim, const size_t* dim, void* mem)
{
  return (afft_64_ri_type*)
    call_selected_impl<CreateTag, afft::detail::OpRI, double>(
      afft_auto, Int(ndim), (const Int*) dim, mem);
}

void afft_64_ri_transform(afft_64_ri_type* state,
  const double* src_re, const double* src_im, double* dst)
{
  irfft((Irfft<double>*) state, src_re, src_im, dst);
}


size_t afft_64_align_size(size_t sz) { return align_size<double>(sz); }

const size_t afft_alignment = 64;

}
