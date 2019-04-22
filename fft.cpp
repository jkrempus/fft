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
    Int memsize_impl(Int ndim, const Int* dim, int format);

    template<Int impl, typename Op, typename T>
    void* create_impl(Int ndim, const Int* dim, int format, void* mem);

#define DECLARE_IMPL_FOR_TYPE(IMPL, T) \
    template<> Int memsize_impl<IMPL, OpC, T>(Int, const Int*, int); \
    template<> Int memsize_impl<IMPL, OpCI, T>(Int, const Int*, int); \
    template<> Int memsize_impl<IMPL, OpR, T>(Int, const Int*, int); \
    template<> Int memsize_impl<IMPL, OpRI, T>(Int, const Int*, int); \
    template<> void* create_impl<IMPL, OpC, T>(Int, const Int*, int, void*); \
    template<> void* create_impl<IMPL, OpCI, T>(Int, const Int*, int, void*); \
    template<> void* create_impl<IMPL, OpR, T>(Int, const Int*, int, void*); \
    template<> void* create_impl<IMPL, OpRI, T>(Int, const Int*, int, void*);

#define DECLARE_IMPL(IMPL) \
    template<> bool impl_supported<IMPL>(); \
    DECLARE_IMPL_FOR_TYPE(IMPL, float) \
    DECLARE_IMPL_FOR_TYPE(IMPL, double)

    DECLARE_IMPL(afft_scalar)
    DECLARE_IMPL(afft_sse2)
    DECLARE_IMPL(afft_avx2)
    DECLARE_IMPL(afft_avx512f)
    DECLARE_IMPL(afft_neon)
  }
}

namespace
{
  struct MemsizeTag{};
  struct CreateTag{};
  struct SupportedTag{};

  template<Int impl, typename Op, typename T>
  Int call_impl(const MemsizeTag&, Int ndim, const Int* dim, int format)
  {
    return afft::detail::memsize_impl<impl, Op, T>(ndim, dim, format);
  }

  template<Int impl, typename Op, typename T>
  void* call_impl(
    const CreateTag&, Int ndim, const Int* dim, int format, void* ptr)
  {
    return afft::detail::create_impl<impl, Op, T>(ndim, dim, format, ptr);
  }

  template<Int impl, typename Op, typename T>
  bool call_impl(const SupportedTag&)
  {
    return afft::detail::impl_supported<impl>();
  }

  Int get_null_value(const MemsizeTag&){ return 0; }
  void* get_null_value(const CreateTag&){ return nullptr; }
  bool get_null_value(const SupportedTag&){ return false; }

  template<typename FnTag, typename Op, typename T, typename... Args>
  auto call_selected_impl(Int impl, Args... args)
  {
    if(impl == afft_auto)
    {
#ifdef AFFT_NEON_ENABLED
      if(afft::detail::impl_supported<afft_neon>())
        return call_impl<afft_neon, Op, T>(FnTag(), args...);
#endif

#ifdef AFFT_AVX512F_ENABLED
      if(afft::detail::impl_supported<afft_avx512f>())
        return call_impl<afft_avx512f, Op, T>(FnTag(), args...);
#endif

#ifdef AFFT_AVX2_ENABLED
      if(afft::detail::impl_supported<afft_avx2>())
        return call_impl<afft_avx2, Op, T>(FnTag(), args...);
#endif

#ifdef AFFT_SSE2_ENABLED
      if(afft::detail::impl_supported<afft_sse2>())
        return call_impl<afft_sse2, Op, T>(FnTag(), args...);
#endif

#ifdef AFFT_SCALAR_ENABLED
      if(afft::detail::impl_supported<afft_scalar>())
        return call_impl<afft_scalar, Op, T>(FnTag(), args...);
#endif

      return get_null_value(FnTag());
    }
    else
    {
      switch(impl)
      {
#ifdef AFFT_NEON_ENABLED
        case afft_neon: return call_impl<afft_neon, Op, T>(FnTag(), args...);
#endif

#ifdef AFFT_AVX512F_ENABLED
        case afft_avx512f: return call_impl<afft_avx512f, Op, T>(FnTag(), args...);
#endif

#ifdef AFFT_AVX2_ENABLED
        case afft_avx2: return call_impl<afft_avx2, Op, T>(FnTag(), args...);
#endif

#ifdef AFFT_SSE2_ENABLED
        case afft_sse2: return call_impl<afft_sse2, Op, T>(FnTag(), args...);
#endif

#ifdef AFFT_SCALAR_ENABLED
        case afft_scalar: return call_impl<afft_scalar, Op, T>(FnTag(), args...);
#endif
      }
    }

    return get_null_value(FnTag());
  }
}

extern "C"
{

size_t afft_32_c_memsize(size_t ndim, const size_t* dim, int format, int impl)
{
  return Uint(call_selected_impl<MemsizeTag, afft::detail::OpC, float>(
    impl, Int(ndim), (const Int*) dim, format));
}

afft_32_c_type* afft_32_c_create(
  size_t ndim, const size_t* dim, int format, void* mem, int impl)
{
  return (afft_32_c_type*)
    call_selected_impl<CreateTag, afft::detail::OpC, float>(
      impl, Int(ndim), (const Int*) dim, format, mem);
}

void afft_32_c_transform(afft_32_c_type* state,
  const float* src_re, const float* src_im, float* dst_re, float* dst_im)
{
  fft((Fft<float>*) state, src_re, src_im, dst_re, dst_im);
}

size_t afft_32_ci_memsize(size_t ndim, const size_t* dim, int format, int impl)
{
  return Uint(call_selected_impl<MemsizeTag, afft::detail::OpCI, float>(
    impl, Int(ndim), (const Int*) dim, format));
}

afft_32_ci_type* afft_32_ci_create(
  size_t ndim, const size_t* dim, int format, void* mem, int impl)
{
  return (afft_32_ci_type*)
    call_selected_impl<CreateTag, afft::detail::OpCI, float>(
      impl, Int(ndim), (const Int*) dim, format, mem);
}

void afft_32_ci_transform(afft_32_ci_type* state,
  const float* src_re, const float* src_im, float* dst_re, float* dst_im)
{
  ifft((Ifft<float>*) state, src_re, src_im, dst_re, dst_im);
}

size_t afft_32_r_memsize(size_t ndim, const size_t* dim, int format, int impl)
{
  return Uint(call_selected_impl<MemsizeTag, afft::detail::OpR, float>(
    impl, Int(ndim), (const Int*) dim, format));
}

afft_32_r_type* afft_32_r_create(
  size_t ndim, const size_t* dim, int format, void* mem, int impl)
{
  return (afft_32_r_type*)
    call_selected_impl<CreateTag, afft::detail::OpR, float>(
      impl, Int(ndim), (const Int*) dim, format, mem);
}

void afft_32_r_transform(afft_32_r_type* state,
  const float* src, float* dst_re, float* dst_im)
{
  rfft((Rfft<float>*) state, src, dst_re, dst_im);
}

size_t afft_32_ri_memsize(size_t ndim, const size_t* dim, int format, int impl)
{
  return Uint(call_selected_impl<MemsizeTag, afft::detail::OpRI, float>(
    impl, Int(ndim), (const Int*) dim, format));
}

afft_32_ri_type* afft_32_ri_create(
  size_t ndim, const size_t* dim, int format, void* mem, int impl)
{
  return (afft_32_ri_type*)
    call_selected_impl<CreateTag, afft::detail::OpRI, float>(
      impl, Int(ndim), (const Int*) dim, format, mem);
}

void afft_32_ri_transform(afft_32_ri_type* state,
  const float* src_re, const float* src_im, float* dst)
{
  irfft((Irfft<float>*) state, src_re, src_im, dst);
}

size_t afft_32_align_size(size_t sz) { return align_size<float>(sz); }

size_t afft_64_c_memsize(size_t ndim, const size_t* dim, int format, int impl)
{
  return Uint(call_selected_impl<MemsizeTag, afft::detail::OpC, double>(
    impl, Int(ndim), (const Int*) dim, format));
}

afft_64_c_type* afft_64_c_create(
  size_t ndim, const size_t* dim, int format, void* mem, int impl)
{
  return (afft_64_c_type*)
    call_selected_impl<CreateTag, afft::detail::OpC, double>(
      impl, Int(ndim), (const Int*) dim, format, mem);
}

void afft_64_c_transform(afft_64_c_type* state,
  const double* src_re, const double* src_im, double* dst_re, double* dst_im)
{
  fft((Fft<double>*) state, src_re, src_im, dst_re, dst_im);
}

size_t afft_64_ci_memsize(size_t ndim, const size_t* dim, int format, int impl)
{
  return Uint(call_selected_impl<MemsizeTag, afft::detail::OpCI, double>(
    impl, Int(ndim), (const Int*) dim, format));
}

afft_64_ci_type* afft_64_ci_create(
  size_t ndim, const size_t* dim, int format, void* mem, int impl)
{
  return (afft_64_ci_type*)
    call_selected_impl<CreateTag, afft::detail::OpCI, double>(
      impl, Int(ndim), (const Int*) dim, format, mem);
}

void afft_64_ci_transform(afft_64_ci_type* state,
  const double* src_re, const double* src_im, double* dst_re, double* dst_im)
{
  ifft((Ifft<double>*) state, src_re, src_im, dst_re, dst_im);
}

size_t afft_64_r_memsize(size_t ndim, const size_t* dim, int format, int impl)
{
  return Uint(call_selected_impl<MemsizeTag, afft::detail::OpR, double>(
    impl, Int(ndim), (const Int*) dim, format));
}

afft_64_r_type* afft_64_r_create(
  size_t ndim, const size_t* dim, int format, void* mem, int impl)
{
  return (afft_64_r_type*)
    call_selected_impl<CreateTag, afft::detail::OpR, double>(
      impl, Int(ndim), (const Int*) dim, format, mem);
}

void afft_64_r_transform(afft_64_r_type* state,
  const double* src, double* dst_re, double* dst_im)
{
  rfft((Rfft<double>*) state, src, dst_re, dst_im);
}

size_t afft_64_ri_memsize(size_t ndim, const size_t* dim, int format, int impl)
{
  return Uint(call_selected_impl<MemsizeTag, afft::detail::OpRI, double>(
    impl, Int(ndim), (const Int*) dim, format));
}

afft_64_ri_type* afft_64_ri_create(
  size_t ndim, const size_t* dim, int format, void* mem, int impl)
{
  return (afft_64_ri_type*)
    call_selected_impl<CreateTag, afft::detail::OpRI, double>(
      impl, Int(ndim), (const Int*) dim, format, mem);
}

void afft_64_ri_transform(afft_64_ri_type* state,
  const double* src_re, const double* src_im, double* dst)
{
  irfft((Irfft<double>*) state, src_re, src_im, dst);
}

size_t afft_64_align_size(size_t sz) { return align_size<double>(sz); }

int afft_supported(int impl)
{
  return call_selected_impl<SupportedTag, afft::detail::OpC, float>(impl);
}

const size_t afft_alignment = 64;

}
