#include "fft_internal.hpp"
#include <cstdio>

using Cf = complex_format::Split;

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

    template<>
    Int memsize_impl<impl_idx, OpC, float>(Int ndim, const Int* dim)
    {
      return fft_memsize<FloatVec, Cf, Cf>(ndim, dim);
    }

    template<>
    void* create_impl<impl_idx, OpC, float>(
      Int ndim, const Int* dim, void* mem)
    {
      return fft_create<FloatVec, Cf, Cf>(ndim, dim, mem);
    }
    
    template<>
    Int memsize_impl<impl_idx, OpCI, float>(Int ndim, const Int* dim)
    {
      return ifft_memsize<FloatVec, Cf, Cf>(ndim, dim);
    }

    template<>
    void* create_impl<impl_idx, OpCI, float>(
      Int ndim, const Int* dim, void* mem)
    {
      return ifft_create<FloatVec, Cf, Cf>(ndim, dim, mem);
    }
    
    template<>
    Int memsize_impl<impl_idx, OpR, float>(Int ndim, const Int* dim)
    {
      return rfft_memsize<FloatVec, Cf>(ndim, dim);
    }

    template<>
    void* create_impl<impl_idx, OpR, float>(
      Int ndim, const Int* dim, void* mem)
    {
      return rfft_create<FloatVec, Cf>(ndim, dim, mem);
    }
    
    template<>
    Int memsize_impl<impl_idx, OpRI, float>(Int ndim, const Int* dim)
    {
      return irfft_memsize<FloatVec, Cf>(ndim, dim);
    }

    template<>
    void* create_impl<impl_idx, OpRI, float>(
      Int ndim, const Int* dim, void* mem)
    {
      return irfft_create<FloatVec, Cf>(ndim, dim, mem);
    }

    template<>
    Int memsize_impl<impl_idx, OpC, double>(Int ndim, const Int* dim)
    {
      return fft_memsize<DoubleVec, Cf, Cf>(ndim, dim);
    }

    template<>
    void* create_impl<impl_idx, OpC, double>(
      Int ndim, const Int* dim, void* mem)
    {
      return fft_create<DoubleVec, Cf, Cf>(ndim, dim, mem);
    }
    
    template<>
    Int memsize_impl<impl_idx, OpCI, double>(Int ndim, const Int* dim)
    {
      return ifft_memsize<DoubleVec, Cf, Cf>(ndim, dim);
    }

    template<>
    void* create_impl<impl_idx, OpCI, double>(
      Int ndim, const Int* dim, void* mem)
    {
      return ifft_create<DoubleVec, Cf, Cf>(ndim, dim, mem);
    }
    
    template<>
    Int memsize_impl<impl_idx, OpR, double>(Int ndim, const Int* dim)
    {
      return rfft_memsize<DoubleVec, Cf>(ndim, dim);
    }

    template<>
    void* create_impl<impl_idx, OpR, double>(
      Int ndim, const Int* dim, void* mem)
    {
      return rfft_create<DoubleVec, Cf>(ndim, dim, mem);
    }
    
    template<>
    Int memsize_impl<impl_idx, OpRI, double>(Int ndim, const Int* dim)
    {
      return irfft_memsize<DoubleVec, Cf>(ndim, dim);
    }

    template<>
    void* create_impl<impl_idx, OpRI, double>(
      Int ndim, const Int* dim, void* mem)
    {
      return irfft_create<DoubleVec, Cf>(ndim, dim, mem);
    }
  }
}

