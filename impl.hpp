#include "fft_internal.hpp"
#include <cstdio>

namespace cf = complex_format;

namespace afft
{
  enum {split_format = 0, interleaved_format = 1};

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

    template<>
    Int memsize_impl<impl_idx, OpC, float>(Int ndim, const Int* dim, int format)
    {
#ifdef AFFT_SPLIT_ENABLED
      if(format == split_format)
        return fft_memsize<FloatVec, cf::Split, cf::Split>(ndim, dim);
#endif
#ifdef AFFT_INTERLEAVED_ENABLED
      if(format == interleaved_format)
        return fft_memsize<FloatVec, cf::Scal, cf::Scal>(ndim, dim);
#endif

      ASSERT(0);
      return -1;
    }

    template<>
    void* create_impl<impl_idx, OpC, float>(
      Int ndim, const Int* dim, int format, void* mem)
    {
#ifdef AFFT_SPLIT_ENABLED
      if(format == split_format)
      {
        return fft_create<FloatVec, cf::Split, cf::Split>(ndim, dim, mem);
      }
#endif
#ifdef AFFT_INTERLEAVED_ENABLED
      if(format == interleaved_format)
      {
        return fft_create<FloatVec, cf::Scal, cf::Scal>(ndim, dim, mem);
      }
#endif

      ASSERT(0);
      return NULL;
    }

    template<>
    Int memsize_impl<impl_idx, OpCI, float>(Int ndim, const Int* dim, int format)
    {
#ifdef AFFT_SPLIT_ENABLED
      if(format == split_format)
        return ifft_memsize<FloatVec, cf::Split, cf::Split>(ndim, dim);
#endif
#ifdef AFFT_INTERLEAVED_ENABLED
      if(format == interleaved_format)
        return ifft_memsize<FloatVec, cf::Scal, cf::Scal>(ndim, dim);
#endif

      ASSERT(0);
      return -1;
    }

    template<>
    void* create_impl<impl_idx, OpCI, float>(
      Int ndim, const Int* dim, int format, void* mem)
    {
#ifdef AFFT_SPLIT_ENABLED
      if(format == split_format)
      {
        return ifft_create<FloatVec, cf::Split, cf::Split>(ndim, dim, mem);
      }
#endif
#ifdef AFFT_INTERLEAVED_ENABLED
      if(format == interleaved_format)
      {
        return ifft_create<FloatVec, cf::Scal, cf::Scal>(ndim, dim, mem);
      }
#endif

      ASSERT(0);
      return NULL;
    }

    template<>
    Int memsize_impl<impl_idx, OpR, float>(Int ndim, const Int* dim, int format)
    {
#ifdef AFFT_SPLIT_ENABLED
      if(format == split_format)
        return rfft_memsize<FloatVec, cf::Split>(ndim, dim);
#endif
#ifdef AFFT_INTERLEAVED_ENABLED
      if(format == interleaved_format)
        return rfft_memsize<FloatVec, cf::Scal>(ndim, dim);
#endif

      ASSERT(0);
      return -1;
    }

    template<>
    void* create_impl<impl_idx, OpR, float>(
      Int ndim, const Int* dim, int format, void* mem)
    {
#ifdef AFFT_SPLIT_ENABLED
      if(format == split_format)
      {
        return rfft_create<FloatVec, cf::Split>(ndim, dim, mem);
      }
#endif
#ifdef AFFT_INTERLEAVED_ENABLED
      if(format == interleaved_format)
      {
        return rfft_create<FloatVec, cf::Scal>(ndim, dim, mem);
      }
#endif

      ASSERT(0);
      return NULL;
    }

    template<>
    Int memsize_impl<impl_idx, OpRI, float>(Int ndim, const Int* dim, int format)
    {
#ifdef AFFT_SPLIT_ENABLED
      if(format == split_format)
        return irfft_memsize<FloatVec, cf::Split>(ndim, dim);
#endif
#ifdef AFFT_INTERLEAVED_ENABLED
      if(format == interleaved_format)
        return irfft_memsize<FloatVec, cf::Scal>(ndim, dim);
#endif

      ASSERT(0);
      return -1;
    }

    template<>
    void* create_impl<impl_idx, OpRI, float>(
      Int ndim, const Int* dim, int format, void* mem)
    {
#ifdef AFFT_SPLIT_ENABLED
      if(format == split_format)
      {
        return irfft_create<FloatVec, cf::Split>(ndim, dim, mem);
      }
#endif
#ifdef AFFT_INTERLEAVED_ENABLED
      if(format == interleaved_format)
      {
        return irfft_create<FloatVec, cf::Scal>(ndim, dim, mem);
      }
#endif

      ASSERT(0);
      return NULL;
    }

    template<>
    Int memsize_impl<impl_idx, OpC, double>(Int ndim, const Int* dim, int format)
    {
#ifdef AFFT_SPLIT_ENABLED
      if(format == split_format)
        return fft_memsize<DoubleVec, cf::Split, cf::Split>(ndim, dim);
#endif
#ifdef AFFT_INTERLEAVED_ENABLED
      if(format == interleaved_format)
        return fft_memsize<DoubleVec, cf::Scal, cf::Scal>(ndim, dim);
#endif

      ASSERT(0);
      return -1;
    }

    template<>
    void* create_impl<impl_idx, OpC, double>(
      Int ndim, const Int* dim, int format, void* mem)
    {
#ifdef AFFT_SPLIT_ENABLED
      if(format == split_format)
      {
        return fft_create<DoubleVec, cf::Split, cf::Split>(ndim, dim, mem);
      }
#endif
#ifdef AFFT_INTERLEAVED_ENABLED
      if(format == interleaved_format)
      {
        return fft_create<DoubleVec, cf::Scal, cf::Scal>(ndim, dim, mem);
      }
#endif

      ASSERT(0);
      return NULL;
    }

    template<>
    Int memsize_impl<impl_idx, OpCI, double>(Int ndim, const Int* dim, int format)
    {
#ifdef AFFT_SPLIT_ENABLED
      if(format == split_format)
        return ifft_memsize<DoubleVec, cf::Split, cf::Split>(ndim, dim);
#endif
#ifdef AFFT_INTERLEAVED_ENABLED
      if(format == interleaved_format)
        return ifft_memsize<DoubleVec, cf::Scal, cf::Scal>(ndim, dim);
#endif

      ASSERT(0);
      return -1;
    }

    template<>
    void* create_impl<impl_idx, OpCI, double>(
      Int ndim, const Int* dim, int format, void* mem)
    {
#ifdef AFFT_SPLIT_ENABLED
      if(format == split_format)
      {
        return ifft_create<DoubleVec, cf::Split, cf::Split>(ndim, dim, mem);
      }
#endif
#ifdef AFFT_INTERLEAVED_ENABLED
      if(format == interleaved_format)
      {
        return ifft_create<DoubleVec, cf::Scal, cf::Scal>(ndim, dim, mem);
      }
#endif

      ASSERT(0);
      return NULL;
    }

    template<>
    Int memsize_impl<impl_idx, OpR, double>(Int ndim, const Int* dim, int format)
    {
#ifdef AFFT_SPLIT_ENABLED
      if(format == split_format)
        return rfft_memsize<DoubleVec, cf::Split>(ndim, dim);
#endif
#ifdef AFFT_INTERLEAVED_ENABLED
      if(format == interleaved_format)
        return rfft_memsize<DoubleVec, cf::Scal>(ndim, dim);
#endif

      ASSERT(0);
      return -1;
    }

    template<>
    void* create_impl<impl_idx, OpR, double>(
      Int ndim, const Int* dim, int format, void* mem)
    {
#ifdef AFFT_SPLIT_ENABLED
      if(format == split_format)
      {
        return rfft_create<DoubleVec, cf::Split>(ndim, dim, mem);
      }
#endif
#ifdef AFFT_INTERLEAVED_ENABLED
      if(format == interleaved_format)
      {
        return rfft_create<DoubleVec, cf::Scal>(ndim, dim, mem);
      }
#endif

      ASSERT(0);
      return NULL;
    }

    template<>
    Int memsize_impl<impl_idx, OpRI, double>(Int ndim, const Int* dim, int format)
    {
#ifdef AFFT_SPLIT_ENABLED
      if(format == split_format)
        return irfft_memsize<DoubleVec, cf::Split>(ndim, dim);
#endif
#ifdef AFFT_INTERLEAVED_ENABLED
      if(format == interleaved_format)
        return irfft_memsize<DoubleVec, cf::Scal>(ndim, dim);
#endif

      ASSERT(0);
      return -1;
    }

    template<>
    void* create_impl<impl_idx, OpRI, double>(
      Int ndim, const Int* dim, int format, void* mem)
    {
#ifdef AFFT_SPLIT_ENABLED
      if(format == split_format)
      {
        return irfft_create<DoubleVec, cf::Split>(ndim, dim, mem);
      }
#endif
#ifdef AFFT_INTERLEAVED_ENABLED
      if(format == interleaved_format)
      {
        return irfft_create<DoubleVec, cf::Scal>(ndim, dim, mem);
      }
#endif

      ASSERT(0);
      return NULL;
    }
  }
}

