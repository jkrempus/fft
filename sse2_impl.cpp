#include "sse.hpp"
#include "x86_features.hpp"
using FloatVec = SseFloat;
using DoubleVec = SseDouble;
static constexpr int impl_idx = 2;

#include "impl.hpp"

namespace afft
{
  namespace detail
  {
    template<> bool impl_supported<impl_idx>()
    {
      return x86_features::supports_sse2();
    }
  }
}
