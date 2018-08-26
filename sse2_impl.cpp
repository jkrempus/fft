#include "sse.hpp"
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
      __builtin_cpu_init();
      return __builtin_cpu_supports("sse");
    }
  }
}
