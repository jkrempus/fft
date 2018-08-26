#include "avx.hpp"
using FloatVec = AvxFloat;
using DoubleVec = AvxDouble;
static constexpr int impl_idx = 3;

#include "impl.hpp"

namespace afft
{
  namespace detail
  {
    template<> bool impl_supported<impl_idx>()
    {
      __builtin_cpu_init();
      return __builtin_cpu_supports("avx2");
    }
  }
}
