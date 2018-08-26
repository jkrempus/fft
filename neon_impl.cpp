#include "neon.hpp"
using FloatVec = Neon;
using DoubleVec = Scalar<double>;
static constexpr int impl_idx = 4;

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
