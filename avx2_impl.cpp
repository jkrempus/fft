#include "avx.hpp"
#include "x86_features.hpp"
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
      return x86_features::supports_avx2();
    }
  }
}
