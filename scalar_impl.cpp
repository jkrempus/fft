#include "common.hpp"
using FloatVec = Scalar<float>;
using DoubleVec = Scalar<double>;
static constexpr int impl_idx = 1;

#include "impl.hpp"

namespace afft
{
  namespace detail
  {
    template<> bool impl_supported<impl_idx>() { return true; }
  }
}
