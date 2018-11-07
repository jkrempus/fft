//#define ARRAY_IPC_ENABLED
#include "misc/array_ipc.h"

#include "fft.hpp"
#include "testing.hpp"

#include <cassert>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <random>
#include <cstdint>
#include <unordered_set>
#include <chrono>
#include <type_traits>

#ifdef HAVE_FFTW
#include "fftw3.h"
#endif

//#define INTERLEAVED 1

static constexpr Int maxdim = 64;
static constexpr Int chunk_size = 0;

using std::chrono::high_resolution_clock;

template<typename Rep, typename Period>
double to_seconds(const std::chrono::duration<Rep, Period>& d)
{
  return std::chrono::duration<double>(d).count();
}

template<typename T> T sq(T a){ return a * a; }

struct IterateMultidim
{
  Int ndim;
  const Int* size;  
  Int idx[sizeof(Int) * 8];

  IterateMultidim(Int ndim, const Int* size) : ndim(ndim), size(size)
  {
    assert(ndim < sizeof(idx) / sizeof(idx[0]));
    for(Int i = 0; i < ndim; i++) idx[i] = 0;
  }

  bool empty() { return idx[0] == size[0]; }

  void advance()
  {
    for(Int i = ndim - 1; i >= 0; i--)
    {
      idx[i]++;
      if(idx[i] < size[i]) return;
      idx[i] = 0;
    }

    idx[0] = size[0];
  }
};

extern "C" void* valloc(size_t);

void* alloc(Int n)
{
#ifdef _WIN32
  return _aligned_malloc(n, 4096);
#else
  return valloc(n);
#endif
}

void dealloc(void* p)
{
#ifdef _WIN32
  _aligned_free(p);
#else
  free(p);
#endif
}

template<typename T>
T* alloc_array(Int n)
{
  auto r = (T*) alloc(n * sizeof(T));
  //printf("allocated range %p %p\n", r, r + 2 * n * sizeof(T));
  return r;
}

Int chunked_index(Int i, Int chunk_size)
{
  if(chunk_size == 0)
    return i;
  else
    return 2 * i - (i & (chunk_size - 1));
}

template<typename T> struct View
{
  Int ndim = 0; 
  Int size[maxdim];
  Int stride[maxdim];
  T* data = nullptr;
  Int chunk_size;

  View() = default;

  View(const View& other)
    : ndim(other.ndim), data(other.data), chunk_size(other.chunk_size)
  {
    std::copy_n(other.size, other.ndim, size);
    std::copy_n(other.stride, other.ndim - 1, stride);
  }

  View& operator=(const View& other)
  {
    ndim = other.ndim;
    data = other.data;
    chunk_size = other.chunk_size;
    std::copy_n(other.size, other.ndim, size);
    std::copy_n(other.stride, other.ndim - 1, stride);
    return *this;
  }

  View plane(Int i) const
  {
    View<T> r;
    r.ndim = ndim - 1;
    r.chunk_size = chunk_size;
    std::copy_n(size + 1, r.ndim, r.size);
    std::copy_n(stride + 1, r.ndim - 1, r.stride);
    r.data = data + stride[0] * i;
    return r;
  }

  View sub(Int dim, Int start, Int end) const
  {
    View<T> r;
    r.ndim = ndim;
    r.chunk_size = chunk_size;
    std::copy_n(size, ndim, r.size);
    std::copy_n(stride, ndim - 1, r.stride);
    r.size[dim] = end - start;
    r.data = data + (dim < ndim - 1 ? stride[dim] : 1) * start;
    return r;
  }

  T* ptr(Int* idx) const
  {
    auto p = data + chunked_index(idx[ndim - 1], chunk_size);
    for(Int i = 0; i < ndim - 1; i++) p += idx[i] * stride[i];
    return p;
  }
};

template<typename T, typename U>
void copy_view(const View<T>& src, const View<U>& dst)
{
  if(src.ndim == 1)
    for(Int i = 0; i < std::min(src.size[0], dst.size[0]); i++)
      dst.data[chunked_index(i, dst.chunk_size)] =
        src.data[chunked_index(i, src.chunk_size)];
  else
    for(Int i = 0; i < std::min(src.size[0], dst.size[0]); i++)
      copy_view(src.plane(i), dst.plane(i));
}

template<typename T, typename U>
void fill_view(const T& value, const View<U>& dst)
{
  if(dst.ndim == 1)
    for(Int i = 0; i < dst.size[0]; i++)
      dst.data[chunked_index(i, dst.chunk_size)] = value;
  else
    for(Int i = 0; i < dst.size[0]; i++)
      fill_view(value, dst.plane(i));
}

Int mirror_idx(Int size, Int idx) { return (size - 1) & (size - idx); }

template<bool is_antisym, typename T, typename U>
void copy_symmetric_view(const View<T>& src, const View<U>& dst)
{
  for(IterateMultidim it(dst.ndim, dst.size); !it.empty(); it.advance())
  {
    bool mirror = false;
    for(Int j = 0; j < src.ndim; j++) mirror = mirror || it.idx[j] >= src.size[j];

    Int s = 0;
    Int s_mirrored = 0;
    Int d = 0;
    Int midx[maxdim];
    for(Int j = 0; j < src.ndim; j++) midx[j] = mirror_idx(dst.size[j], it.idx[j]);

    if(is_antisym)
    {
      bool is_equal = true;
      for(Int j = 0; j < src.ndim; j++) is_equal = is_equal && it.idx[j] == midx[j];

      if(is_equal)
        *dst.ptr(it.idx) = 0;
      else
        *dst.ptr(it.idx) = mirror ? -*src.ptr(midx) : *src.ptr(it.idx);
    }
    else
      *dst.ptr(it.idx) = mirror ? *src.ptr(midx) : *src.ptr(it.idx);
  }
}

static constexpr Int get_im_offset(Int split_im_offset)
{
  return chunk_size ? chunk_size : split_im_offset;
}

template<typename T>
View<T> create_view(T* ptr, const std::vector<Int>& size, Int chunk_size)
{
  View<T> r;
  r.ndim = size.size();
  std::copy_n(&size[0], r.ndim, r.size); 
  Int s = chunk_size ? 2 : 1;
  for(Int i = r.ndim - 1; i > 0; i--)
  {
    s *= r.size[i];
    r.stride[i - 1] = s;
  }

  r.data = ptr;
  r.chunk_size = chunk_size;
  return r;
}

template<typename T>
T product(const T* begin, const T* end)
{
  T r(1);
  for(const T* p = begin; p != end; p++) r *= *p;
  return r;
}

template<typename T>
T product(const std::vector<T>& v)
{
  T r(1);
  for(auto& e : v) r *= e;
  return r;
}

template<
  typename T, typename ImOff, typename HalvedDim,
  bool is_real, bool is_inverse>
struct SplitWrapperBase { };

template<typename T, typename ImOff, typename HalvedDim, bool is_inverse_>
struct SplitWrapperBase<T, ImOff, HalvedDim, false, is_inverse_>
{
  std::vector<Int> size;
  T* src;
  T* dst;
  
  SplitWrapperBase(const std::vector<Int> size) :
    size(size),
    src((T*) alloc(2 * sizeof(T) * product(size))),
    dst((T*) alloc(2 * sizeof(T) * product(size))) { }

  ~SplitWrapperBase()
  {
    dealloc(src);
    dealloc(dst);
  }

  template<typename U>
  void set_input(U* p)
  {
    Int n = product(size);
    copy_view(create_view(p, size, 0), create_view(src, size, 0));
    copy_view(create_view(p + n, size, 0), create_view(src + n, size, 0));
  }

  template<typename U>
  void get_output(U* p)
  {
    Int n = product(size);
    copy_view(create_view(dst, size, 0), create_view(p, size, 0));
    copy_view(create_view(dst + n, size, 0), create_view(p + n, size, 0));
  }
};

template<typename T, typename ImOff, typename HalvedDim>
struct SplitWrapperBase<T, ImOff, HalvedDim, true, false>
{
  Int im_off;
  std::vector<Int> size;
  T* src;
  T* dst;
  
  SplitWrapperBase(const std::vector<Int>& size) :
    im_off(ImOff::template get<T>(size)),
    size(size),
    src(alloc_array<T>(product(size))),
    dst(alloc_array<T>(2 * im_off)) { }

  ~SplitWrapperBase()
  {
    dealloc(src);
    dealloc(dst);
  }

  template<typename U>
  void set_input(U* p)
  {
    copy_view(create_view(p, size, 0), create_view(src, size, 0));
  }

  template<typename U>
  void get_output(U* p)
  {
    Int n = product(size);

    Int hd = HalvedDim::get(size);
    auto symmetric_size = size;
    symmetric_size[hd] = size[hd] / 2 + 1;

    copy_symmetric_view<false>(
      create_view(dst, symmetric_size, 0),
      create_view(p, size, 0));
  
    copy_symmetric_view<true>(
      create_view(dst + im_off, symmetric_size, 0),
      create_view(p + n, size, 0));
  }
};

template<typename T, typename ImOff, typename HalvedDim>
struct SplitWrapperBase<T, ImOff, HalvedDim, true, true>
{
  Int im_off;
  std::vector<Int> size;
  std::vector<Int> symmetric_size;
  T* src;
  T* dst;

  SplitWrapperBase(const std::vector<Int>& size) :
    im_off(ImOff::template get<T>(size)),
    size(size),
    dst(alloc_array<T>(product(size))),
    src(alloc_array<T>(2 * im_off))
  {
    Int hd = HalvedDim::get(size);
    symmetric_size = size;
    symmetric_size[hd] = size[hd] / 2 + 1;
  }

  ~SplitWrapperBase()
  {
    dealloc(src);
    dealloc(dst);
  }

  template<typename U>
  void set_input(U* p)
  {
    copy_view(
      create_view(p, size, 0),
      create_view(src, symmetric_size, 0));

    copy_view(
      create_view(p + product(size), size, 0),
      create_view(src + im_off, symmetric_size, 0));
  }

  template<typename U>
  void get_output(U* p)
  {
    copy_view(
      create_view(dst, size, 0),
      create_view(p, size, 0));

    fill_view(U(0), create_view(p + product(size), size, 0));
  }
};

template<typename T, typename HalvedDim, bool is_real, bool is_inverse>
struct InterleavedWrapperBase { };

template<typename T, typename HalvedDim, bool is_inverse_>
struct InterleavedWrapperBase<T, HalvedDim, false, is_inverse_>
{
  std::vector<Int> size;
  T* src;
  T* dst;

  InterleavedWrapperBase(const std::vector<Int>& size) :
    size(size),
    src((T*) alloc(2 * sizeof(T) * product(size))),
    dst((T*) alloc(2 * sizeof(T) * product(size))) { }

  ~InterleavedWrapperBase()
  {
    dealloc(src);
    dealloc(dst);
  };

  template<typename U>
  void set_input(U* p)
  {
    Int n = product(size);
    copy_view(create_view(p, size, 0), create_view(src, size, 1));
    copy_view(create_view(p + n, size, 0), create_view(src + 1, size, 1));
  }

  template<typename U>
  void get_output(U* p)
  {
    Int n = product(size);
    copy_view(create_view(dst, size, 1), create_view(p, size, 0));
    copy_view(create_view(dst + 1, size, 1), create_view(p + n, size, 0));
  }
};

template<typename T, typename HalvedDim>
struct InterleavedWrapperBase<T, HalvedDim, true, false>
{
  std::vector<Int> size;
  std::vector<Int> symmetric_size;
  T* src;
  T* dst;

  InterleavedWrapperBase(const std::vector<Int>& size) : size(size)
  {
    Int hd = HalvedDim::get(size);
    symmetric_size = size;
    symmetric_size[hd] = size[hd] / 2 + 1;
    dst = (T*) alloc(2 * sizeof(T) * product(symmetric_size));
    src = (T*) alloc(sizeof(T) * product(size));
  }

  ~InterleavedWrapperBase()
  {
    dealloc(src);
    dealloc(dst);
  };

  template<typename U>
  void set_input(U* p)
  {
    copy_view(create_view(p, size, 0), create_view(src, size, 0));
  }

  template<typename U>
  void get_output(U* p)
  {
    Int n = product(size);

    copy_symmetric_view<false>(
      create_view(dst, symmetric_size, 1),
      create_view(p, size, 0));

    copy_symmetric_view<true>(
      create_view(dst + 1, symmetric_size, 1),
      create_view(p + n, size, 0));
  }
};

template<typename T, typename HalvedDim>
struct InterleavedWrapperBase<T, HalvedDim, true, true>
{
  std::vector<Int> size;
  std::vector<Int> symmetric_size;
  T* src;
  T* dst;
  
  InterleavedWrapperBase(const std::vector<Int>& size) : size(size)
  {
    Int hd = HalvedDim::get(size);
    symmetric_size = size;
    symmetric_size[hd] = size[hd] / 2 + 1;

    Int src_n = product(symmetric_size);
    src = (T*) alloc(2 * sizeof(T) * src_n);
    dst = (T*) alloc(sizeof(T) * product(size));
  }

  ~InterleavedWrapperBase()
  {
    dealloc(src);
    dealloc(dst);
  };

  template<typename U>
  void set_input(U* p)
  {
    Int n = product(size);
    copy_view(create_view(p, size, 0), create_view(src, symmetric_size, 1));
    copy_view(create_view(p + n, size, 0), create_view(src + 1, symmetric_size, 1));
  }

  template<typename U>
  void get_output(U* p)
  {
    Int n = product(size);
    copy_view(create_view(dst, size, 0), create_view(p, size, 0));
    fill_view(U(0), create_view(p + n, size, 0));
  }
};

struct HalvedDimFirst
{
  static Int get(const std::vector<Int>& size)
  {
    auto it = std::find_if(size.begin(), size.end(),
      [](Int e){ return e > 1; });

    return it - size.begin();
  }
};

struct HalvedDimLast
{
  static Int get(const std::vector<Int>& size)
  {
    return size.size() - 1;
  }
};

struct AlignedImOff
{
  template<typename T>
  static Int get(const std::vector<Int>& size)
  {
    auto it = std::find_if(size.begin(), size.end(),
      [](Int e){ return e > 1; });

    Int inner = product(&it[1], &size[0] + size.size());
    return afft_32_align_size((it[0] / 2 + 1) * inner);
  }
};

#ifdef INTERLEAVED
template<typename T, bool is_real, bool is_inverse>
using Base = InterleavedWrapperBase<T, HalvedDimFirst, is_real, is_inverse>;
#else
template<typename T, bool is_real, bool is_inverse>
using Base = SplitWrapperBase<
  T, AlignedImOff, HalvedDimFirst, is_real, is_inverse>;
#endif

template<typename T, bool is_real, bool is_inverse>
struct TestWrapper { };

template<typename T>
struct TestWrapper<T, false, false>
: public Base<T, false, false>
{
  static const bool is_real = false;
  static const bool is_inverse = false;
  typedef T value_type;
  afft::complex_transform<T> t;
  TestWrapper(const std::vector<Int>& size, Int simd_impl, bool is_bench) :
    Base<T, false, false>(size),
    t(size.size(), (const Uint*) &size[0], nullptr, simd_impl) {}

  void transform()
  {
    Int n = product(this->size);
    t(this->src, this->src + n, this->dst, this->dst + n);
  }
};

template<typename T>
struct TestWrapper<T, false, true>
: public Base<T, false, true>
{
  static const bool is_real = false;
  static const bool is_inverse = true;
  typedef T value_type;
  afft::inverse_complex_transform<T> t;
  TestWrapper(const std::vector<Int>& size, Int simd_impl, bool is_bench) :
    Base<T, false, true>(size),
    t(size.size(), (const Uint*) &size[0], nullptr, simd_impl) {}

  void transform()
  {
    Int n = product(this->size);
    t(this->src, this->src + n, this->dst, this->dst + n);
  }
};

template<typename T>
struct TestWrapper<T, true, false>
: public Base<T, true, false>
{
  static const bool is_real = true;
  static const bool is_inverse = false;
  typedef T value_type;
  afft::real_transform<T> t;

  TestWrapper(const std::vector<Int>& size, Int simd_impl, bool is_bench) :
    Base<T, true, false>(size),
    t(size.size(), (const Uint*) &size[0], nullptr, simd_impl) {}

  void transform()
  {
    t(this->src, this->dst, this->dst + AlignedImOff::get<T>(this->size));
  }
};

template<typename T>
struct TestWrapper<T, true, true>
: public Base<T, true, true>
{
  static const bool is_real = true;
  static const bool is_inverse = true;
  typedef T value_type;
  afft::inverse_real_transform<T> t;

  TestWrapper(const std::vector<Int>& size, Int simd_impl, bool is_bench) :
    Base<T, true, true>(size),
    t(size.size(), (const Uint*) &size[0], nullptr, simd_impl) {}

  void transform()
  {
    t(this->src, this->src + AlignedImOff::get<T>(this->size), this->dst);
  }
};

#ifdef HAVE_FFTW

struct FftwSplitImOff
{
  template<typename T>
  static Int get(const std::vector<Int>& size)
  {
    Int outer = product(&size[0], &size[0] + size.size() - 1);
    return (size.back() / 2 + 1) * outer;
  }
};

template<typename T, bool is_real_, bool is_inverse_>
struct FftwTestWrapper :
  public InterleavedWrapperBase<T, HalvedDimLast, is_real_, is_inverse_>
{
  using Base = InterleavedWrapperBase<T, HalvedDimLast, is_real_, is_inverse_>;

  static const bool is_real = is_real_;
  static const bool is_inverse = is_inverse_;
  typedef float value_type;
  std::conditional_t<std::is_same_v<T, float>, fftwf_plan, fftw_plan> plan;

  unsigned fftw_flags;

  FftwTestWrapper(const std::vector<Int>& size, Int simd_impl, bool is_bench)
  : Base(size), fftw_flags(is_bench ? FFTW_PATIENT : FFTW_ESTIMATE)
  {
    int idx[maxdim];
    std::copy_n(&size[0], size.size(), idx);
    auto src = this->src;
    auto dst = this->dst;

    if constexpr(is_real_)
    {
      if constexpr(is_inverse_)
      {
        if constexpr(std::is_same_v<T, float>)
          plan = fftwf_plan_dft_c2r(
            size.size(), idx, (fftwf_complex*) src, dst, fftw_flags);
        else
          plan = fftw_plan_dft_c2r(
            size.size(), idx, (fftw_complex*) src, dst, fftw_flags);
      }
      else
      {
        if constexpr(std::is_same_v<T, float>)
          plan = fftwf_plan_dft_r2c(
            size.size(), idx, src, (fftwf_complex*) dst, fftw_flags);
        else
          plan = fftw_plan_dft_r2c(
            size.size(), idx, src, (fftw_complex*) dst, fftw_flags);
      }
    }
    else
    {
      if constexpr(is_inverse_)
      {
        if constexpr(std::is_same_v<T, float>)
          plan = fftwf_plan_dft(
            size.size(), idx, 
            (fftwf_complex*) src, (fftwf_complex*) dst,
            FFTW_BACKWARD, fftw_flags);
        else
          plan = fftw_plan_dft(
            size.size(), idx, 
            (fftw_complex*) src, (fftw_complex*) dst,
            FFTW_BACKWARD, fftw_flags);
      }
      else
      {
        if constexpr(std::is_same_v<T, float>)
          plan = fftwf_plan_dft(
            size.size(), idx, 
            (fftwf_complex*) src, (fftwf_complex*) dst,
            FFTW_FORWARD, fftw_flags);
        else
          plan = fftw_plan_dft(
            size.size(), idx, 
            (fftw_complex*) src, (fftw_complex*) dst,
            FFTW_FORWARD, fftw_flags);
      }
    }
  }

  ~FftwTestWrapper()
  {
    if constexpr(std::is_same_v<T, float>)
      fftwf_destroy_plan(plan);
    else
      fftw_destroy_plan(plan);
  }

  void transform()
  {
    Int n = product(this->size);
    if constexpr(std::is_same_v<T, float>)
      fftwf_execute(plan);
    else
      fftw_execute(plan);
  }
};

template<typename T, bool is_real_, bool is_inverse_>
struct FftwSplitTestWrapper :
  public SplitWrapperBase<T, FftwSplitImOff, HalvedDimLast, is_real_, is_inverse_>
{
  using Base =
    SplitWrapperBase<T, FftwSplitImOff, HalvedDimLast, is_real_, is_inverse_>;

  static constexpr bool is_real = is_real_;
  static constexpr bool is_inverse = is_inverse_;
  using value_type = T;
  std::conditional_t<std::is_same_v<T, float>, fftwf_plan, fftw_plan> plan;

  unsigned fftw_flags;

  FftwSplitTestWrapper(
    const std::vector<Int>& size, Int simd_impl, Int is_bench)
  : Base(size), fftw_flags(is_bench ? FFTW_PATIENT : FFTW_ESTIMATE)
  {
    auto src = this->src;
    auto dst = this->dst;

    fftw_iodim dims[maxdim];
    Int src_stride = 1;
    Int dst_stride = 1;
    for(Int i = size.size() - 1; i >= 0; i--)
    {
      ptrdiff_t n = size[i];
      dims[i].n = n;
      dims[i].is = src_stride;
      dims[i].os = dst_stride;
      if(is_real && i == size.size() - 1)
      {
        if(is_inverse)
        {
          src_stride *= n / 2 + 1;
          dst_stride *= n;
        }
        else
        {
          src_stride *= n;
          dst_stride *= n / 2 + 1;
        }
      }
      else
      {
        src_stride *= n;
        dst_stride *= n;
      }
    }

    if constexpr(is_real_)
    {
      if constexpr(is_inverse_)
      {
        if constexpr(std::is_same_v<T, float>)
          plan = fftwf_plan_guru_split_dft_c2r(
            size.size(), dims,
            0, nullptr,
            src, src + this->im_off, dst,
            fftw_flags);
        else
          plan = fftw_plan_guru_split_dft_c2r(
            size.size(), dims,
            0, nullptr,
            src, src + this->im_off, dst,
            fftw_flags);
      }
      else
      {
        if constexpr(std::is_same_v<T, float>)
          plan = fftwf_plan_guru_split_dft_r2c(
              size.size(), dims,
              0, nullptr,
              src, dst, dst + this->im_off,
              fftw_flags);
        else
          plan = fftw_plan_guru_split_dft_r2c(
              size.size(), dims,
              0, nullptr,
              src, dst, dst + this->im_off,
              fftw_flags);
      }
    }
    else
    {
      Int n = product(this->size);
      Int re_off = is_inverse ? n : 0;
      Int im_off = is_inverse ? 0 : n;
      if constexpr(std::is_same_v<T, float>)
        plan = fftwf_plan_guru_split_dft(
            size.size(), dims,
            0, nullptr,
            src + re_off, src + im_off, dst + re_off, dst + im_off,
            fftw_flags);
      else
        plan = fftw_plan_guru_split_dft(
            size.size(), dims,
            0, nullptr,
            src + re_off, src + im_off, dst + re_off, dst + im_off,
            fftw_flags);
    }
  }

  ~FftwSplitTestWrapper()
  {
    if constexpr(std::is_same_v<T, float>)
      fftwf_destroy_plan(plan);
    else
      fftw_destroy_plan(plan);
  }

  void transform()
  {
    Int n = product(this->size);
    if constexpr(std::is_same_v<T, float>)
      fftwf_execute(plan);
    else
      fftw_execute(plan);
  }
};
#endif

template<typename T, bool is_inverse_>
struct ReferenceFft :
  public InterleavedWrapperBase<T, void, false, is_inverse_>
{
  struct Complex
  {
    T re;
    T im;
    Complex mul_neg_i() { return {im, -re}; }
    Complex adj() { return {re, -im}; }
    Complex operator+(Complex other)
    {
      return {re + other.re, im + other.im};
    }

    Complex operator-(Complex other)
    {
      return {re - other.re, im - other.im};
    }

    Complex operator*(Complex other)
    {
      return {
        re * other.re - im * other.im,
        re * other.im + im * other.re};
    }

    Complex operator*(T other)
    {
      return {re * other, im * other};
    }
  };

  static Complex load(const T* ptr) { return { ptr[0], ptr[1] }; }
  static void store(Complex val, T* ptr) { ptr[0] = val.re; ptr[1] = val.im; }

  struct Onedim
  {
    std::vector<Complex> twiddle;
    std::vector<T> working;
    Int n;

    Onedim(Int n) : n(n), working(2 * n), twiddle(n / 2)
    {
      auto pi = std::acos(T(-1));
      for(Int i = 0; i < n / 2; i++)
      {
        auto phi = (is_inverse_ ? 1 : -1) * i * pi / (n / 2);
        twiddle[i] = {std::cos(phi), std::sin(phi)};
      }
    }

    void transform(T* src, T* dst)
    {
      std::copy_n(src, 2 * n, dst);
      for(Int dft_size = 1; dft_size < n; dft_size *= 2)
      {
        static constexpr Int stride = 2;

        std::copy_n(dst, 2 * n, &working[0]);
        Int twiddle_stride = n / 2 / dft_size;
        for(Int i = 0; i < n / 2; i += dft_size)
        {
          Int src_i = i;
          Int dst_i = 2 * i;
          Int twiddle_i = 0;
          for(; src_i < i + dft_size;)
          {
            auto a = load(&working[0] + src_i * stride);
            auto b = load(&working[0] + (src_i + n / 2) * stride);

            auto mul = twiddle[twiddle_i] * b;
            store(a + mul, dst + dst_i * stride);
            store(a - mul, dst + (dst_i + dft_size) * stride);
            src_i++;
            dst_i++;
            twiddle_i += twiddle_stride;
          }  
        }
      }
    }
  };

  static const bool is_real = false;
  static const bool is_inverse = is_inverse_;
  typedef T value_type;
  std::vector<Onedim> onedim;
  std::vector<T> working;

  ReferenceFft(const std::vector<Int>& size, Int simd_impl, bool is_bench)
    : InterleavedWrapperBase<T, void, false, is_inverse_>(size)
  {
    for(auto e : size) onedim.emplace_back(e);
  }

  void transform()
  {
    std::copy_n(this->src, 2 * product(this->size), this->dst);
    for(Int dim = 0; dim < this->size.size(); dim++)
    {
      Int s[maxdim];
      std::copy_n(&this->size[0], this->size.size(), s);
      s[dim] = 1;
      auto dst_view = create_view(this->dst, this->size, 1);
      for(IterateMultidim it(this->size.size(), s); !it.empty(); it.advance())
      {
        Int n = this->size[dim];
        working.resize(2 * n);
        auto p = dst_view.ptr(it.idx);
        if(dim == this->size.size() - 1)
        {
          onedim[dim].transform(p, p);
        }
        else
        {
          for(Int i = 0; i < n; i++)
          {
            working[2 * i] = p[i * dst_view.stride[dim]];
            working[2 * i + 1] = p[i * dst_view.stride[dim] + 1];
          }

          onedim[dim].transform(&working[0], &working[0]); 
          
          for(Int i = 0; i < n; i++)
          {
            p[i * dst_view.stride[dim]] = working[2 * i];
            p[i * dst_view.stride[dim] + 1] = working[2 * i + 1];
          }
        }
      }
    }
  }
};

struct TestResult
{
  double flops = 0.0;
  double error = 0.0;
  double error_factor = 0.0;
  double time_per_element = 0.0;
  uint64_t element_iterations = 0;
  bool success = true;
};

template<typename Fft>
TestResult bench(
  const std::vector<Int>& size, double requested_operations, Int simd_impl)
{
  Int n = product(size);
  typedef typename Fft::value_type T;
  Fft fft(size, simd_impl, true);

  T* src = alloc_array<T>(2 * n);
  for(Int i = 0; i < n * 2; i++) src[i] = 0.0f;
  fft.set_input(src);

  double const_part = Fft::is_real ? 2.5 : 5.0;
  auto iter = std::max<uint64_t>(requested_operations / (const_part * n * log2(n)), 1);
  auto operations = iter * (const_part * n * log2(n));

  auto t0 = high_resolution_clock::now();
  for(int64_t i = 0; i < iter; i++)
  {
    fft.transform();
    //int64_t j = i / (iter / 10);
    //if(j * (iter / 10) == i) { printf("%d ", j); fflush(stdout); }
  }

  auto t1 = high_resolution_clock::now(); 

  TestResult r;
  r.flops = operations / to_seconds(t1 - t0);
  r.element_iterations = iter * uint64_t(n);
  r.time_per_element = to_seconds(t1 - t0) / iter / n;
  return r;
}

template<typename Fft0, typename Fft1>
TestResult compare(const std::vector<Int>& size, Int simd_impl)
{
  static_assert(Fft0::is_inverse == Fft1::is_inverse, "");
  typedef typename Fft0::value_type T;
  Int n = product(size);
  Fft0 fft0(size, simd_impl, false);
  Fft1 fft1(size, simd_impl, false);
  T* src = alloc_array<T>(2 * n);
  T* dst0 = alloc_array<T>(2 * n);
  T* dst1 = alloc_array<T>(2 * n);
  
  std::mt19937 mt;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for(Int i = 0; i < n * 2; i++) src[i] = dist(mt);
  
  //array_ipc::send("s_before", src, 2 * n);

  if(Fft0::is_real || Fft1::is_real)
  {
    auto src_re = create_view(src, size, 0);
    auto src_im = create_view(src + n, size, 0);
    if(Fft0::is_inverse)
    {
      std::vector<T> tmp(2 * n);
      auto tmp_re = create_view(&tmp[0], size, 0);
      auto tmp_im = create_view(&tmp[n], size, 0);
      copy_view(src_re, tmp_re);
      copy_view(src_im, tmp_im);
      for(IterateMultidim it(src_re.ndim, src_re.size); !it.empty(); it.advance())
      {
        Int midx[maxdim];
        for(Int i = 0; i < src_re.ndim; i++)
          midx[i] = mirror_idx(src_re.size[i], it.idx[i]);

        *src_re.ptr(it.idx) = *tmp_re.ptr(it.idx) + *tmp_re.ptr(midx);
        *src_re.ptr(midx  ) = *tmp_re.ptr(it.idx) + *tmp_re.ptr(midx);

        *src_im.ptr(it.idx) = *tmp_im.ptr(midx  ) - *tmp_im.ptr(it.idx);
        *src_im.ptr(midx  ) = *tmp_im.ptr(it.idx) - *tmp_im.ptr(midx);
      }
    }
    else
      for(Int i = n; i < n * 2; i++) src[i] = T(0);
  }
  
  array_ipc::send("s", src, 2 * n);

  fft0.set_input(src);
  fft1.set_input(src);

  fft0.transform();
  fft1.transform();

  fft0.get_output(dst0);
  fft1.get_output(dst1);

  array_ipc::send("d0", dst0, 2 * n);
  array_ipc::send("d1", dst1, 2 * n);

  auto sum_sumsq = T(0);
  auto diff_sumsq = T(0);
  for(Int i = 0; i < 2 * n; i++)
  {
    sum_sumsq += sq(dst0[i]);
    sum_sumsq += sq(dst1[i]);
    diff_sumsq += sq(dst1[i] - dst0[i]);
  }

  dealloc(src);
  dealloc(dst0);
  dealloc(dst1);

  TestResult r;
  r.error = std::sqrt(diff_sumsq / sum_sumsq);
  r.error_factor = std::sqrt(log2(n));
  return r;
}

extern "C" void* aligned_alloc(size_t, size_t);

std::istream& operator>>(std::istream& stream, SizeRange& size_range)
{
  std::string s(std::istreambuf_iterator<char>(stream), {});
  const char* p = s.c_str();

  char* next_p;
  size_range.begin = strtol(p, &next_p, 0);

  if(next_p == p)
  {
    stream.setstate(std::ios_base::failbit);
    return stream;
  }

  p = next_p;

  if(*p == 0)
  {
    size_range.end = size_range.begin + 1;
    return stream;
  }

  if(*p != '-')
  {
    stream.setstate(std::ios_base::failbit);
    return stream;
  }

  p++;

  size_range.end = strtol(p, &next_p, 0);

  if(next_p == p || *next_p != 0)
  {
    stream.setstate(std::ios_base::failbit);
    return stream;
  }

  return stream;
}

std::istream& operator>>(std::istream& stream, SimdImpl& simd_impl)
{
  std::string s(std::istreambuf_iterator<char>(stream), {});

  if(s == "auto") simd_impl.val = 0;
  else if(s == "scalar") simd_impl.val = 1;
  else if(s == "sse2") simd_impl.val = 2;
  else if(s == "avx2") simd_impl.val = 3;
  else if(s == "avx512f") simd_impl.val = 4;
  else if(s == "neon") simd_impl.val = 5;
  else stream.setstate(std::ios_base::failbit);

  return stream;
}

OptionParser::Result parse_options(int argc, char** argv, Options* dst)
{
  OptionParser parser;
  parser.add_switch("-d", "Use double precision.", &dst->is_double);
  parser.add_switch("-r", "Test real transform.", &dst->is_real);
  parser.add_switch("-i", "Test inverse transform.", &dst->is_inverse);
  parser.add_switch("-b", "Perform a benchmark.", &dst->is_bench);
  parser.add_optional_flag(
    "-p", "Required relative precision", &dst->precision);

  dst->num_ops = 1e11;
  parser.add_optional_flag(
    "-n", "Approximage number of operations to perform during benchmarking.",
    &dst->num_ops);

  dst->simd_impl = {afft_auto};
  parser.add_optional_flag(
    "--simd", "Which SIMD implementation to use", &dst->simd_impl);

  parser.add_positional(
    "implementation", "Fft implementation to test.", &dst->implementation);

  parser.add_multi_positional("size", "Data size.", 1, 64, &dst->size);
  return parser.parse(argc, argv);
}

template<typename Fft>
TestResult test_or_bench4(const Options& opt, const std::vector<Int>& lsz)
{
  std::vector<Int> size;
  for(auto e : lsz) size.push_back(1 << e);

  if(opt.is_bench)
    return bench<Fft>(size, opt.num_ops, opt.simd_impl.val);
  else
    //TODO: Use long double for ReferenceFft
    return 
      compare<ReferenceFft<double, Fft::is_inverse>, Fft>(
        size, opt.simd_impl.val);
}

template<typename ET, bool is_real, bool is_inverse>
TestResult test_or_bench3(const Options& opt, const std::vector<Int>& lsz)
{
  if(opt.implementation == "fft")
    return test_or_bench4<TestWrapper<ET, is_real, is_inverse>>(opt, lsz);
#ifdef HAVE_FFTW
  else if(opt.implementation == "fftw")
    return
      test_or_bench4<FftwTestWrapper<ET, is_real, is_inverse>>(opt, lsz);
  else if(opt.implementation == "fftw-split")
    return
      test_or_bench4<FftwSplitTestWrapper<ET, is_real, is_inverse>>(opt, lsz);
#endif
  else
    abort();
}

template<typename ET, bool is_real>
TestResult test_or_bench2(const Options& opt, const std::vector<Int>& lsz)
{
  if(opt.is_inverse)
    return test_or_bench3<ET, is_real, true>(opt, lsz);
  else
    return test_or_bench3<ET, is_real, false>(opt, lsz);
}

template<typename ET>
TestResult test_or_bench1(const Options& opt, const std::vector<Int>& lsz)
{
  if(opt.is_real)
    return test_or_bench2<ET, true>(opt, lsz);
  else
    return test_or_bench2<ET, false>(opt, lsz);
}

TestResult test_or_bench0(const Options& opt, const std::vector<Int>& lsz)
{
  if(opt.is_double)
    return test_or_bench1<double>(opt, lsz);
  else
    return test_or_bench1<float>(opt, lsz);
}


void stream_printf(std::ostream& out, const char* format, ...)
{
  char buf[1 << 10];

  va_list args;
  va_start(args, format);
  int n = vsnprintf(buf, sizeof(buf), format, args);
  va_end(args); 

  if(n > 0) out.write(buf, n);
}

bool run_test(const Options& opt, std::ostream& out)
{
  std::vector<Int> sz;
  for(auto& e : opt.size) sz.push_back(e.begin);

  while(true)
  {
    for(auto e : sz) stream_printf(out, "%2d ", e);
    out.flush();

    if(opt.is_bench)
    {
      auto r = test_or_bench0(opt, sz);

      stream_printf(
        out, "%f GFLOPS  %f ns\n", r.flops * 1e-9, r.time_per_element * 1e9);
    }
    else
    {
      TestResult test_result = test_or_bench0(opt, sz);

      stream_printf(out, "%g\n", test_result.error);

      if(opt.precision)
      {
        double max_error = *opt.precision * test_result.error_factor;
        if(test_result.error > max_error)
        {
          out <<
            "Error exceeds maximal allowed value, which is " <<
            max_error <<  "." << std::endl;

          return false;
        }
      }
    }

    bool break_outer = true;
    for(Int i = sz.size() - 1; i >= 0; i--)
      if(sz[i] + 1 < opt.size[i].end)
      {
        break_outer = false;
        sz[i] = sz[i] + 1;
        break;
      }
      else
        sz[i] = opt.size[i].begin;

    if(break_outer) break;
  }

  return true;
}
