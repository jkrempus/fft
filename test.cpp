#include "misc/array_ipc.h"

#include "fft_core.h"

#include <string>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <sys/time.h>
#include <fstream>
#include <unistd.h>
#include <random>
#include <cstdint>
#include <unordered_set>
#include <sstream>

#ifdef HAVE_FFTW
#include "fftw3.h"
#endif

extern "C" void* valloc(size_t);

double get_time()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

template<typename T_>
void dump(T_* ptr, Int n, const char* name, ...)
{
  char buf[1 << 10];

  va_list args;
  va_start(args, name);
  vsprintf(buf, name, args);
  va_end(args); 

  std::ofstream(buf, std::ios_base::binary).write((char*) ptr, sizeof(T_) * n);
}

#if 0
char mem[1 << 30];
char* mem_ptr = mem;

void* alloc(Int n)
{
#if 0
  void* r = valloc(n);
  //printf("valloc_ allocated range %p %p\n", r, (void*)(Uint(r) + n));
  return r;
#else
  mem_ptr = (char*) align_size(Uint(mem_ptr));
  auto r = mem_ptr;
  mem_ptr += n;
  return r;
#endif
}

void dealloc(void* p)
{
  //free(p);
  mem_ptr = mem;
}
#else
void* alloc(Int n)
{
  void* r = valloc(n);
  return r;
}

void dealloc(void* p)
{
  free(p);
}
#endif

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

template<typename A>
void print_range(const A& a)
{
  for(auto& e : a) std::cout << e << " ";
  std::cout << std::endl;
}

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

template<typename V, typename Cf>
constexpr Int chunk_size()
{
  return
    SameType<cf::Scal<V>, Cf>::value ? 1 :
    SameType<cf::Vec<V>, Cf>::value ? V::vec_size : 0;
}

template<typename V, typename Cf>
constexpr Int get_im_offset(Int split_im_offset)
{
  return chunk_size<V, Cf>() ? chunk_size<V, Cf>() : split_im_offset;
}

template<typename T>
void print_view(const View<T>& v)
{
  printf("view %p (");
  for(Int i = 0; i < v.ndim; i++) printf("%s%d", i == 0 ? "" : " ", v.size[i]);
  printf(") (");
  for(Int i = 0; i < v.ndim - 1; i++) printf("%s%d", i == 0 ? "" : " ", v.stride[i]);
  printf(")\n");
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
  //printf("create_view "); print_view(r);
  return r;
}

template<typename T>
T product(const std::vector<T>& v)
{
  T r(1);
  for(auto& e : v) r *= e;
  return r;
}

template<typename T, bool is_real, bool is_inverse>
struct SplitWrapperBase { };

template<typename T, bool is_inverse_>
struct SplitWrapperBase<T, false, is_inverse_>
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

template<typename T>
struct SplitWrapperBase<T, true, false>
{
  Int im_off;
  std::vector<Int> size;
  std::vector<Int> symmetric_size;
  T* src;
  T* dst;
  
  SplitWrapperBase(const std::vector<Int>& size, Int im_off) :
    im_off(im_off),
    size(size),
    src(alloc_array<T>(product(size))),
    dst(alloc_array<T>(2 * im_off))
  {
    symmetric_size = size;
    symmetric_size.front() = symmetric_size.front() / 2 + 1;
  }

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

    copy_symmetric_view<false>(
      create_view(dst, symmetric_size, 0),
      create_view(p, size, 0));
  
    copy_symmetric_view<true>(
      create_view(dst + im_off, symmetric_size, 0),
      create_view(p + n, size, 0));
  }
};

template<typename T>
struct SplitWrapperBase<T, true, true>
{
  Int im_off;
  std::vector<Int> size;
  std::vector<Int> symmetric_size;
  T* src;
  T* dst;

  SplitWrapperBase(const std::vector<Int>& size, Int im_off) :
    im_off(im_off),
    size(size),
    dst(alloc_array<T>(product(size))),
    src(alloc_array<T>(2 * im_off))
  {
    symmetric_size = size;
    symmetric_size.front() = symmetric_size.front() / 2 + 1;
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
      create_view(p, symmetric_size, 0),
      create_view(src, symmetric_size, 0));

    copy_view(
      create_view(p + product(size), symmetric_size, 0),
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


template<typename T, bool is_real, bool is_inverse>
struct InterleavedWrapperBase { };

template<typename T, bool is_inverse_>
struct InterleavedWrapperBase<T, false, is_inverse_>
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

template<typename T>
struct InterleavedWrapperBase<T, true, false>
{
  std::vector<Int> size;
  std::vector<Int> symmetric_size;
  T* src;
  T* dst;
  
  InterleavedWrapperBase(const std::vector<Int>& size) : size(size)
  {
    symmetric_size = size;
    symmetric_size.back() = symmetric_size.back() / 2 + 1;
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

template<typename T>
struct InterleavedWrapperBase<T, true, true>
{
  std::vector<Int> size;
  std::vector<Int> symmetric_size;
  Int im_offset;
  T* src;
  T* dst;
  
  InterleavedWrapperBase(const std::vector<Int>& size) :
    size(size),
    im_offset(align_size<T>(product(size) / 2 + 1))
  {
    symmetric_size = size;
    symmetric_size.back() = symmetric_size.back() / 2 + 1;
    src = (T*) alloc(2 * sizeof(T) * im_offset);
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

template<
  typename V,
  template<typename> class CfT,
  bool is_real,
	bool is_inverse>
struct TestWrapper { };

template<typename V, template<typename> class CfT>
struct TestWrapper<V, CfT, false, false>
: public SplitWrapperBase<typename V::T, false, false>
{
  static const bool is_real = false;
  static const bool is_inverse = false;
  VEC_TYPEDEFS(V);
  typedef T value_type;
  MultidimState<T>* state;
  TestWrapper(const std::vector<Int>& size) :
    SplitWrapperBase<T, false, false>(size),
    state(multidim_fft_state<V, CfT, CfT>(
      size.size(),
      &size[0],
      alloc(multidim_state_memory_size<V>(size.size(), &size[0])))) {}

  ~TestWrapper() { dealloc(state); }
  void transform() { multidim_fft<T>(state, this->src, this->dst); }
};

template<typename V, template<typename> class CfT>
struct TestWrapper<V, CfT, false, true>
: public SplitWrapperBase<typename V::T, false, true>
{
  static const bool is_real = false;
  static const bool is_inverse = true;
  VEC_TYPEDEFS(V);
  typedef T value_type;
  InverseMultidimState<T>* state;
  TestWrapper(const std::vector<Int>& size) :
    SplitWrapperBase<T, false, true>(size),
    state(inverse_multidim_fft_state<V, CfT, CfT>(
      size.size(),
      &size[0],
      alloc(inverse_multidim_state_memory_size<V>(size.size(), &size[0])))) {}

  ~TestWrapper() { dealloc(state); }
  void transform() { inverse_multidim_fft<T>(state, this->src, this->dst); }
};

template<typename V, template<typename> class CfT>
struct TestWrapper<V, CfT, true, false>
: public SplitWrapperBase<typename V::T, true, false>
{
  static const bool is_real = true;
  static const bool is_inverse = false;
  VEC_TYPEDEFS(V);
  typedef typename V::T value_type;
  RealMultidimState<T>* state;

  static Int im_offset(const std::vector<Int>& size)
  {
    Int inner = product(ptr_range(&size[1], size.size() - 1));
    return align_size<T>((size[0] / 2 + 1) * inner);
  }

  TestWrapper(const std::vector<Int>& size) :
    SplitWrapperBase<T, true, false>(size, im_offset(size)),
    state(real_multidim_fft_state<V, CfT>(size.size(), &size[0], 
      alloc(real_multidim_state_memory_size<V>(size.size(), &size[0])))) {}

  ~TestWrapper() { dealloc(state); }

  void transform()
  {
    real_multidim_fft(state, this->src, this->dst);
  }
};

template<typename V, template<typename> class CfT>
struct TestWrapper<V, CfT, true, true>
: public SplitWrapperBase<typename V::T, true, true>
{
  static const bool is_real = true;
  static const bool is_inverse = true;
  VEC_TYPEDEFS(V);
  typedef T value_type;
  InverseRealState<T>* state;
  
  static Int im_offset(const std::vector<Int>& size)
  {
    Int inner = product(ptr_range(&size[1], size.size() - 1));
    return align_size<T>((size[0] / 2 + 1) * inner);
  }

  TestWrapper(const std::vector<Int>& size) :
    SplitWrapperBase<T, true, true>(size, im_offset(size)),
    state(inverse_rfft_state<V, CfT>(
        size[0], alloc(inverse_rfft_state_memory_size<V>(size[0])))) {}

  ~TestWrapper() { dealloc(state); }

  void transform() { inverse_rfft(state, this->src, this->dst); }
};

#ifdef HAVE_FFTW
template<bool is_real, bool is_inverse, typename T>
fftwf_plan make_plan(const std::vector<Int>& size, T* src, T* dst);

const unsigned fftw_flags = FFTW_PATIENT;

template<> fftwf_plan make_plan<false, false, float>(
  const std::vector<Int>& size, float* src, float* dst)
{
  int idx[maxdim];
  std::copy_n(&size[0], size.size(), idx);
  return fftwf_plan_dft(
    size.size(), idx, 
    (fftwf_complex*) src, (fftwf_complex*) dst,
    FFTW_FORWARD, fftw_flags);
}

template<> fftwf_plan make_plan<false, true, float>(
  const std::vector<Int>& size, float* src, float* dst)
{
  int idx[maxdim];
  std::copy_n(&size[0], size.size(), idx);
  return fftwf_plan_dft(
    size.size(), idx, 
    (fftwf_complex*) src, (fftwf_complex*) dst,
    FFTW_BACKWARD, fftw_flags);
}

template<> fftwf_plan make_plan<true, false, float>(
  const std::vector<Int>& size, float* src, float* dst)
{
  int idx[maxdim];
  std::copy_n(&size[0], size.size(), idx);
  return fftwf_plan_dft_r2c(
    size.size(), idx, 
    src, (fftwf_complex*) dst,
    fftw_flags);
}

template<> fftwf_plan make_plan<true, true, float>(
  const std::vector<Int>& size, float* src, float* dst)
{
  int idx[maxdim];
  std::copy_n(&size[0], size.size(), idx);
  return fftwf_plan_dft_c2r(
    size.size(), idx, 
    (fftwf_complex*) src, dst,
    fftw_flags);
}

template<typename T, bool is_real_, bool is_inverse_>
struct FftwTestWrapper : public InterleavedWrapperBase<T, is_real_, is_inverse_>
{
  static const bool is_real = is_real_;
  static const bool is_inverse = is_inverse_;
  typedef float value_type;
  fftwf_plan plan;

  FftwTestWrapper(const std::vector<Int>& size)
    : InterleavedWrapperBase<T, is_real, is_inverse_>(size)
  {
    plan = make_plan<is_real, is_inverse_>(size, this->src, this->dst);
  }

  ~FftwTestWrapper() { fftwf_destroy_plan(plan); }

  void transform()
  {
    Int n = product(this->size);
    fftwf_execute(plan);
  }
};
#endif

template<typename T, bool is_inverse_>
struct ReferenceFft : public InterleavedWrapperBase<T, false, is_inverse_>
{
  struct Onedim
  {
    std::vector<Complex<T>> twiddle;
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
      copy(src, 2 * n, dst);
      for(Int dft_size = 1; dft_size < n; dft_size *= 2)
      {
        copy(dst, 2 * n, &working[0]);
        typedef complex_format::Scal<Scalar<T>> CF;
        Int twiddle_stride = n / 2 / dft_size;
        for(Int i = 0; i < n / 2; i += dft_size)
        {
          Int src_i = i;
          Int dst_i = 2 * i;
          Int twiddle_i = 0;
          for(; src_i < i + dft_size;)
          {
            auto a = CF::load(&working[0] + src_i * CF::stride, 0);
            auto b = CF::load(&working[0] + (src_i + n / 2) * CF::stride, 0);
            auto mul = twiddle[twiddle_i] * b;
            CF::store(a + mul, dst + dst_i * CF::stride, 0);
            CF::store(a - mul, dst + (dst_i + dft_size) * CF::stride, 0);
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

  ReferenceFft(const std::vector<Int>& size)
    : InterleavedWrapperBase<T, false, is_inverse_>(size)
  {
    for(auto e : size) onedim.emplace_back(e);
  }

  void transform()
  {
    copy(this->src, 2 * product(this->size), this->dst);
    for(Int dim = 0; dim < this->size.size(); dim++)
    {
      Int s[maxdim];
      copy(&this->size[0], this->size.size(), s);
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

template<typename Fft>
double bench(const std::vector<Int>& size, double requested_operations)
{
  Int n = product(size);
  typedef typename Fft::value_type T;
  Fft fft(size);

  T* src = alloc_array<T>(2 * n);
  for(Int i = 0; i < n * 2; i++) src[i] = 0.0f;
  fft.set_input(src);

  double const_part = Fft::is_real ? 2.5 : 5.0;
  auto iter = max<uint64_t>(requested_operations / (const_part * n * log2(n)), 1);
  auto operations = iter * (const_part * n * log2(n));

  double t0 = get_time();
  for(int64_t i = 0; i < iter; i++)
  {
    fft.transform();
    int64_t j = i / (iter / 10);
    if(j * (iter / 10) == i)
    {
      printf("%d ", j);
      fflush(stdout);
    }
  }
  double t1 = get_time(); 
  
  return operations / (t1 - t0);
}

template<typename Fft0, typename Fft1>
typename Fft0::value_type compare(const std::vector<Int>& size)
{
  static_assert(Fft0::is_inverse == Fft1::is_inverse, "");
  typedef typename Fft0::value_type T;
  Int n = product(size);
  Fft0 fft0(size);
  Fft1 fft1(size);
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
  
  //array_ipc::send("s", src, 2 * n);

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

  return std::sqrt(diff_sumsq / sum_sumsq);
}

extern "C" void* aligned_alloc(size_t, size_t);

#ifdef NO_SIMD
typedef Scalar<float> V;
#elif defined __ARM_NEON__
typedef Neon V;
#elif defined __AVX__
typedef AvxFloat V;
#elif defined __SSE2__
typedef SseFloat V;
#else
typedef Scalar<float> V;
#endif

template<typename T>
using CfT = complex_format::Split<T>;

struct Options
{
  std::unordered_set<std::string> flags;
  std::vector<std::string> positional;
};

Options parse_options(int argc, char** argv)
{
  Options r;
  for(Int i = 1; i < argc; i++)
    if(argv[i][0] == '-')
      r.flags.emplace(argv[i]);
    else
      r.positional.emplace_back(argv[i]);

  return r;
}

std::pair<std::vector<Int>, std::vector<Int>>
parse_sizes(const std::vector<std::string>& positional)
{
  std::pair<std::vector<Int>, std::vector<Int>> r;
  for(Int i = 1; i < positional.size(); i++)
  {
    auto dot_pos = positional[i].find("-");
    if(dot_pos == std::string::npos)
    {
      Int sz = std::stoi(positional[i]);
      r.first.push_back(sz);
      r.second.push_back(sz + 1);
    }
    else
    {
      r.first.push_back(std::stoi(positional[i].substr(0, dot_pos)));
      r.second.push_back(std::stoi(positional[i].substr(dot_pos + 1)));
    }
  }

  return r;
}

template<typename Fft>
double test_or_bench3(
  const std::string& impl,
  const std::vector<Int>& lsz,
  const std::unordered_set<std::string>& flags)
{
  std::vector<Int> size;
  for(auto e : lsz) size.push_back(1 << e);

  if(flags.count("-b"))
    return bench<Fft>(size, 1e11);
  else
    //TODO: Use long double for ReferenceFft
    return compare<ReferenceFft<double, Fft::is_inverse>, Fft>(size);
}

template<bool is_real, bool is_inverse>
double test_or_bench2(
  const std::string& impl,
  const std::vector<Int>& lsz,
  const std::unordered_set<std::string>& flags)
{
  if(impl == "fft")
    return test_or_bench3<TestWrapper<V, CfT, is_real, is_inverse>>(impl, lsz, flags);
#ifdef HAVE_FFTW
  else if(impl == "fftw")
    return
      test_or_bench3<FftwTestWrapper<float, is_real, is_inverse>>(impl, lsz, flags);
#endif
  else
    abort();
}

template<bool is_real>
double test_or_bench1(
  const std::string& impl,
  const std::vector<Int>& lsz,
  const std::unordered_set<std::string>& flags)
{
  if(flags.count("-i"))
    return test_or_bench2<is_real, true>(impl, lsz, flags);
  else
    return test_or_bench2<is_real, false>(impl, lsz, flags);
}

double test_or_bench0(
  const std::string& impl,
  const std::vector<Int>& lsz,
  const std::unordered_set<std::string>& flags)
{
  if(flags.count("-r"))
    return test_or_bench1<true>(impl, lsz, flags);
  else
    return test_or_bench1<false>(impl, lsz, flags);
}

int main(int argc, char** argv)
{
  Options opt = parse_options(argc, argv);
  if(opt.positional.size() < 2) abort();

  auto size_range = parse_sizes(opt.positional);

  auto sz = size_range.first;
  while(true)
  {
    for(auto e : sz) printf("%2d ", e);
    fflush(stdout);
    if(opt.flags.count("-b") > 0)
      printf("%f GFLOPS\n", test_or_bench0(opt.positional[0], sz, opt.flags) * 1e-9);
    else
      printf("%g\n", test_or_bench0(opt.positional[0], sz, opt.flags));

    bool break_outer = true;
    for(Int i = sz.size() - 1; i >= 0; i--)
      if(sz[i] + 1 < size_range.second[i])
      {
        break_outer = false;
        sz[i] = sz[i] + 1;
        break;
      }
      else
        sz[i] = size_range.first[i];

    if(break_outer) break;
  }

  return 0;
}
