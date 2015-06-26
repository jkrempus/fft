#include "fft_core.h"

#include <string>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
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

template<typename T>
T* alloc_complex_array(Int n)
{
  return (T*) valloc(2 * n * sizeof(T));
}

template<typename T> struct View
{
  static const int maxdim = 64;
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

  View get_plane(Int i) const
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
};

Int chunked_index(Int i, Int chunk_size)
{
  if(chunk_size == 0)
    return i;
  else
    return 2 * i - (i & (chunk_size - 1));
}

template<typename T, typename U>
void copy_view(const View<T>& src, const View<U>& dst)
{
  if(src.ndim == 1)
    for(Int i = 0; i < std::min(src.size[0], dst.size[0]); i++)
      dst.data[chunked_index(i, dst.chunk_size)] =
        src.data[chunked_index(i, src.chunk_size)];
  else
    for(Int i = 0; i < std::min(src.size[0], dst.size[0]); i++)
      copy_view(src.get_plane(i), dst.get_plane(i));
}

template<typename T, typename U>
void fill_view(const T& value, const View<U>& dst)
{
  if(dst.ndim == 1)
    for(Int i = 0; i < dst.size[0]; i++)
      dst.data[chunked_index(i, dst.chunk_size)] = value;
  else
    for(Int i = 0; i < dst.size[0]; i++)
      fill_view(value, dst.get_plane(i));
}

template<bool is_antisym, typename T, typename U>
void copy_symmetric_view(const View<T>& src, const View<U>& dst)
{
  struct Recurse
  {
    View<T> src;
    View<U> dst;
    Int idx[View<T>::maxdim];
    void f(Int idx_len)
    {
      for(Int i = 0; i < dst.size[idx_len]; i++)
      {
        idx[idx_len] = i;
        if(idx_len == src.ndim - 1)
        {
          T* s = src.data;
          U* d = dst.data;
          bool mirror = false;
          for(Int j = 0; j < src.ndim; j++)
            mirror = mirror || idx[j] >= src.size[j];

          for(Int j = 0; j < src.ndim - 1; j++)
          {
            s += src.stride[j] * (mirror ? dst.size[j] - idx[j] : idx[j]);
            d += dst.stride[j] * idx[j];
          }

          {
            Int j = src.ndim - 1;
            s += chunked_index(mirror ? dst.size[j] - idx[j] : idx[j], src.chunk_size);
            d += chunked_index(idx[j], dst.chunk_size);
          }

          *d = *s * T((mirror && is_antisym) ? -1 : 1);
        }
        else
          f(idx_len + 1);
      }
    } 
  };

  ((Recurse){src, dst}).f(0);
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
View<T> create_view(T* ptr, const std::vector<Int>& size, Int chunk_size)
{
  View<T> r;
  r.ndim = size.size();
  std::copy_n(&size[0], r.ndim, r.size); 
  Int s = chunk_size;
  for(Int i = r.ndim - 1; i > 0; i--)
  {
    s *= r.size[0];
    r.stride[i - 1] = 0;
  }

  r.data = ptr;
  r.chunk_size = chunk_size;
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
    src((T*) valloc(2 * sizeof(T) * product(size))),
    dst((T*) valloc(2 * sizeof(T) * product(size))) { }

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
    src((T*) valloc(2 * sizeof(T) * product(size))),
    dst((T*) valloc(2 * sizeof(T) * product(size))) { }

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
  
  InterleavedWrapperBase(const std::vector<Int>& size) :
    size(size),
    src((T*) valloc(sizeof(T) * product(size))),
    dst((T*) valloc(2 * sizeof(T) * (product(size) / 2 + 1)))
  {
    symmetric_size = size;
    symmetric_size.back() = symmetric_size.back() / 2 + 1;
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
    src = (T*) valloc(2 * sizeof(T) * im_offset);
    dst = (T*) valloc(sizeof(T) * product(size));
  }

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
  State<T>* state;
  TestWrapper(const std::vector<Int>& size) :
    SplitWrapperBase<T, false, false>(size),
    state(fft_state<V, CfT, CfT>(
      size[0], valloc(fft_state_memory_size<V>(size[0])))) {}

  ~TestWrapper() { free(fft_state_memory_ptr(state)); }
  void transform() { fft<T>(state, this->src, this->dst); }
};

template<typename V, template<typename> class CfT>
struct TestWrapper<V, CfT, false, true>
: public SplitWrapperBase<typename V::T, false, true>
{
  static const bool is_real = false;
  static const bool is_inverse = true;
  VEC_TYPEDEFS(V);
  typedef T value_type;
  State<T>* state;
  TestWrapper(const std::vector<Int>& size) :
    SplitWrapperBase<T, false, true>(size),
    state(inverse_fft_state<V, CfT, CfT>(
        size[0], valloc(inverse_fft_state_memory_size<V>(size[0])))) {}

  ~TestWrapper() { free(inverse_fft_state_memory_ptr(state)); }
  void transform() { inverse_fft<T>(state, this->src, this->dst); }
};

template<typename V, template<typename> class CfT>
struct TestWrapper<V, CfT, true, false>
{
  static const bool is_real = true;
  static const bool is_inverse = false;
  VEC_TYPEDEFS(V);
  typedef T value_type;
  RealState<T>* state;
  InverseRealState<T>* inverse_state;
  Int n;
  Int im_offset;
  T* src;
  T* dst;
  TestWrapper(const std::vector<Int>& size) :
    state(rfft_state<V, CfT>(size[0], valloc(rfft_state_memory_size<V>(size[0])))),
    n(size[0]),
    im_offset(align_size<T>(size[0] / 2 + 1))
  {
    src = (T*) valloc(n * sizeof(T));
    dst = (T*) valloc(2 * im_offset * sizeof(T));
    inverse_state = 
      inverse_rfft_state<V, CfT>(n, valloc(rfft_state_memory_size<V>(n)));
  }

  ~TestWrapper()
  {
    free(rfft_state_memory_ptr(state));
    free(src);
    free(dst);
  }

  void transform()
  {
    rfft(state, src, dst);
    inverse_rfft(inverse_state, dst, src);
  }

  template<typename U>
  void set_input(U* p)
  {
  }

  template<typename U>
  void get_output(U* p)
  {
  }
};

template<typename V, template<typename> class CfT>
struct TestWrapper<V, CfT, true, true>
{
  static const bool is_real = true;
  static const bool is_inverse = true;
  VEC_TYPEDEFS(V);
  typedef T value_type;
  InverseRealState<T>* state;
  Int n;
  Int im_offset;
  T* src;
  T* dst;
  TestWrapper(const std::vector<Int>& size) :
    im_offset(align_size<T>(size[0] / 2 + 1)),
    n(size[0]),
    state(inverse_rfft_state<V, CfT>(
        size[0], valloc(inverse_rfft_state_memory_size<V>(size[0]))))
  {
    src = (T*) valloc(2 * im_offset * sizeof(T));
    dst = (T*) valloc(n * sizeof(T));
  }

  ~TestWrapper()
  {
    free(inverse_rfft_state_memory_ptr(state));
    free(src);
    free(dst);
  }

  void transform() { inverse_rfft(state, src, dst); }

  template<typename U>
  void set_input(U* p)
  {
  }

  template<typename U>
  void get_output(U* p)
  {
  }
};

#ifdef HAVE_FFTW
template<bool is_real, bool is_inverse, typename T>
fftwf_plan make_plan(Int n, T* src, T* dst);

const unsigned fftw_flags = FFTW_PATIENT;

template<> fftwf_plan make_plan<false, false, float>(Int n, float* src, float* dst)
{
  return fftwf_plan_dft_1d(
    n, (fftwf_complex*) src, (fftwf_complex*) dst, FFTW_FORWARD, fftw_flags);
}

template<> fftwf_plan make_plan<false, true, float>(Int n, float* src, float* dst)
{
  return fftwf_plan_dft_1d(
    n, (fftwf_complex*) src, (fftwf_complex*) dst, FFTW_BACKWARD, fftw_flags);
}

template<> fftwf_plan make_plan<true, false, float>(Int n, float* src, float* dst)
{
  return fftwf_plan_dft_r2c_1d(n, src, (fftwf_complex*) dst, fftw_flags);
}

template<> fftwf_plan make_plan<true, true, float>(Int n, float* src, float* dst)
{
  return fftwf_plan_dft_c2r_1d(n, (fftwf_complex*) src, dst, fftw_flags);
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
    plan = make_plan<is_real, is_inverse_>(size[0], this->src, this->dst);
  }

  ~FftwTestWrapper() { fftwf_destroy_plan(plan); }

  void transform() { fftwf_execute(plan); }
};
#endif

template<typename T, bool is_inverse_>
struct ReferenceFft : public InterleavedWrapperBase<T, false, is_inverse_>
{
  struct Onedim
  {
    Complex<T>* twiddle;
    T* working;
    Int n;
    Onedim(Int n) : n(n)
    {
      working = new T[2 * n];
      twiddle = new Complex<T>[n / 2];
      auto pi = std::acos(T(-1));
      for(Int i = 0; i < n / 2; i++)
      {
        auto phi = (is_inverse ? 1 : -1) * i * pi / (n / 2);
        twiddle[i] = {std::cos(phi), std::sin(phi)};
      }
    }

    ~Onedim()
    {
      delete twiddle;
      delete working;
    }

    void transform(T* src, T* dst)
    {
      copy(src, 2 * n, dst);
      for(Int dft_size = 1; dft_size < n; dft_size *= 2)
      {
        swap(dst, working);

        typedef complex_format::Scal<Scalar<T>> CF;
        Int twiddle_stride = n / 2 / dft_size;

        for(Int i = 0; i < n / 2; i += dft_size)
        {
          Int src_i = i;
          Int dst_i = 2 * i;
          Int twiddle_i = 0;
          for(; src_i < i + dft_size;)
          {
            auto a = CF::load(working + src_i * CF::stride, 0);
            auto b = CF::load(working + (src_i + n / 2) * CF::stride, 0);
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
  Onedim onedim;

  ReferenceFft(const std::vector<Int>& size)
    : InterleavedWrapperBase<T, false, is_inverse_>(size), onedim(size[0]) { }

  void transform() { onedim.transform(this->src, this->dst); }
};

template<typename Fft>
void bench(const std::vector<Int>& size, double requested_operations)
{
  Int n = product(size);
  typedef typename Fft::value_type T;
  Fft fft(size);
  T* src = alloc_complex_array<T>(n);
  T* dst = alloc_complex_array<T>(n);
  
  for(Int i = 0; i < n * 2; i++) src[i] = 0.0f;

  double const_part = Fft::is_real ? 2.5 : 5.0;
  auto iter = max<uint64_t>(requested_operations / (const_part * n * log2(n)), 1);
  auto operations = iter * (const_part * n * log2(n));

  double t0 = get_time();
  for(int i = 0; i < iter; i++) fft.transform();
  double t1 = get_time(); 

  printf("%f gflops\n", operations * 1e-9 / (t1 - t0));
}

template<typename Fft0, typename Fft1>
typename Fft0::value_type compare(const std::vector<Int>& size)
{
  static_assert(Fft0::is_inverse == Fft1::is_inverse, "");
  typedef typename Fft0::value_type T;
  Int n = product(size);
  Fft0 fft0(size);
  Fft1 fft1(size);
  T* src = alloc_complex_array<T>(n);
  T* dst0 = alloc_complex_array<T>(n);
  T* dst1 = alloc_complex_array<T>(n);
  
  std::mt19937 mt;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for(Int i = 0; i < n * 2; i++) src[i] = dist(mt);
  if(Fft0::is_real || Fft1::is_real)
  {
    if(Fft0::is_inverse)
    {
      auto re = create_view(src, size, 0);
      copy_symmetric_view<false>(re.sub(0, 0, size[0] / 2 + 1), re);

      auto im = create_view(src + n, size, 0);
      copy_symmetric_view<true>(im.sub(0, 0, size[0] / 2 + 1 ), im);
      src[n + n / 2] = 0;
      src[n] = 0;
    }
    else
      for(Int i = n; i < n * 2; i++) src[i] = T(0);
  }

  fft0.set_input(src);
  fft1.set_input(src);

  fft0.transform();
  fft1.transform();

  fft0.get_output(dst0);
  fft1.get_output(dst1);

  dump(src, 2 * n, "src.float64");
  dump(dst0, 2 * n, "dst0.float64");
  dump(dst1, 2 * n, "dst1.float64");

  auto sum_sumsq = T(0);
  auto diff_sumsq = T(0);
  for(Int i = 0; i < 2 * n; i++)
  {
    sum_sumsq += sq(dst0[i]);
    sum_sumsq += sq(dst1[i]);
    diff_sumsq += sq(dst1[i] - dst0[i]);
  }

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

template<typename Fft>
void test(const std::vector<Int>& size)
{
  //TODO: Use long double for ReferenceFft
  printf("difference %e\n",
		 compare<ReferenceFft<double, Fft::is_inverse>, Fft>(size));
}

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

template<typename Fft>
void test_or_bench3(const Options& opt)
{
  std::vector<Int> size;
  for(Int i = 1; i < opt.positional.size(); i++)
  {    
    Int log2n;
    std::stringstream(opt.positional[1]) >> log2n;
    size.push_back(1 << log2n);
  }

  if(opt.flags.count("-b"))
    bench<Fft>(size, 1e11);
  else
    test<Fft>(size);
}

template<bool is_real, bool is_inverse>
void test_or_bench2(const Options& opt)
{
  if(opt.positional[0] == "fft")
    test_or_bench3<TestWrapper<V, CfT, is_real, is_inverse>>(opt);
#ifdef HAVE_FFTW
  else if(opt.positional[0] == "fftw")
    test_or_bench3<FftwTestWrapper<float, is_real, is_inverse>>(opt);
#endif
  else
    abort();
}

template<bool is_real>
void test_or_bench1(const Options& opt)
{
  if(opt.flags.count("-i"))
    test_or_bench2<is_real, true>(opt);
  else
    test_or_bench2<is_real, false>(opt);
}

void test_or_bench0(const Options& opt)
{
  if(opt.flags.count("-r"))
    test_or_bench1<true>(opt);
  else
    test_or_bench1<false>(opt);
}

int main(int argc, char** argv)
{
  Options opt = parse_options(argc, argv);
  if(opt.positional.size() != 2) abort();
  test_or_bench0(opt);
  return 0;
}
